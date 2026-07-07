"""Book Alchemy phase state machine.

The worker repeatedly calls :func:`run_step`, which performs exactly ONE bounded
unit of work for a project (parse, analyze one chunk, build the curriculum,
script one session, render one session's audio, finalize) and persists progress
to Postgres. This guarantees:

  * no single AI request ever processes a whole book / all scripts / all audio,
  * crash-resumability (state is in the DB; each phase is idempotent),
  * granular progress ("Processing Session 4 of 15").

Phase cursor (``book_alchemy_projects.phase``):
    uploaded -> analyze -> curriculum -> sessions -> audio -> finalize -> complete
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from .. import voices as voice_store
from . import db, prompts
from .audio import render_script_to_audio
from .chunking import build_chunks
from .deepseek_client import DeepSeekClient
from .parsers import ScannedPdfError, UnsupportedSourceError, parse_source
from .storage import session_audio_path
from .validation import validate_script

log = logging.getLogger("book_alchemy.pipeline")

KINDS = ["concept", "definition", "framework", "example", "conclusion"]
MAX_REGEN = 2                 # re-script attempts before flagging a session
MAX_CHUNKS_PER_SESSION = 10   # cap source excerpts fed into one script prompt
MAX_CONCEPTS_FOR_CURRICULUM = 60   # per kind, after de-duplication


class TerminalError(Exception):
    """A non-recoverable error; the project should be marked failed."""


async def run_step(project: dict, client: Optional[DeepSeekClient] = None) -> bool:
    """Advance one project by a single unit. Returns True if more work remains."""
    phase = project["phase"]
    if phase in ("complete", "failed"):
        return False

    client = client or DeepSeekClient.from_env()
    pid = int(project["id"])

    if phase in ("uploaded", "parse"):
        await _phase_parse(project, client)
    elif phase == "analyze":
        await _phase_analyze(project, client)
    elif phase == "curriculum":
        await _phase_curriculum(project, client)
    elif phase == "sessions":
        await _phase_sessions(project, client)
    elif phase == "audio":
        await _phase_audio(project)
    elif phase == "finalize":
        await _phase_finalize(project)
    else:
        raise TerminalError(f"Unknown phase: {phase!r}")

    refreshed = await db.get_project(pid)
    return bool(refreshed and refreshed["phase"] not in ("complete", "failed"))


# --------------------------------------------------------------- phase: parse
async def _phase_parse(project: dict, client: DeepSeekClient) -> None:
    pid = int(project["id"])
    # Idempotent: if chunks already exist we already parsed; just advance.
    if await db.count_chunks(pid) > 0:
        await db.set_project(pid, phase="analyze", analyze_cursor=0, stage="Analyzing with DeepSeek", progress=10)
        return

    await db.set_project(pid, stage="Extracting content", progress=4)
    fmt = project["source_format"]
    source_path = project.get("source_path")
    try:
        doc = parse_source(
            source_format=fmt,
            path=source_path,
            url=project.get("source_url"),
            title_hint=project.get("name"),
        )
    except ScannedPdfError:
        # Image/scanned PDF: recover the text with the existing OCR pipeline
        # (render -> Tesseract -> DeepSeek clean), then continue as plain text.
        doc = await _ocr_pdf_to_doc(project, source_path)
    except UnsupportedSourceError as exc:
        raise TerminalError(str(exc)) from exc

    chunks = build_chunks(doc)
    if not chunks:
        raise TerminalError("No readable text could be extracted from the source.")

    await db.set_project(pid, stage="Chunking content", progress=8)
    await db.insert_chunks(pid, chunks)

    # Derive a clean course title — uploads can have very long or mis-encoded
    # filenames. Done once, early, so the dashboard (and the download zip name)
    # shows a clean title throughout the long analysis/audio phases.
    sample = "\n\n".join(c["text"] for c in chunks[:2])[:2000]
    clean = await _clean_title(client, project.get("name") or doc.title, sample)

    await db.set_project(
        pid, name=clean, phase="analyze", analyze_cursor=0,
        stage="Analyzing with DeepSeek", progress=10,
    )


async def _ocr_pdf_to_doc(project: dict, source_path: Optional[str]):
    """Run the existing SpokenVerse OCR pipeline on a scanned PDF and return a
    ParsedDoc of the recovered text. Requires Tesseract (a SpokenVerse system
    dependency) and DeepSeek; failures surface as a clear TerminalError."""
    pid = int(project["id"])
    if not source_path:
        raise TerminalError("Scanned PDF but no source file is available for OCR.")

    await db.set_project(pid, stage="Running OCR (scanned PDF)", progress=6)
    try:
        from phansora.products.spokenverse.txt_to_voice.pdf_pipeline import PdfConverter, PdfToTxtConfig  # lazy

        pdf_path = Path(source_path)
        out_txt = pdf_path.with_suffix(".ocr.txt")
        cfg = PdfToTxtConfig(keep_page_breaks=False, to_chapters=False)
        await PdfConverter(cfg).convert_pdf_to_txt_async(pdf_path, out_txt)

        text = out_txt.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            raise TerminalError("OCR produced no readable text from the scanned PDF.")
        # Re-use the plain-text parser on the recovered text.
        return parse_source(source_format="text", path=str(out_txt), title_hint=project.get("name"))
    except TerminalError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise TerminalError(f"OCR failed for scanned PDF: {exc}") from exc


# --------------------------------------------------------------- phase: analyze
async def _phase_analyze(project: dict, client: DeepSeekClient) -> None:
    pid = int(project["id"])
    total = await db.count_chunks(pid)
    cursor = int(project["analyze_cursor"])
    if cursor >= total:
        await db.set_project(pid, phase="curriculum", stage="Building curriculum", progress=50)
        return

    chunk = await db.get_chunk_by_ordinal(pid, cursor)
    if chunk is None:  # gap safety
        await db.set_project(pid, analyze_cursor=cursor + 1)
        return

    extracted = await client.chat_json(
        system=prompts.ANALYZE_SYSTEM,
        user=prompts.analyze_user(chunk["text"], chapter=chunk["chapter"]),
        max_output_tokens=2000,
    )
    concepts = _concepts_from_extraction(extracted, source_chunk_id=int(chunk["id"]))
    await db.insert_concepts(pid, concepts)

    done = cursor + 1
    progress = 10 + int(40 * done / max(1, total))
    await db.set_project(
        pid, analyze_cursor=done,
        stage=f"Extracting concepts ({done}/{total})", progress=min(progress, 49),
    )


def _concepts_from_extraction(extracted: Any, *, source_chunk_id: int) -> list[dict]:
    out: list[dict] = []
    if not isinstance(extracted, dict):
        return out
    for kind in KINDS:
        items = extracted.get(kind + "s") or extracted.get(kind) or []
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                title = str(item.get("title") or "").strip()
                body = str(item.get("body") or "").strip()
            else:
                title, body = "", str(item).strip()
            if not (title or body):
                continue
            out.append({
                "kind": kind,
                "content": {"title": title, "body": body},
                "source_chunk_ids": [source_chunk_id],
            })
    return out


# --------------------------------------------------------------- phase: curriculum
async def _phase_curriculum(project: dict, client: DeepSeekClient) -> None:
    pid = int(project["id"])

    # Idempotent: if sessions already exist, advance.
    if await db.get_sessions(pid):
        await db.set_project(pid, phase="sessions", stage="Creating sessions", progress=55)
        return

    concept_rows = await db.get_concepts(pid)
    if not concept_rows:
        raise TerminalError("No concepts could be extracted from the source material.")

    title_to_chunks: dict[str, list[int]] = {}
    # Dedupe concepts by (kind, lowercased title), merging their source chunk
    # ids. Big books repeat the same concept across many chunks; without this the
    # curriculum prompt (and the model's echoed output) balloon and truncate.
    merged: dict[tuple[str, str], dict] = {}
    for row in concept_rows:
        content = _as_dict(row["content"])
        title = str(content.get("title") or "").strip()
        body = str(content.get("body") or "").strip()
        if title:
            title_to_chunks.setdefault(title.lower(), [])
            title_to_chunks[title.lower()].extend(list(row["source_chunk_ids"] or []))
        key = (row["kind"], title.lower())
        existing = merged.get(key)
        if existing:
            if not existing["body"] and body:
                existing["body"] = body[:200]
        else:
            merged[key] = {"kind": row["kind"], "title": title, "body": body[:200]}

    grouped: dict[str, list[dict]] = {k: [] for k in KINDS}
    for item in merged.values():
        grouped[item["kind"]].append({"title": item["title"], "body": item["body"]})
    grouped = {k: v[:MAX_CONCEPTS_FOR_CURRICULUM] for k, v in grouped.items() if v}

    plan = await client.chat_json(
        system=prompts.CURRICULUM_SYSTEM,
        user=prompts.curriculum_user(grouped),
        max_output_tokens=8000,
    )
    sessions = plan.get("sessions") if isinstance(plan, dict) else None
    if not sessions:
        raise TerminalError("Could not generate a curriculum from the source material.")

    for i, s in enumerate(sessions, start=1):
        if not isinstance(s, dict):
            continue
        concept_titles = [str(t).lower() for t in (s.get("concept_titles") or [])]
        chunk_ids: list[int] = []
        for t in concept_titles:
            chunk_ids.extend(title_to_chunks.get(t, []))
        chunk_ids = _dedupe(chunk_ids)[:MAX_CHUNKS_PER_SESSION]
        await db.create_session(
            project_id=pid,
            ordinal=int(s.get("ordinal") or i),
            title=str(s.get("title") or f"Session {i}"),
            summary=str(s.get("summary") or ""),
            outline=s.get("outline") or [],
            source_chunk_ids=chunk_ids,
        )

    await db.set_project(
        pid, curriculum=plan, phase="sessions",
        stage="Creating sessions", progress=55,
    )


# --------------------------------------------------------------- phase: sessions (script + validate)
async def _phase_sessions(project: dict, client: DeepSeekClient) -> None:
    pid = int(project["id"])
    sessions = await db.get_sessions(pid)
    total = len(sessions)
    sess = await db.next_session_needing(pid, ["pending"])
    if sess is None:
        await db.set_project(pid, phase="audio", stage="Generating audio", progress=80)
        return

    done = sum(1 for s in sessions if s["status"] not in ("pending",))
    await db.set_project(
        pid, stage=f"Creating session {sess['ordinal']} of {total}",
        progress=min(55 + int(23 * done / max(1, total)), 79),
    )

    chunk_ids = list(sess["source_chunk_ids"] or [])
    chunks = await db.get_chunks_by_ids(pid, chunk_ids)
    chunk_dicts = [_chunk_dict(c) for c in chunks]
    outline = _as_list(sess["outline"])

    if not chunk_dicts:
        # No grounded source mapped to this session: do not invent content.
        # Mark it complete-but-flagged with an empty script so the audio phase
        # skips it and the UI surfaces why.
        await db.set_session(
            sess["id"], status="complete", validation_status="flagged", script="",
            validation_notes={"notes": "No source excerpts mapped; skipped to avoid unsupported content."},
        )
        return

    script = ""
    validation = {"supported": False, "flagged": [], "notes": ""}
    regen = 0
    while regen <= MAX_REGEN:
        script = await client.chat(
            system=prompts.SCRIPT_SYSTEM,
            user=prompts.script_user(sess["title"], outline, chunk_dicts),
            max_output_tokens=4000,
        )
        validation = await validate_script(client, script=script, chunks=chunk_dicts)
        if validation["supported"]:
            break
        regen += 1

    await db.set_session(
        sess["id"],
        script=script,
        status="validated",
        validation_status="passed" if validation["supported"] else "flagged",
        validation_notes=validation,
        regen_count=regen,
    )


# --------------------------------------------------------------- phase: audio
async def _phase_audio(project: dict) -> None:
    pid = int(project["id"])
    sessions = await db.get_sessions(pid)
    total = len(sessions)
    sess = await db.next_session_needing(pid, ["validated"])
    if sess is None:
        await db.set_project(pid, phase="finalize", stage="Finalizing assets", progress=98)
        return

    if not (sess["script"] or "").strip():
        await db.set_session(sess["id"], status="complete", audio_seconds=0)
        return

    done = sum(1 for s in sessions if s["status"] == "complete")
    await db.set_project(
        pid, stage=f"Processing session {sess['ordinal']} of {total}",
        progress=min(80 + int(18 * done / max(1, total)), 97),
    )

    options = _as_dict(project.get("options"))
    voice = str(options.get("voice") or "default")
    # A saved cloned voice is stored per-user; the engine clones from a file path,
    # so resolve the voice id -> its reference clip (a bare id would silently fall
    # back to the default voice). Mirrors the txt-to-audio server's resolution.
    if voice and voice != "default":
        clip = voice_store.voice_path(str(project["user_id"]), voice)
        if clip is not None:
            voice = str(clip)
    out_path: Path = session_audio_path(project["user_id"], pid, sess["ordinal"], "mp3")
    seconds = await render_script_to_audio(script=sess["script"], out_path=out_path, voice=voice)

    await db.set_session(
        sess["id"], status="complete", audio_path=str(out_path),
        audio_seconds=seconds, generated_at=_now(),
    )


# --------------------------------------------------------------- phase: finalize
async def _phase_finalize(project: dict) -> None:
    pid = int(project["id"])
    sessions = await db.get_sessions(pid)
    total_seconds = sum(int(s["audio_seconds"] or 0) for s in sessions)
    any_flagged = any(s["validation_status"] == "flagged" for s in sessions)
    await db.set_project(
        pid,
        total_audio_seconds=total_seconds,
        validation_status="flagged" if any_flagged else "passed",
        status="complete", phase="complete", stage="Complete", progress=100,
    )


# --------------------------------------------------------------- helpers
async def _clean_title(client: DeepSeekClient, raw_title: Optional[str], sample: str) -> str:
    """Ask DeepSeek for a clean, concise course title; fall back to a sanitized
    version of the raw title if the call fails or returns nothing."""
    fallback = _sanitize_title(raw_title) or "Untitled Course"
    try:
        out = await client.chat(
            system=prompts.TITLE_SYSTEM,
            user=prompts.title_user(str(raw_title or "Untitled"), sample),
            max_output_tokens=60,
        )
        cleaned = _sanitize_title(out)
        return cleaned or fallback
    except Exception:  # noqa: BLE001
        return fallback


def _sanitize_title(text: Optional[str]) -> str:
    """Strip control/non-printable chars, surrounding quotes, stray prefixes,
    collapse whitespace, and cap length."""
    import re
    import unicodedata

    s = unicodedata.normalize("NFC", str(text or ""))
    s = "".join(ch for ch in s if ch == " " or unicodedata.category(ch)[0] != "C")
    s = re.sub(r"\s+", " ", s).strip().strip("\"'").strip()
    s = re.sub(r"^(book alchemy|course|title)\s*[:\-–]\s*", "", s, flags=re.I).strip()
    return s[:120]


def _chunk_dict(row: Any) -> dict:
    return {
        "id": int(row["id"]),
        "text": row["text"],
        "chapter": row["chapter"],
        "section": row["section"],
        "page_start": row["page_start"],
        "page_end": row["page_end"],
    }


def _as_dict(value: Any) -> dict:
    import json
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _as_list(value: Any) -> list:
    import json
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:  # noqa: BLE001
            return []
    return []


def _dedupe(items: list[int]) -> list[int]:
    seen, out = set(), []
    for i in items:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)
