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
import math
from pathlib import Path
from typing import Any, Optional

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

# --- Listening budget ---------------------------------------------------------
# Book Alchemy adapts a work; it does not expand it. Lesson count is therefore a
# function of how much source there is — never of how many topics it touches. A
# source that fits one comfortable sitting becomes exactly one lesson, and that
# decision is made here in code rather than left to the model.
WORDS_PER_MINUTE = 150           # narration pace
TARGET_LESSON_MINUTES = 14       # comfortable default sitting
MAX_LESSON_MINUTES = 20          # a part longer than this gets split
MIN_LESSON_MINUTES = 8           # below this, merging beats splitting
TARGET_LESSON_WORDS = WORDS_PER_MINUTE * TARGET_LESSON_MINUTES   # 2100
MAX_LESSON_WORDS = WORDS_PER_MINUTE * MAX_LESSON_MINUTES         # 3000
MIN_LESSON_WORDS = WORDS_PER_MINUTE * MIN_LESSON_MINUTES         # 1200

# Narration length relative to its own source segments: parity, not expansion.
# Under the floor means information was dropped; over the ceiling means padding.
DENSITY_MIN = 0.85
DENSITY_MAX = 1.15

MAX_TOPICS_PER_SEGMENT = 5        # detail carried into the segmentation prompt
MAX_SEGMENT_DIGEST_CHARS = 60_000 # keep that prompt inside the context window


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
        log.exception("OCR failed for scanned PDF (project %s)", pid)
        raise TerminalError(_ocr_error_message(exc)) from exc


def _ocr_error_message(exc: Exception) -> str:
    """Turn an OCR failure into something the reader can act on.

    Tesseract is a SYSTEM dependency (the `pytesseract` wheel is only a wrapper), so a host
    that skipped it fails at the first scanned PDF with an internal message — "tesseract is
    not installed or it's not in your PATH. See README file" — which tells the reader of a
    book nothing. Name the two setup faults explicitly instead; anything else keeps the
    original text, which is usually a genuine per-document problem.
    """
    detail = str(exc).strip()
    name = type(exc).__name__

    if name == "TesseractNotFoundError" or "not installed or it's not in your PATH" in detail:
        return (
            "This PDF is scanned, and text recognition isn't available on the server right "
            "now. Please contact support — the Tesseract OCR engine needs to be installed."
        )
    # Engine present but no language data: tessdata ships empty on some distros.
    if "traineddata" in detail or "Error opening data file" in detail:
        return (
            "This PDF is scanned, and the server is missing the OCR language data needed to "
            "read it. Please contact support — the Tesseract English language pack is missing."
        )
    return f"OCR failed for scanned PDF: {detail}"


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

    try:
        extracted = await client.chat_json(
            system=prompts.ANALYZE_SYSTEM,
            user=prompts.analyze_user(chunk["text"], chapter=chunk["chapter"]),
            max_output_tokens=2000,
        )
        concepts = _concepts_from_extraction(extracted, source_chunk_id=int(chunk["id"]))
        await db.insert_concepts(pid, concepts)
    except Exception as exc:  # noqa: BLE001
        # One chunk that won't parse (e.g. the model emitted JSON too long for the
        # token budget and it truncated) must not fail the whole course. Skip it
        # and advance — the concepts from every other chunk still build the
        # curriculum, and if *every* chunk fails, _phase_curriculum raises a
        # TerminalError on zero concepts.
        log.warning(
            "Project %s: skipping chunk %s (ordinal) — concept extraction failed: %s",
            pid, cursor, exc,
        )

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
    """Cut the work into the fewest comfortable parts.

    The result is a *contiguous partition* of the source segments: lessons run in
    source order, every segment lands in exactly one lesson, and none is dropped.
    That is what stops the same material being narrated twice and stops source
    text going missing because no concept title happened to reference it.
    """
    pid = int(project["id"])

    # Idempotent: if sessions already exist, advance.
    if await db.get_sessions(pid):
        await db.set_project(pid, phase="sessions", stage="Creating sessions", progress=55)
        return

    chunks = await db.get_all_chunks(pid)
    if not chunks:
        raise TerminalError("No readable source segments were found.")

    digests = await _chunk_digests(pid, chunks)
    source_words = sum(d["words"] for d in digests)
    min_lessons, suggested, max_lessons = _lesson_budget(source_words)

    if max_lessons == 1:
        # Short enough for one sitting. Decided here, not by the model: a source
        # that covers several topics is still one lesson.
        log.info("Project %s: %s source words -> single lesson", pid, source_words)
        entries = [{
            "start": 0,
            "title": str(project.get("name") or "").strip() or "Full text",
            "summary": "",
            "topics": [t for d in digests for t in d["topics"]],
        }]
    else:
        raw = await client.chat_json(
            system=prompts.SEGMENT_SYSTEM,
            user=prompts.segment_user(
                _trim_digests(digests),
                min_lessons=min_lessons,
                suggested_lessons=suggested,
                max_lessons=max_lessons,
                max_lesson_words=MAX_LESSON_WORDS,
            ),
            max_output_tokens=8000,
        )
        entries = _entries_from_plan(raw, segment_count=len(digests), max_lessons=max_lessons)

    lessons = _resolve_lessons(entries, digests)
    log.info(
        "Project %s: %s source words -> %s lesson(s) (suggested %s, max %s)",
        pid, source_words, len(lessons), suggested, max_lessons,
    )

    for lesson in lessons:
        await db.create_session(
            project_id=pid,
            ordinal=lesson["ordinal"],
            title=lesson["title"],
            summary=lesson["summary"],
            outline=lesson["topics"],
            source_chunk_ids=[
                int(chunks[i]["id"]) for i in range(lesson["start"], lesson["end"] + 1)
            ],
        )

    plan = {
        "work_title": project.get("name"),
        "source_words": source_words,
        "lesson_count": len(lessons),
        "sessions": [
            {
                "ordinal": l["ordinal"],
                "title": l["title"],
                "summary": l["summary"],
                "topics": l["topics"],
                "segment_range": [l["start"], l["end"]],
                "source_words": l["words"],
            }
            for l in lessons
        ],
    }
    await db.set_project(
        pid, curriculum=plan, phase="sessions",
        stage="Creating sessions", progress=55,
    )


def _lesson_budget(source_words: int) -> tuple[int, int, int]:
    """(minimum, suggested, maximum) lesson count for a source of this length.

    A source that fits one comfortable sitting returns (1, 1, 1) — the caller
    then skips the segmentation model entirely, so a short work can never be
    split into a multi-lesson "course"."""
    if source_words <= MAX_LESSON_WORDS:
        return 1, 1, 1
    minimum = max(1, math.ceil(source_words / MAX_LESSON_WORDS))
    maximum = max(minimum, math.ceil(source_words / MIN_LESSON_WORDS))
    suggested = min(maximum, max(minimum, math.ceil(source_words / TARGET_LESSON_WORDS)))
    return minimum, suggested, maximum


async def _chunk_digests(project_id: int, chunks: list[Any]) -> list[dict]:
    """One planning line per source segment: where it sits, how long it is, and
    what the analyze phase indexed in it. Used only to place lesson boundaries."""
    by_chunk: dict[int, list[str]] = {}
    for row in await db.get_concepts(project_id):
        title = str(_as_dict(row["content"]).get("title") or "").strip()
        if not title:
            continue
        for cid in (row["source_chunk_ids"] or []):
            topics = by_chunk.setdefault(int(cid), [])
            if title not in topics and len(topics) < MAX_TOPICS_PER_SEGMENT:
                topics.append(title)

    return [
        {
            "ordinal": i,
            "chapter": c["chapter"],
            "words": _word_count(c["text"]),
            "topics": by_chunk.get(int(c["id"]), []),
        }
        for i, c in enumerate(chunks)
    ]


def _trim_digests(digests: list[dict]) -> list[dict]:
    """Shrink the planning digest until the segmentation prompt fits in context.

    Long books have too many segments to describe in full. Detail is dropped
    before segments are — boundaries can still be placed on chapter and length
    alone, but a missing segment would silently lose source text."""
    trimmed = digests
    for limit in (MAX_TOPICS_PER_SEGMENT, 3, 2, 1, 0):
        trimmed = [{**d, "topics": d["topics"][:limit]} for d in digests]
        if sum(len(str(d["topics"])) + 40 for d in trimmed) <= MAX_SEGMENT_DIGEST_CHARS:
            break
    return trimmed


def _entries_from_plan(raw: Any, *, segment_count: int, max_lessons: int) -> list[dict]:
    """Pull lesson start-points out of the model's plan, sanitized.

    Only ``start_segment`` is trusted as structure; the ranges themselves are
    derived in :func:`_resolve_lessons` so coverage can be guaranteed."""
    entries: list[dict] = []
    sessions = raw.get("sessions") if isinstance(raw, dict) else None
    for s in sessions or []:
        if not isinstance(s, dict):
            continue
        try:
            start = int(s.get("start_segment"))
        except (TypeError, ValueError):
            continue
        entries.append({
            "start": max(0, min(segment_count - 1, start)),
            "title": str(s.get("title") or "").strip(),
            "summary": str(s.get("summary") or "").strip(),
            "topics": [str(t).strip() for t in (s.get("topics") or []) if str(t).strip()],
        })

    entries.sort(key=lambda e: e["start"])
    deduped: list[dict] = []
    seen: set[int] = set()
    for entry in entries:
        if entry["start"] in seen:
            continue
        seen.add(entry["start"])
        deduped.append(entry)

    if not deduped:
        deduped = [{"start": 0, "title": "", "summary": "", "topics": []}]
    deduped[0]["start"] = 0          # the work always starts at its first segment
    return deduped[:max_lessons]


def _resolve_lessons(entries: list[dict], digests: list[dict]) -> list[dict]:
    """Turn start-points into a complete, non-overlapping cover of the segments.

    Each lesson runs from its own start to the segment before the next one, the
    last runs to the end, and any lesson longer than ``MAX_LESSON_WORDS`` is
    split on segment boundaries (which also keeps each script inside the model's
    output-token ceiling)."""
    total = len(digests)
    ranges: list[dict] = []
    for i, entry in enumerate(entries):
        start = entry["start"]
        end = (entries[i + 1]["start"] - 1) if i + 1 < len(entries) else total - 1
        if end < start:                       # degenerate ordering; fold away
            continue
        ranges.append({**entry, "end": end})

    if not ranges:
        ranges = [{"start": 0, "end": total - 1, "title": "", "summary": "", "topics": []}]
    ranges[0]["start"] = 0
    ranges[-1]["end"] = total - 1

    lessons: list[dict] = []
    for r in ranges:
        for piece in _split_to_length(r, digests):
            lessons.append(piece)

    for i, lesson in enumerate(lessons, start=1):
        lesson["ordinal"] = i
        lesson["words"] = _range_words(digests, lesson["start"], lesson["end"])
        if not lesson["title"]:
            lesson["title"] = f"Part {i}"
        if not lesson["topics"]:
            lesson["topics"] = [
                t for d in digests[lesson["start"]: lesson["end"] + 1] for t in d["topics"]
            ]
    return lessons


def _split_to_length(rng: dict, digests: list[dict]) -> list[dict]:
    """Split one range into contiguous parts that each fit MAX_LESSON_WORDS."""
    words = _range_words(digests, rng["start"], rng["end"])
    span = rng["end"] - rng["start"] + 1
    if words <= MAX_LESSON_WORDS or span <= 1:
        return [dict(rng)]

    parts = max(2, math.ceil(words / MAX_LESSON_WORDS))
    per_part = words / parts
    out: list[dict] = []
    start = rng["start"]
    running = 0
    for i in range(rng["start"], rng["end"] + 1):
        running += digests[i]["words"]
        last_segment = i == rng["end"]
        parts_left = parts - len(out)
        segments_left = rng["end"] - i
        # Close this part once it has its share, but never leave a later part empty.
        if last_segment or (running >= per_part and parts_left > 1 and segments_left >= parts_left - 1):
            out.append({
                "start": start,
                "end": i,
                "title": rng["title"],
                "summary": rng["summary"],
                "topics": [],       # refilled from the segments actually covered
            })
            start, running = i + 1, 0
            if len(out) == parts and not last_segment:
                out[-1]["end"] = rng["end"]
                break

    for i, piece in enumerate(out):
        if i and piece["title"]:
            piece["title"] = f"{piece['title']} (part {i + 1})"
    return out


def _range_words(digests: list[dict], start: int, end: int) -> int:
    return sum(d["words"] for d in digests[start: end + 1])


def _word_count(text: Any) -> int:
    return len(str(text or "").split())


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

    # Density target: the narration carries this part's information at roughly
    # the length the author used for it. Coming in short means content was
    # dropped; running long means filler was added.
    source_words = sum(_word_count(c["text"]) for c in chunk_dicts)
    min_words = max(60, int(source_words * DENSITY_MIN))
    max_words = max(min_words + 40, int(source_words * DENSITY_MAX))
    # ~1.4 tokens per word, plus headroom, capped at the DeepSeek output ceiling.
    token_budget = min(8000, int(max_words * 1.7) + 400)

    script = ""
    validation = {"supported": False, "flagged": [], "notes": ""}
    feedback: Optional[list] = None
    regen = 0
    for attempt in range(MAX_REGEN + 1):
        regen = attempt
        script = await client.chat(
            system=prompts.SCRIPT_SYSTEM,
            user=prompts.script_user(
                sess["title"], outline, chunk_dicts,
                source_words=source_words,
                min_words=min_words,
                max_words=max_words,
                feedback=feedback,
            ),
            max_output_tokens=token_budget,
            # A retry must not reproduce the rejected script verbatim. The
            # feedback already changes the prompt; a little temperature helps the
            # model leave a phrasing it has settled on. (At temperature 0 with an
            # unchanged prompt, the old loop regenerated the identical script.)
            temperature=0.0 if attempt == 0 else 0.3,
        )
        validation = await validate_script(client, script=script, chunks=chunk_dicts)
        if script.strip() and validation["supported"]:
            break
        feedback = validation.get("flagged") or []

    written = _word_count(script)
    if script.strip() and not (min_words <= written <= max_words):
        log.info(
            "Project %s session %s: %s narration words against a %s-%s target "
            "(%s source words)",
            pid, sess["ordinal"], written, min_words, max_words, source_words,
        )

    if not script.strip():
        # The model returned nothing on every attempt. Previously this was written out as a
        # blank script and only noticed when the finished course had no audio.
        log.warning(
            "Project %s session %s: model returned an empty script after %s attempt(s) "
            "(budget %s tokens, %s source words)",
            pid, sess["ordinal"], regen + 1, token_budget, source_words,
        )

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
        # A blank script silently yields a course with no audio, which is the hardest
        # possible failure to diagnose from the outside — say so.
        log.warning(
            "Project %s session %s: empty script, no audio rendered (validation=%s)",
            pid, sess["ordinal"], sess.get("validation_status"),
        )
        await db.set_session(sess["id"], status="complete", audio_seconds=0)
        return

    done = sum(1 for s in sessions if s["status"] == "complete")
    await db.set_project(
        pid, stage=f"Processing session {sess['ordinal']} of {total}",
        progress=min(80 + int(18 * done / max(1, total)), 97),
    )

    options = _as_dict(project.get("options"))
    voice = str(options.get("voice") or "default")
    # Pass the raw voice id + user_id to the shared txt-to-audio endpoint; it
    # resolves a cloned voice id -> its reference clip and its stored transcript
    # (ref_text -> prompt_text). (Resolving to a path here would make the endpoint
    # fall back to the default voice.)
    out_path: Path = session_audio_path(project["user_id"], pid, sess["ordinal"], "mp3")
    seconds = await render_script_to_audio(
        script=sess["script"], out_path=out_path,
        user_id=project["user_id"], voice=voice,
    )

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


def _now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)
