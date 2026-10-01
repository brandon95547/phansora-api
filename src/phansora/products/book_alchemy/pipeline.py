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

There is a second, unrelated sense of "phase" in this product. The cursor above
is the internal PIPELINE phase. A DELIVERY phase (``book_alchemy_phases``, see
phases.py) is a batch of lessons the listener asks for one at a time, so a
twenty-hour course arrives in sittings rather than in one twenty-hour run. When
the worker finishes a delivery phase and the listener has not asked for the next,
the cursor parks at ``awaiting_user`` until they do.

Delivery phases change nothing about what the model is asked. Each script is
conditioned on ``_prior_coverage``, which reads only the titles and outlines
written for EVERY lesson back at the curriculum step — so as long as lessons are
still written in ascending ordinal order, splitting them into batches leaves
every prompt byte-for-byte identical.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any, NamedTuple, Optional

from . import db, prompts
from .audio import VoiceServiceBusy, VoiceServiceError, render_script_to_audio
from .chunking import build_chunks
from .deepseek_client import DeepSeekClient
from .parsers import ParsedDoc, ScannedPdfError, UnsupportedSourceError, parse_source
from .sentences import complete_ending
from .storage import session_audio_path
from .validation import scrub_apparatus, validate_script

log = logging.getLogger("book_alchemy.pipeline")

KINDS = ["concept", "definition", "framework", "example", "conclusion"]
MAX_REGEN = 2                 # re-script attempts before flagging a session
_BUSY_RETRIES = 20            # how many times a lesson waits out a busy voice service
_BUSY_MAX_WAIT_S = 60         # ceiling on one back-off, so a long Retry-After cannot stall a job
# Recording attempts per lesson for a failure that is NOT "busy". One bad render used to
# fail the whole book, with no retry at all in its first phase; a second try clears a
# one-off engine fault, and a lesson that fails every time still fails the book.
_LESSON_ATTEMPTS = max(1, int(os.getenv("BOOK_ALCHEMY_LESSON_ATTEMPTS", "3")))
_LESSON_RETRY_WAIT_S = 30

# --- Listening budget ---------------------------------------------------------
# Book Alchemy adapts a work; it does not expand it. Lesson count is therefore a
# function of how much source there is — never of how many topics it touches. A
# source that fits one comfortable sitting becomes exactly one lesson, and that
# decision is made here in code rather than left to the model.
WORDS_PER_MINUTE = 150           # narration pace
# A lesson runs at least MIN minutes wherever the source has that much to teach,
# and then stops at the first place the material itself stops — see _plan_cuts,
# which is where these three are enforced. Env-overridable because they are
# planning figures: what a lesson actually runs depends on how the writer paces
# it, and that can only be measured on finished courses.
MIN_LESSON_MINUTES = max(1, int(os.getenv("BOOK_ALCHEMY_MIN_LESSON_MINUTES", "10")))
# What the segmentation model is told to expect. Kept just above the floor, so
# the parts it proposes are already about the size a lesson is allowed to be.
TARGET_LESSON_MINUTES = max(
    MIN_LESSON_MINUTES, int(os.getenv("BOOK_ALCHEMY_TARGET_LESSON_MINUTES", "12"))
)
# A part longer than this gets split.
MAX_LESSON_MINUTES = max(
    TARGET_LESSON_MINUTES, int(os.getenv("BOOK_ALCHEMY_MAX_LESSON_MINUTES", "20"))
)
TARGET_LESSON_WORDS = WORDS_PER_MINUTE * TARGET_LESSON_MINUTES   # 1800
MAX_LESSON_WORDS = WORDS_PER_MINUTE * MAX_LESSON_MINUTES         # 3000
MIN_LESSON_WORDS = WORDS_PER_MINUTE * MIN_LESSON_MINUTES         # 1500

# --- How long a lesson runs ---------------------------------------------------
# Lesson length is a function of HOW MANY IDEAS the lesson has to teach, not of
# how many words the source spent on them. That single change is what turns this
# product from a re-voicing into a course.
#
# It used to be a density band against source words (DENSITY_MIN 0.9 / MAX 1.6).
# The floor made compression structurally impossible: a lesson could never run
# shorter than 90% of its source, so 780,000 words of Bible could not become
# fewer than about 700,000 narrated ones — ~370 lessons, ~108 hours, most of it
# reciting genealogies name by name. The band was tuned against a real failure
# (a 334-word letter blown into ~1,290 words across four lessons), and that
# failure is still guarded — by the ceiling below, which is still expressed as a
# share of source words.
#
# The ratio is self-adjusting because the concept index already collapses
# repetition: forty genealogy entries index as ONE concept, so they get one
# concept's worth of words. Dense argument indexes as a dozen concepts per page
# and barely compresses, which is correct — you cannot teach twelve ideas in two
# hundred words.
#
# The numbers below are starting points, not measurements. They are the dial to
# turn if courses come back too long or too thin; every one is env-overridable so
# a run can be retuned without a deploy.
class Depth(NamedTuple):
    words_per_concept: int   # narration words to teach one indexed idea
    floor_share: float       # never shorter than this share of the source
    ceiling_share: float     # never longer than this share of the source
    planning_ratio: float    # expected overall narration:source, for lesson counts


DEPTHS: dict[str, Depth] = {
    # The pre-existing behavior, kept so a reader who wants the whole text read
    # back still can: near parity, expanding where explaining costs words.
    "comprehensive": Depth(320, 0.85, 1.60, 1.25),
    # The default. Teaches every indexed idea; drops the enumeration.
    "standard": Depth(150, 0.15, 0.60, 0.42),
    # A survey. Every idea still gets named and taught, in fewer words each.
    "overview": Depth(80, 0.06, 0.30, 0.20),
}
DEFAULT_DEPTH = os.getenv("BOOK_ALCHEMY_DEFAULT_DEPTH", "standard")

# How far the writer may drift either side of the computed target before the
# length is worth logging. Wide, because the concept count is an estimate and the
# prompt is explicit that the budget is a guide rather than a quota.
BUDGET_UNDERSHOOT = 0.75
BUDGET_OVERSHOOT = 1.25

MAX_TOPICS_PER_SEGMENT = 5        # detail carried into the segmentation prompt
MAX_SEGMENT_DIGEST_CHARS = 60_000 # keep that prompt inside the context window
MAX_PRIOR_COVERAGE_CHARS = 6_000  # what earlier lessons taught, carried into each script


class TerminalError(Exception):
    """A non-recoverable error; the project should be marked failed."""


class RetryableError(Exception):
    """The step was interrupted by something that isn't the book's fault.

    The project keeps its place and is picked up again — by this worker or the
    next one — rather than being failed and refunded. Restarting the worker is
    the case that matters: systemd sends SIGTERM to the whole control group, so
    the Tesseract child of a running OCR dies too, and that used to surface as
    "OCR failed for scanned PDF: (-15, ...)" and permanently fail a book that had
    nothing wrong with it.
    """


# Set by the worker's signal handler. A failure that happens while this is true
# is a consequence of the shutdown, not a verdict on the source document.
_shutting_down = False


def signal_shutdown() -> None:
    global _shutting_down
    _shutting_down = True


def shutting_down() -> bool:
    return _shutting_down


def _killed_by_signal(exc: BaseException) -> bool:
    """Did this exception come from a child process being killed?

    pytesseract reports the child's exit status, and a negative status is the
    POSIX convention for "died on signal N" — SIGTERM from a restart, or SIGKILL
    from the OOM killer. Neither says anything about the PDF.
    """
    status = getattr(exc, "status", None)
    if isinstance(status, int) and status < 0:
        return True
    return bool(re.search(r"^\(-\d+,", str(exc).strip()))


# Cursor values that mean "there is nothing for the worker to do here". One list,
# imported by worker.py too, so the three places that ask that question can never
# drift apart. 'awaiting_user' differs from the other two only in that a click
# brings it back.
TERMINAL_PHASES = frozenset({"complete", "failed"})
IDLE_PHASES = TERMINAL_PHASES | {"awaiting_user"}


async def run_step(project: dict, client: Optional[DeepSeekClient] = None) -> bool:
    """Advance one project by a single unit. Returns True if more work remains."""
    phase = project["phase"]
    if phase in IDLE_PHASES:
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
    return bool(refreshed and refreshed["phase"] not in IDLE_PHASES)


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
    # A URL project may carry up to ten. They are read in the order they were added
    # and concatenated into ONE document, because that is what the course is: the
    # chunker, the concept index and the lesson budget all reason over a single text,
    # and ten separate documents would be ten courses.
    source_urls = [
        u for u in (_as_dict(project.get("options")).get("source_urls") or []) if str(u).strip()
    ] or ([project.get("source_url")] if project.get("source_url") else [])
    try:
        # In a thread: parsing is synchronous (PyMuPDF, a blocking URL fetch) and the
        # worker runs several books on one event loop. Inline, a long parse froze every
        # other lane AND their lease heartbeats — and a lease that lapses mid-step lets
        # a second lane claim the same book.
        if fmt == "url" and len(source_urls) > 1:
            doc = await asyncio.to_thread(_parse_urls, source_urls, title_hint=project.get("name"))
        else:
            doc = await asyncio.to_thread(
                parse_source,
                source_format=fmt,
                path=source_path,
                url=(source_urls[0] if source_urls else project.get("source_url")),
                title_hint=project.get("name"),
            )
    except ScannedPdfError:
        # Image/scanned PDF: recover the text with the existing OCR pipeline
        # (render -> Tesseract -> DeepSeek clean), then continue as plain text.
        doc = await _ocr_pdf_to_doc(project, source_path)
    except UnsupportedSourceError as exc:
        raise TerminalError(str(exc)) from exc

    chunks = await asyncio.to_thread(build_chunks, doc)
    if not chunks:
        raise TerminalError("No readable text could be extracted from the source.")

    # Where the material actually begins. Settled here, before anything is
    # stored, so no later phase ever sees the front matter as teachable.
    chunks = await _mark_front_matter(client, chunks)

    await db.set_project(pid, stage="Chunking content", progress=CLEAN_PROGRESS_CEILING)
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


# ---------------------------------------------------------- front matter
# A document describes itself before it describes its subject: the filename it
# ships under, where to download it, what may be done with copies, who wrote it
# and how to reach them. A course narrated all of that and then framed itself
# around the author's degrees, because neither existing filter could see it —
# parsers.py only catches lines SHAPED like references, and an "about the
# author" note is ordinary prose, while the analyze phase's `teachable` verdict
# is per-chunk and the front matter shared a 4000-char chunk with the opening of
# the real material.
#
# So it is answered ONCE, here, with the opening excerpts side by side. Front
# matter is a positional fact — one contiguous run from position zero — and that
# is exactly the shape a per-excerpt verdict cannot express.
FRONT_MATTER_MAX_CHUNKS = 12      # front matter never runs deeper than this
# Mirrors chunking.APPARATUS_MAX_SHARE and TEACHABLE_MIN_SHARE: a classifier that
# claims implausibly much of a source has its verdict refused wholesale rather
# than silently eating the book. Set higher than their 25% because this claim is
# far more constrained — one run from the start, not a scattered regex verdict.
FRONT_MATTER_MAX_SHARE = 0.30


async def _mark_front_matter(client: DeepSeekClient, chunks: list[dict]) -> list[dict]:
    """Mark the leading front-matter run not teachable, splitting the boundary chunk.

    Nothing is deleted — same contract as the parser's apparatus mark: the reader
    paid to convert this file, so every word of it stays in the database and only
    the verdict changes.
    """
    if len(chunks) < 2:
        # A single chunk IS the whole source; there is nothing to cut it against,
        # and a verdict here could leave the course with no material at all.
        return chunks

    head = chunks[:FRONT_MATTER_MAX_CHUNKS]
    try:
        verdict = await client.chat_json(
            system=prompts.FRONT_MATTER_SYSTEM,
            user=prompts.front_matter_user(head),
            max_output_tokens=500,
        )
    except Exception as exc:  # noqa: BLE001
        # Never fail a job over this: the course is merely duller without it.
        log.warning("Front-matter boundary check failed, keeping everything: %s", exc)
        return chunks

    if not isinstance(verdict, dict):
        return chunks
    try:
        boundary = int(verdict.get("first_material_ordinal") or 0)
    except (TypeError, ValueError):
        return chunks
    boundary = max(0, min(boundary, len(head) - 1))

    sentence = verdict.get("first_material_sentence")
    sentence = str(sentence).strip() if sentence else ""

    # The straddling chunk is the case that broke this: front matter and the
    # opening of chapter one in one chunk, where no whole-chunk verdict is right.
    # Split rather than trim, so the invariant build_chunks maintains holds — a
    # chunk is entirely teachable or entirely not, never a blend.
    split_at = -1
    if sentence:
        found = chunks[boundary]["text"].find(sentence)
        if found > 0:
            split_at = found

    if boundary == 0 and split_at < 0:
        return chunks   # opens straight onto material

    dropped = sum(_word_count(c["text"]) for c in chunks[:boundary])
    if split_at > 0:
        dropped += _word_count(chunks[boundary]["text"][:split_at])
    if not dropped:
        return chunks

    total = sum(_word_count(c["text"]) for c in chunks) or 1
    share = dropped / total
    if share > FRONT_MATTER_MAX_SHARE:
        log.warning(
            "Front-matter boundary would drop %.1f%% of this source (%s of %s words) — "
            "above the %.0f%% ceiling, so keeping everything. Narrating a preface beats "
            "silently eating the opening of the work.",
            share * 100, dropped, total, FRONT_MATTER_MAX_SHARE * 100,
        )
        return chunks

    out: list[dict] = []
    for i, chunk in enumerate(chunks):
        if i < boundary:
            out.append({**chunk, "teachable": False})
        elif i == boundary and split_at > 0:
            front = chunk["text"][:split_at].strip()
            rest = chunk["text"][split_at:].strip()
            if front and rest:
                out.append({
                    **chunk, "text": front, "teachable": False,
                    "char_end": chunk["char_start"] + len(front),
                })
                out.append({
                    **chunk, "text": rest, "teachable": True,
                    "char_start": chunk["char_start"] + split_at,
                })
            else:
                out.append(dict(chunk))
        else:
            out.append(dict(chunk))

    # The split added a chunk, and every later phase indexes by ordinal.
    for i, chunk in enumerate(out):
        chunk["ordinal"] = i

    log.info(
        "Front matter: material begins at excerpt %s%s — %s of %s words (%.1f%%) marked "
        "not teachable",
        boundary, " (mid-excerpt)" if split_at > 0 else "", dropped, total, share * 100,
    )
    return out


# How the scanned-PDF detour spends the progress bar. Parse owns 4-10%: the OCR
# read and the cleaning pass split 4-9 between them, leaving 9-10 for chunking.
# The numbers are small on purpose — this is early work — but they must MOVE.
# Recognizing a thousand-page scan runs well over an hour, and reporting a single
# flat 6% for all of it is indistinguishable from a hung job; the stage line
# carries the page count that actually tells a reader it is alive.
OCR_PROGRESS_FLOOR = 4
OCR_PROGRESS_CEILING = 7
CLEAN_PROGRESS_CEILING = 9
OCR_PROGRESS_INTERVAL_S = 2.0     # at most one row update this often


def _ocr_progress_reporter(pid: int):
    """Build the callback the PDF pipeline ticks, throttled to spare the DB.

    Pages land every second or so across four OCR workers; the reader gains
    nothing from a write per page, so updates are rate-limited and only the last
    one in each stage is guaranteed.
    """
    state = {"last_write": 0.0, "last_done": {"ocr": 0, "clean": 0}}

    async def report(step: str, done: int, total: int) -> None:
        total = max(1, total)
        finished = done >= total
        now = time.monotonic()
        if not finished and (now - state["last_write"]) < OCR_PROGRESS_INTERVAL_S:
            return
        # Ticks arrive from concurrent tasks, so a slow write could otherwise
        # land after a newer one and walk the count backwards.
        if done <= state["last_done"].get(step, 0) and not finished:
            return
        state["last_write"] = now
        state["last_done"][step] = done

        if step == "ocr":
            span = OCR_PROGRESS_CEILING - OCR_PROGRESS_FLOOR
            progress = OCR_PROGRESS_FLOOR + int(span * done / total)
            stage = f"Reading scanned pages ({done:,}/{total:,})"
        else:
            span = CLEAN_PROGRESS_CEILING - OCR_PROGRESS_CEILING
            progress = OCR_PROGRESS_CEILING + int(span * done / total)
            stage = f"Cleaning recognized text ({done:,}/{total:,})"

        try:
            await db.set_project(pid, stage=stage, progress=min(progress, CLEAN_PROGRESS_CEILING))
        except Exception:  # noqa: BLE001 — a progress write must never end a book
            log.warning("Could not write OCR progress for project %s", pid, exc_info=True)

    return report


def ocr_cache_dir(pdf_path: Path) -> Path:
    """Where finished OCR pages and cleaned batches are parked for resume.

    Inside the project's own folder, so it is removed with the source file once
    parsing is behind us.
    """
    return pdf_path.parent / "ocr_cache"


async def _ocr_pdf_to_doc(project: dict, source_path: Optional[str]):
    """Run the existing SpokenVerse OCR pipeline on a scanned PDF and return a
    ParsedDoc of the recovered text. Requires Tesseract (a SpokenVerse system
    dependency) and DeepSeek; failures surface as a clear TerminalError."""
    pid = int(project["id"])
    if not source_path:
        raise TerminalError("Scanned PDF but no source file is available for OCR.")

    await db.set_project(pid, stage="Reading scanned pages", progress=OCR_PROGRESS_FLOOR)
    try:
        from phansora.products.spokenverse.txt_to_voice.pdf_pipeline import PdfConverter, PdfToTxtConfig  # lazy

        pdf_path = Path(source_path)
        out_txt = pdf_path.with_suffix(".ocr.txt")
        cfg = PdfToTxtConfig(keep_page_breaks=False, to_chapters=False)
        await PdfConverter(cfg).convert_pdf_to_txt_async(
            pdf_path, out_txt,
            cache_dir=ocr_cache_dir(pdf_path),
            on_progress=_ocr_progress_reporter(pid),
        )

        text = out_txt.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            raise TerminalError("OCR produced no readable text from the scanned PDF.")
        # Re-use the plain-text parser on the recovered text.
        return parse_source(source_format="text", path=str(out_txt), title_hint=project.get("name"))
    except TerminalError:
        raise
    except Exception as exc:  # noqa: BLE001
        # A book interrupted by a deploy is not a book that failed. Every page
        # already recognized is on disk, so the retry resumes rather than restarts.
        if shutting_down() or _killed_by_signal(exc):
            raise RetryableError(f"OCR interrupted by shutdown: {exc}") from exc
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

    if not _is_teachable(chunk):
        # The parser already recognized this as contents/index/copyright matter.
        # Indexing it would cost a model call to describe something no lesson
        # will ever teach.
        await db.set_project(
            pid, analyze_cursor=cursor + 1,
            stage=f"Extracting concepts ({cursor + 1}/{total})",
            progress=min(10 + int(40 * (cursor + 1) / max(1, total)), 49),
        )
        return

    try:
        extracted = await client.chat_json(
            system=prompts.ANALYZE_SYSTEM,
            user=prompts.analyze_user(chunk["text"], chapter=chunk["chapter"]),
            # Sized for the closed-book notes (bodies up to ~60 words), which are
            # now the writer's only material — a truncated note is course content
            # lost, not just a thinner index. chat_json retries bigger on
            # truncation anyway; starting near the need avoids the doubled call.
            max_output_tokens=6000,
        )
        # The reader of the excerpt gets the final say on apparatus: it can see
        # that "The book of Job has forty-two chapters" is a chapter listing
        # written as a sentence, which no pattern over line shapes can.
        if isinstance(extracted, dict) and extracted.get("teachable") is False:
            await db.set_chunk_teachable(int(chunk["id"]), False)
            log.info(
                "Project %s: chunk %s indexed as apparatus, not teachable", pid, cursor
            )
        else:
            if isinstance(extracted, dict) and extracted.get("truncated"):
                # The index is the coverage contract now, so a truncated one is
                # content the lesson will never be required to teach.
                log.warning(
                    "Project %s: chunk %s hit the concept-index item ceiling; some "
                    "ideas in it are outside the coverage contract", pid, cursor,
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

    all_chunks = await db.get_all_chunks(pid)
    if not all_chunks:
        raise TerminalError("No readable source segments were found.")

    depth = resolve_depth(project.get("options"))
    chunks = _teachable_chunks(pid, all_chunks)

    digests = await _chunk_digests(pid, chunks)
    source_words = sum(d["words"] for d in digests)
    min_lessons, suggested, max_lessons = _lesson_budget(source_words, depth)

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
                max_lesson_words=max_source_words_per_lesson(depth),
            ),
            max_output_tokens=8000,
        )
        entries = _entries_from_plan(raw, segment_count=len(digests), max_lessons=max_lessons)

    lessons = _resolve_lessons(entries, digests, depth)
    log.info(
        "Project %s: %s source words -> %s lesson(s) at depth ratio %.2f "
        "(suggested %s, max %s)",
        pid, source_words, len(lessons), depth.planning_ratio, suggested, max_lessons,
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

    # Delivery phases: which lessons the listener receives as one batch. Planned
    # here because everything it needs — the lessons, their source word counts and
    # the chapter each came from — is already in memory. It costs no query and no
    # model call, and it is the one and only time these boundaries are decided: a
    # listener told "Phase 3 of 7" must never watch that renumber.
    #
    # Imported inside the function on purpose. phases.py reads this module's
    # listening constants, so a top-level import here would be a cycle.
    from . import phases as phases_mod

    planned = phases_mod.plan_phases([
        {
            "ordinal": l["ordinal"],
            # The same figure the lesson boundaries were placed by, so a phase's
            # estimate and its lessons' lengths cannot disagree.
            "seconds": int(l["planned_words"] * 60 / WORDS_PER_MINUTE),
            "chapter": chunks[l["start"]]["chapter"],
        }
        for l in lessons
    ])
    await db.create_phases(pid, planned)
    log.info(
        "Project %s: %s lesson(s) -> %s delivery phase(s) (cap %ss)",
        pid, len(lessons), len(planned), phases_mod.PHASE_TARGET_SECONDS,
    )

    plan = {
        "work_title": project.get("name"),
        "source_words": source_words,
        "lesson_count": len(lessons),
        # Recorded so a finished course can be read back against the settings that
        # produced it — "why is this one 30 hours and that one 8" is otherwise
        # unanswerable once the run is over.
        "depth": _as_dict(project.get("options")).get("depth") or DEFAULT_DEPTH,
        "skipped_apparatus_chunks": len(all_chunks) - len(chunks),
        "phases": planned,
        "sessions": [
            {
                "ordinal": l["ordinal"],
                "title": l["title"],
                "summary": l["summary"],
                "topics": l["topics"],
                "segment_range": [l["start"], l["end"]],
                "source_words": l["words"],
                "planned_words": l["planned_words"],
            }
            for l in lessons
        ],
    }
    await db.set_project(
        pid, curriculum=plan, phase="sessions",
        stage="Creating sessions", progress=55,
    )


def _parse_urls(urls: list, *, title_hint: Optional[str] = None) -> ParsedDoc:
    """Read several article URLs into one document.

    Every page contributes its blocks in the order it was added. A page that cannot be
    fetched or parsed is SKIPPED with a log line rather than failing the course — one
    dead link out of ten is not a reason to lose the other nine — but if every one
    fails there is nothing to teach and the caller's "no readable text" check fires.

    The title comes from the project name when there is one, else the first page that
    parsed, because "3 articles" is what the row already says.
    """
    blocks: list = []
    titles: list[str] = []
    for u in urls:
        try:
            part = parse_source(source_format="url", url=u, title_hint=None)
        except Exception as exc:  # noqa: BLE001 — one bad link must not lose the rest
            log.warning("Skipping unreadable source URL %s: %s", u, exc)
            continue
        if part.blocks:
            blocks.extend(part.blocks)
            titles.append(part.title)
    return ParsedDoc(title=title_hint or (titles[0] if titles else "Untitled"), blocks=blocks)


def resolve_depth(options: Any) -> Depth:
    """The depth profile for a project, falling back to the default on anything
    unrecognized (including projects created before the option existed)."""
    name = str(_as_dict(options).get("depth") or "").strip().lower()
    return DEPTHS.get(name) or DEPTHS.get(DEFAULT_DEPTH) or DEPTHS["standard"]


def lesson_word_budget(
    *, concept_count: int, source_words: int, depth: Depth
) -> tuple[int, int]:
    """(min_words, max_words) of narration for one lesson.

    Driven by the number of distinct ideas indexed in the lesson's segments, then
    clamped to a share of the source so neither failure mode can run away: a
    lesson cannot balloon past what the source can support, and cannot collapse
    to a summary when the index came back thin.
    """
    target = _budget_target(concept_count, source_words, depth)

    min_words = max(60, int(target * BUDGET_UNDERSHOOT))
    if target >= MIN_LESSON_WORDS:
        # A lesson planned to fill a sitting is never asked for less than one. The
        # undershoot allowance is for the estimate being rough, not a licence to
        # hand back seven minutes where ten were planned.
        min_words = max(min_words, MIN_LESSON_WORDS)
    max_words = max(min_words + 60, int(target * BUDGET_OVERSHOOT))
    return min_words, max_words


def _budget_target(concept_count: int, source_words: int, depth: Depth) -> int:
    """The narration length a lesson's ideas call for, clamped to its source."""
    target = max(1, concept_count) * depth.words_per_concept
    floor = int(source_words * depth.floor_share)
    ceiling = int(source_words * depth.ceiling_share)
    if ceiling > 0:
        target = max(floor, min(ceiling, target))
    return target


def planned_lesson_words(*, concept_count: int, source_words: int, depth: Depth) -> int:
    """How long a lesson over this much source is planned to run, in narration words.

    The smaller of two estimates: what the depth expects of any source this long
    (``planning_ratio``), and what the writer will actually be asked for given the
    ideas indexed in it. The second is what the first cannot see — a genealogy is
    thousands of source words and two concepts, so a lesson cut to "ten minutes"
    by its word count alone would be asked for three.
    """
    expected = int(max(0, source_words) * depth.planning_ratio)
    return min(expected, _budget_target(concept_count, source_words, depth))


def _lesson_budget(source_words: int, depth: Depth) -> tuple[int, int, int]:
    """(minimum, suggested, maximum) lesson COUNT for a source of this length.

    Computed in projected NARRATION words, not source words. The listener's unit
    is a sitting, so what has to divide evenly is the finished audio; dividing
    source words instead would silently shrink every lesson as compression rose —
    a 2,100-word lesson target against a 0.42 ratio yields ~5,000 source words per
    lesson, and using the source figure would have produced 2.4x too many lessons,
    each about six minutes long.

    A source that fits one comfortable sitting returns (1, 1, 1) — the caller then
    skips the segmentation model entirely, so a short work can never be split into
    a multi-lesson "course"."""
    narration_words = max(0, int(source_words * depth.planning_ratio))
    if narration_words <= MAX_LESSON_WORDS:
        return 1, 1, 1
    minimum = max(1, math.ceil(narration_words / MAX_LESSON_WORDS))
    maximum = max(minimum, math.ceil(narration_words / MIN_LESSON_WORDS))
    suggested = min(maximum, max(minimum, math.ceil(narration_words / TARGET_LESSON_WORDS)))
    return minimum, suggested, maximum


def max_source_words_per_lesson(depth: Depth) -> int:
    """MAX_LESSON_WORDS expressed in SOURCE words.

    Segmentation reasons about source segments and their word counts, so the cap
    it is given has to be in the same currency it counts in.
    """
    return max(MIN_LESSON_WORDS, int(MAX_LESSON_WORDS / max(0.01, depth.planning_ratio)))


# The most of a source that may be dropped as apparatus before the verdict is
# refused wholesale. Mirrors chunking.APPARATUS_MAX_SHARE, but guards the OTHER
# classifier: the parser's regexes are checked there, the analyze phase's
# judgment is checked here. Both can misfire, and neither is allowed to quietly
# delete half of what the reader uploaded — a course that narrates some contents
# pages is a much better failure than one missing a third of the book.
TEACHABLE_MIN_SHARE = 0.75


def _teachable_chunks(project_id: int, chunks: list[Any]) -> list[Any]:
    """Drop the chunks classified as front/back matter, unless there are too many.

    Positional integrity matters downstream: _resolve_lessons and the
    create_session loop both index into the list this returns, so the filtering
    has to happen once, here, and every later reference must use the result.
    """
    keep = [c for c in chunks if _is_teachable(c)]
    if not keep:
        log.warning(
            "Project %s: every chunk was classified as apparatus — keeping all %s",
            project_id, len(chunks),
        )
        return list(chunks)

    kept_words = sum(_word_count(c["text"]) for c in keep)
    total_words = sum(_word_count(c["text"]) for c in chunks) or 1
    share = kept_words / total_words
    if share < TEACHABLE_MIN_SHARE:
        log.warning(
            "Project %s: apparatus filter would drop %.1f%% of the source words "
            "(%s of %s chunks kept) — above the ceiling, so keeping everything.",
            project_id, (1 - share) * 100, len(keep), len(chunks),
        )
        return list(chunks)

    if len(keep) != len(chunks):
        log.info(
            "Project %s: skipping %s of %s chunks as front/back matter (%.1f%% of words kept)",
            project_id, len(chunks) - len(keep), len(chunks), share * 100,
        )
    return keep


def _is_teachable(chunk: Any) -> bool:
    """Default True: rows written before the column existed have no verdict, and
    absence of a verdict is not a verdict."""
    try:
        value = chunk["teachable"]
    except (KeyError, TypeError, IndexError):
        return True
    return True if value is None else bool(value)


async def _chunk_digests(project_id: int, chunks: list[Any]) -> list[dict]:
    """One planning line per source segment: where it sits, how long it is, and
    what the analyze phase indexed in it. Used only to place lesson boundaries."""
    by_chunk: dict[int, list[str]] = {}
    # Every indexed idea, not just the titles kept for the prompt: this is what
    # the lesson will be asked to teach, so it is what its length is planned from.
    counts: dict[int, int] = {}
    for row in await db.get_concepts(project_id):
        title = str(_as_dict(row["content"]).get("title") or "").strip()
        for cid in (row["source_chunk_ids"] or []):
            counts[int(cid)] = counts.get(int(cid), 0) + 1
            if not title:
                continue
            topics = by_chunk.setdefault(int(cid), [])
            if title not in topics and len(topics) < MAX_TOPICS_PER_SEGMENT:
                topics.append(title)

    return [
        {
            "ordinal": i,
            "chapter": c["chapter"],
            "words": _word_count(c["text"]),
            "topics": by_chunk.get(int(c["id"]), []),
            "concepts": counts.get(int(c["id"]), 0),
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


# --- Where a lesson may end -------------------------------------------------------
# A lesson can only begin or end on a segment boundary, and not all of those are
# equal. Some are places the SOURCE stops and starts again — a new chapter, or a
# shift of subject the segmentation model picked out. The rest are just where
# 4,000 characters happened to run out, in the middle of whatever was being said.
#
# The plan is the set of cuts with the lowest total cost, in planned minutes:
#
#   * a lesson costs nothing at MIN_LESSON_MINUTES and a little more for each
#     minute past it, so lessons sit as close to the floor as the stopping points
#     allow rather than running long for no reason;
#   * a lesson under the floor costs SHORT_LESSON_WEIGHT a minute. This is what
#     folds a three-minute chapter into its neighbor. It used to stand alone:
#     nothing stopped the model proposing a lesson out of one short section, and
#     that lesson then ran for as long as one short section takes to teach;
#   * a lesson over MAX_LESSON_MINUTES costs LONG_LESSON_WEIGHT a minute;
#   * ending a lesson where the source does NOT change subject costs
#     UNNATURAL_CUT_COST, flat. It is priced so that a topic runs a few minutes
#     past the cap before it is cut in half mid-thought — and so that when a
#     source has no structure at all, it gets as few such cuts as possible.
#
# These weights only arbitrate between imperfect options. Where the source offers
# a stopping point between the floor and the cap, the lesson ends there and none
# of them is paid.
SHORT_LESSON_WEIGHT = 8.0
LONG_LESSON_WEIGHT = 4.0
UNNATURAL_CUT_COST = 36.0
_EVEN_SPLIT_WEIGHT = 0.01     # between otherwise equal plans, the more even one


def _lesson_cost(minutes: float) -> float:
    if minutes < MIN_LESSON_MINUTES:
        return (MIN_LESSON_MINUTES - minutes) * SHORT_LESSON_WEIGHT
    past_floor = minutes - MIN_LESSON_MINUTES
    cost = past_floor + _EVEN_SPLIT_WEIGHT * past_floor * past_floor
    if minutes > MAX_LESSON_MINUTES:
        cost += (minutes - MAX_LESSON_MINUTES) * LONG_LESSON_WEIGHT
    return cost


def _chapter_starts(digests: list[dict]) -> set[int]:
    """Segments that open on a different chapter from the one before them."""
    def key(d: dict) -> str:
        return str(d.get("chapter") or "").strip().casefold()

    return {i for i in range(1, len(digests)) if key(digests[i]) != key(digests[i - 1])}


def _plan_cuts(digests: list[dict], natural: set[int], depth: Depth) -> list[int]:
    """The segment each lesson starts on, in order. Always begins with 0.

    ``natural`` is the set of segments a lesson may start on without cutting into
    a topic. Every other boundary is still available — a source with no structure
    has to be cut somewhere — but at a price; see the block comment above.
    """
    total = len(digests)
    words = [0]
    concepts = [0]
    for d in digests:
        words.append(words[-1] + int(d.get("words") or 0))
        concepts.append(concepts[-1] + int(d.get("concepts", len(d.get("topics") or []))))

    def minutes(start: int, stop: int) -> float:
        """Planned length of a lesson over segments [start, stop)."""
        return planned_lesson_words(
            concept_count=concepts[stop] - concepts[start],
            source_words=words[stop] - words[start],
            depth=depth,
        ) / WORDS_PER_MINUTE

    # A work that fits one sitting is one lesson, however many chapters it has.
    if total <= 1 or minutes(0, total) <= MAX_LESSON_MINUTES:
        return [0]

    best = [0.0] + [math.inf] * total      # cheapest plan for the first `stop` segments
    came_from = [0] * (total + 1)
    for stop in range(1, total + 1):
        closing = 0.0 if (stop == total or stop in natural) else UNNATURAL_CUT_COST
        for start in range(stop - 1, -1, -1):
            length = minutes(start, stop)
            # Planned length only grows as a lesson reaches further back, so past
            # twice the cap nothing earlier can be cheaper.
            if length > 2 * MAX_LESSON_MINUTES and start < stop - 1:
                break
            cost = best[start] + _lesson_cost(length) + closing
            if cost < best[stop]:
                best[stop], came_from[stop] = cost, start

    cuts: list[int] = []
    stop = total
    while stop > 0:
        stop = came_from[stop]
        cuts.append(stop)
    return cuts[::-1]


def _resolve_lessons(entries: list[dict], digests: list[dict], depth: Depth) -> list[dict]:
    """Turn the model's start-points into a complete, non-overlapping cover of the segments.

    The model says where the source changes subject; :func:`_plan_cuts` decides
    which of those places lessons actually end on, so that each lesson fills a
    sitting. A proposed part that is too short is folded into its neighbor, and
    one that is too long is split — which also keeps each script inside the
    model's output-token ceiling. Titles and topic lists follow the parts they
    came from.
    """
    total = len(digests)
    parts = sorted(
        (e for e in entries if 0 <= int(e["start"]) < total), key=lambda e: e["start"]
    ) or [{"start": 0, "title": "", "summary": "", "topics": []}]
    parts[0] = {**parts[0], "start": 0}     # the work always starts at its first segment
    part_starts = [int(p["start"]) for p in parts]

    cuts = _plan_cuts(digests, set(part_starts) | _chapter_starts(digests), depth)

    lessons: list[dict] = []
    for i, start in enumerate(cuts):
        end = (cuts[i + 1] - 1) if i + 1 < len(cuts) else total - 1
        # The proposed part this lesson opens in, and any that open inside it.
        owner = max(j for j, s in enumerate(part_starts) if s <= start)
        opening = [p for p in parts if start < int(p["start"]) <= end]

        first = str(parts[owner].get("title") or "").strip()
        if first and start != part_starts[owner]:
            # Opens partway through a proposed part: number it among the lessons
            # that part was divided into. Its opening is part 1 whether that
            # stands alone or was folded into the lesson before.
            earlier = sum(1 for c in cuts if part_starts[owner] < c <= start)
            first = f"{first} (part {earlier + 1})"
        names: list[str] = []
        for name in [first] + [str(p.get("title") or "").strip() for p in opening]:
            if name and name not in names:
                names.append(name)

        # The model's topic list describes whole parts. A lesson made of anything
        # else is refilled below from the segments it actually covers.
        whole_parts = start == part_starts[owner] and (
            end == total - 1 or (end + 1) in part_starts
        )
        topics = (
            [t for p in [parts[owner]] + opening for t in (p.get("topics") or [])]
            if whole_parts else []
        )
        # Same convention as a delivery phase's label: one name, or first – last.
        title = "" if not names else names[0] if len(names) == 1 else f"{names[0]} – {names[-1]}"
        lessons.append({
            "start": start,
            "end": end,
            "title": title[:160],
            "summary": str(parts[owner].get("summary") or ""),
            "topics": topics,
        })

    for i, lesson in enumerate(lessons, start=1):
        covered = digests[lesson["start"]: lesson["end"] + 1]
        lesson["ordinal"] = i
        lesson["words"] = _range_words(digests, lesson["start"], lesson["end"])
        lesson["planned_words"] = planned_lesson_words(
            concept_count=sum(int(d.get("concepts", len(d.get("topics") or []))) for d in covered),
            source_words=lesson["words"],
            depth=depth,
        )
        if not lesson["title"]:
            lesson["title"] = f"Part {i}"
        if not lesson["topics"]:
            lesson["topics"] = [t for d in covered for t in d["topics"]]
    return lessons


def _range_words(digests: list[dict], start: int, end: int) -> int:
    return sum(d["words"] for d in digests[start: end + 1])


def _word_count(text: Any) -> int:
    return len(str(text or "").split())


# --------------------------------------------------------------- phase: sessions (script + validate)
async def _phase_sessions(project: dict, client: DeepSeekClient) -> None:
    pid = int(project["id"])
    # WHOLE-PROJECT on purpose, never scoped to the current delivery phase: this
    # list is the input to _prior_coverage, which is what stops lesson 30
    # re-teaching what lesson 4 already covered. Narrowing it to the phase would
    # silently change every prompt from phase 2 onwards.
    sessions = await db.get_sessions(pid)
    total = len(sessions)
    ceiling = await db.work_ceiling(pid)
    sess = await db.next_session_needing(pid, ["pending"], ceiling)
    # Record each lesson as soon as its script exists, before writing the next one.
    # Writing every script in the phase first meant nothing was playable until all of
    # them were done — ~13 minutes for a six-lesson book, far longer for a big phase —
    # when lesson 1 could have been in the listener's ears after one script and one
    # recording. The order of the scripts is unchanged (each is still conditioned on
    # the ones before it), so nothing about what gets written moves.
    ready = await db.next_session_needing(pid, ["validated"], ceiling)
    if _record_before_writing(ready, sess):
        await _record_lesson(project, ready, total, sessions)
        return
    if sess is None:
        await db.set_project(pid, phase="audio", stage="Generating audio", progress=80)
        return

    await db.set_project(
        pid, stage=await _stage_label(pid, f"writing lesson {sess['ordinal']} of {total}"),
        progress=_course_progress(sessions),
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

    # The lesson's coverage contract: the ideas indexed in exactly these segments.
    # Loaded once and handed to BOTH the writer and the checker — see
    # validate_script for what happens when those two lists differ.
    concept_rows = await db.get_concepts_for_chunks(pid, chunk_ids)
    concepts = [_as_dict(row["content"]) for row in concept_rows]
    checklist = prompts.concept_checklist(concepts)
    if not checklist:
        # The writer no longer sees the chunk text, so with no notes it has only
        # the outline titles to teach from — a thin lesson the validator will
        # judge against the full source. Rare (analyze skips individual failed
        # chunks), but worth a trace when a lesson comes out flagged and empty.
        log.warning(
            "Project %s session %s: no concept notes for its chunks; the "
            "closed-book writer has only the outline to teach from",
            pid, sess["ordinal"],
        )

    # Length follows the number of ideas, not the length of the source. See the
    # Depth table for why, and for the clamps that keep both failure modes shut.
    depth = resolve_depth(project.get("options"))
    source_words = sum(_word_count(c["text"]) for c in chunk_dicts)
    min_words, max_words = lesson_word_budget(
        concept_count=len(checklist), source_words=source_words, depth=depth,
    )
    # Lessons are written in ordinal order, so everything before this one is
    # already settled. Handing that over is what stops a course re-teaching the
    # same ground each time the source circles back to it.
    prior = _prior_coverage(sessions, int(sess["ordinal"]))

    script = ""
    validation = {"supported": False, "flagged": [], "notes": ""}
    feedback: Optional[list] = None
    regen = 0
    for attempt in range(MAX_REGEN + 1):
        regen = attempt
        # chat_to_end, not chat: the writer gets the model's whole output ceiling
        # and is asked to carry on if it still runs out. The cap used to be sized
        # from max_words (words * 1.7 + 400 tokens), but max_words is a pacing
        # hint the writer is free to exceed — it has every note on the list to
        # teach — and when it did, the reply stopped dead at the cap and was
        # recorded as it stood, mid-sentence.
        script, finished = await client.chat_to_end(
            system=prompts.SCRIPT_SYSTEM,
            # Closed-book: the writer gets the concept notes and never the chunk
            # text. chunk_dicts still feed validate_script below — the checker is
            # the one place the source is allowed back in.
            user=prompts.script_user(
                sess["title"], outline,
                min_words=min_words,
                max_words=max_words,
                concepts=concepts,
                feedback=feedback,
                previously_taught=prior,
            ),
            # A retry must not reproduce the rejected script verbatim. The
            # feedback already changes the prompt; a little temperature helps the
            # model leave a phrasing it has settled on. (At temperature 0 with an
            # unchanged prompt, the old loop regenerated the identical script.)
            temperature=0.0 if attempt == 0 else 0.3,
        )
        # The deterministic floor, applied BEFORE the check so the validator
        # judges the text that will actually ship and the word count below stays
        # honest. Everything above this point is a model asked to behave.
        script, scrubbed = scrub_apparatus(script)
        if scrubbed:
            log.warning(
                "Project %s session %s: scrubbed %s sentence(s) naming a contact "
                "address, web address or filename — a packaging guard leaked: %s",
                pid, sess["ordinal"], len(scrubbed), scrubbed[:3],
            )
        # The other deterministic floor: a lesson never ends on half a sentence.
        # Reached only when the writer was still cut off after its continuations,
        # or signed off without finishing one — rare, and worth knowing about,
        # because whatever followed the cut was never taught.
        script, fragment = complete_ending(script)
        if fragment or not finished:
            log.warning(
                "Project %s session %s: script did not end on a finished sentence "
                "(model stopped by its output ceiling: %s) — dropped %r",
                pid, sess["ordinal"], not finished, fragment[-120:],
            )
        validation = await validate_script(
            client, script=script, chunks=chunk_dicts,
            concepts=concepts, previously_taught=prior,
        )
        if script.strip() and validation["supported"]:
            break
        feedback = validation.get("flagged") or []

    # Logged for every lesson, in or out of range: the planning figures above
    # (MIN_LESSON_MINUTES, the Depth table) are estimates of how the writer paces
    # a lesson, and this line is the measurement they are tuned against.
    written = _word_count(script)
    if script.strip():
        log.info(
            "Project %s session %s: %s narration words (~%.1f min) against a %s-%s "
            "target (%s source words, %s concepts, %.2fx)",
            pid, sess["ordinal"], written, written / WORDS_PER_MINUTE, min_words,
            max_words, source_words, len(checklist), written / max(1, source_words),
        )

    if not script.strip():
        # The model returned nothing on every attempt. Previously this was written out as a
        # blank script and only noticed when the finished course had no audio.
        log.warning(
            "Project %s session %s: model returned an empty script after %s attempt(s) "
            "(%s source words, %s concepts)",
            pid, sess["ordinal"], regen + 1, source_words, len(checklist),
        )

    await db.set_session(
        sess["id"],
        script=script,
        status="validated",
        validation_status="passed" if validation["supported"] else "flagged",
        validation_notes=validation,
        regen_count=regen,
    )


def _prior_coverage(sessions: list[Any], current_ordinal: int) -> list[dict]:
    """What the lessons before this one already taught: title plus topic list.

    Titles and topics, not scripts — a twenty-lesson course would otherwise put
    the whole finished course back into every prompt. Topics are the same
    coverage checklist the segmentation phase produced, so they name exactly the
    ground a repeat would cover.
    """
    entries = [
        {
            "ordinal": int(s["ordinal"]),
            "title": (s["title"] or f"Part {s['ordinal']}").strip(),
            "topics": [str(t).strip() for t in _as_list(s["outline"]) if str(t).strip()],
        }
        for s in sessions
        if int(s["ordinal"]) < current_ordinal
    ]
    return _trim_coverage(entries)


def _trim_coverage(entries: list[dict]) -> list[dict]:
    """Hold the coverage list inside its character budget.

    Detail goes before entries do, oldest first: a lesson the listener heard an
    hour ago is the one whose topic list can be spared, while dropping a lesson
    outright would invite the course to teach it all over again.
    """
    def size(entry: dict) -> int:
        return len(entry["title"]) + sum(len(t) + 8 for t in entry["topics"]) + 16

    total = sum(size(e) for e in entries)
    if total <= MAX_PRIOR_COVERAGE_CHARS:
        return entries

    trimmed = [dict(e) for e in entries]
    for entry in trimmed:
        if total <= MAX_PRIOR_COVERAGE_CHARS:
            break
        before = size(entry)
        entry["topics"] = []
        total -= before - size(entry)
    return trimmed


# --------------------------------------------------------------- phase: audio
async def _phase_audio(project: dict) -> None:
    pid = int(project["id"])
    sessions = await db.get_sessions(pid)
    total = len(sessions)
    sess = await db.next_session_needing(pid, ["validated"], await db.work_ceiling(pid))
    if sess is None:
        # Nothing left within the current delivery phase. Either the listener gets
        # the next one now, or the course is done.
        await _close_active_phase(project)
        return

    await _record_lesson(project, sess, total, sessions)


def _record_before_writing(ready: Any, pending: Any) -> bool:
    """Whether the next unit of lesson work is a recording rather than a script.

    Record a written lesson that comes before the next one to write — which, since
    scripts are written in order, means "the lesson just written". A written lesson
    AFTER the next pending one (a Regenerate reopened an earlier lesson) waits: its
    script is not the next thing anyone will listen to.
    """
    if ready is None:
        return False
    return pending is None or int(ready["ordinal"]) < int(pending["ordinal"])


async def _record_lesson(project: dict, sess: Any, total: int, sessions: list[Any]) -> None:
    """Record one written lesson and mark it complete — the moment it is playable."""
    pid = int(project["id"])
    if not (sess["script"] or "").strip():
        # A blank script silently yields a course with no audio, which is the hardest
        # possible failure to diagnose from the outside — say so.
        log.warning(
            "Project %s session %s: empty script, no audio rendered (validation=%s)",
            pid, sess["ordinal"], sess.get("validation_status"),
        )
        await db.set_session(sess["id"], status="complete", audio_seconds=0)
        return

    await db.set_project(
        pid, stage=await _stage_label(pid, f"recording lesson {sess['ordinal']} of {total}"),
        progress=_course_progress(sessions),
    )

    options = _as_dict(project.get("options"))
    voice = str(options.get("voice") or "default")
    # Chosen once for the course, applied to every lesson — the narrator should not
    # change character between session 3 and session 4.
    instruct_text = str(options.get("instruct_text") or "").strip() or None
    # Pass the raw voice id + user_id to the shared txt-to-audio endpoint; it
    # resolves a cloned voice id -> its reference clip and its stored transcript
    # (ref_text -> prompt_text). (Resolving to a path here would make the endpoint
    # fall back to the default voice.)
    out_path: Path = session_audio_path(project["user_id"], pid, sess["ordinal"], "mp3")

    # Wait for the voice service rather than failing the session.
    #
    # Batch work now yields to anyone waiting on a narration in the app (see
    # spokenverse/admission.py), so a busy 503 here is the SYSTEM WORKING, not a fault —
    # it means a person got the slot. Treating it as a failure would abandon a lesson that
    # would have rendered a minute later, which is the wrong trade for a job nobody is
    # watching. That wait (up to _BUSY_RETRIES back-offs) lives in _render_waiting_out_busy.
    #
    # Any other failure gets _LESSON_ATTEMPTS tries: the voice service is shared, and a
    # fault in one render (the 2026-09-27 failure of project 61 left no message at all)
    # is not evidence the next will fail too. A 4xx is the request's own fault and is
    # not retried; neither is anything once the worker is shutting down.
    for attempt in range(1, _LESSON_ATTEMPTS + 1):
        try:
            seconds = await _render_waiting_out_busy(project, sess, out_path, voice, instruct_text)
            break
        except VoiceServiceBusy:
            raise
        except Exception as exc:  # noqa: BLE001
            client_error = isinstance(exc, VoiceServiceError) and exc.status < 500
            if attempt >= _LESSON_ATTEMPTS or client_error or shutting_down():
                raise
            log.warning(
                "Project %s lesson %s: recording failed (attempt %d/%d), retrying in %ss: %s",
                pid, sess["ordinal"], attempt, _LESSON_ATTEMPTS, _LESSON_RETRY_WAIT_S,
                str(exc) or type(exc).__name__,
            )
            await asyncio.sleep(_LESSON_RETRY_WAIT_S)

    await db.set_session(
        sess["id"], status="complete", audio_path=str(out_path),
        audio_seconds=seconds, generated_at=_now(),
    )


async def _render_waiting_out_busy(
    project: dict, sess: Any, out_path: Path, voice: str, instruct_text: Optional[str],
) -> int:
    """One recording of a lesson, backing off for as long as the service says "busy"."""
    for attempt in range(_BUSY_RETRIES):
        try:
            return await render_script_to_audio(
                script=sess["script"], out_path=out_path,
                user_id=project["user_id"], voice=voice,
                instruct_text=instruct_text,
            )
        except VoiceServiceBusy as busy:
            if attempt == _BUSY_RETRIES - 1:
                raise
            wait = min(busy.retry_after, _BUSY_MAX_WAIT_S)
            log.info(
                "voice service busy, backing off %ss (session %s, attempt %d/%d)",
                wait, sess["id"], attempt + 1, _BUSY_RETRIES,
            )
            await asyncio.sleep(wait)
    raise VoiceServiceBusy()


# --------------------------------------------------------------- delivery phases
async def _stage_label(project_id: int, detail: str) -> str:
    """The human line under the progress bar, phase-aware where phases exist.

    Also promotes the active phase from 'queued' to 'processing' the first time
    real work is done on it. 'queued' means the listener has asked but the worker
    has not arrived; without this the rail would read "Starting" for the whole
    three hours.
    """
    active = await db.active_phase(project_id)
    if active is None:
        return detail[:1].upper() + detail[1:]
    if active["status"] == "queued":
        await db.set_phase(active["id"], status="processing")
    total = await db.count_phases(project_id)
    return f"Phase {active['ordinal']} of {total} · {detail}"


def _course_progress(sessions: list[Any]) -> int:
    """Whole-course completion, 55-99.

    Weighted across every lesson in the book rather than the current batch: a
    lesson counts half once its script is written and whole once its audio
    exists. That is the only weighting that stays monotonic when phases are
    delivered one at a time, when "process everything" runs them back to back,
    and when a single lesson is regenerated after the fact.
    """
    total = len(sessions) or 1
    done = sum(
        1.0 if s["status"] == "complete" else 0.5 if s["status"] == "validated" else 0.0
        for s in sessions
    )
    return min(99, 55 + int(44 * done / total))


async def _rollup(project_id: int) -> None:
    """Refresh the course-level totals from the lessons that exist so far, so the
    numbers a listener sees between phases are real rather than final-only."""
    sessions = await db.get_sessions(project_id)
    await db.set_project(
        project_id,
        total_audio_seconds=sum(int(s["audio_seconds"] or 0) for s in sessions),
        validation_status=(
            "flagged" if any(s["validation_status"] == "flagged" for s in sessions) else "passed"
        ),
    )


async def _close_active_phase(project: dict) -> None:
    """The active delivery phase has recorded its last lesson.

    Bank it, then either roll straight into the next one or park until the
    listener asks for it.
    """
    pid = int(project["id"])
    active = await db.active_phase(pid)

    if active is None:
        # No phase is in flight, so we did not arrive here at the end of a batch.
        # This is a per-session Regenerate that re-opened a project which was
        # already parked or already finished. Put it back the way it was rather
        # than walking the phase pointer forward and re-offering delivered work.
        await _restore_after_regenerate(project)
        return

    await db.set_phase(
        active["id"], status="complete", completed_at=_now(),
        audio_seconds=await db.phase_audio_seconds(pid, int(active["ordinal"])),
    )
    await _rollup(pid)

    nxt = await db.next_phase_after(pid, int(active["ordinal"]))
    if nxt is None:
        await db.set_project(pid, phase="finalize", stage="Finalizing assets", progress=98)
        return

    total = await db.count_phases(pid)

    if project.get("auto_continue"):
        # The listener asked for the rest of the book up front. Read at the
        # boundary rather than cached, so turning it off takes effect here.
        await db.set_phase(nxt["id"], status="queued", requested_at=_now())
        await db.set_project(
            pid, phase="sessions", stage=f"Phase {nxt['ordinal']} of {total} · starting",
        )
        return

    await db.set_phase(nxt["id"], status="ready")
    await db.set_project(
        pid, status="awaiting_user", phase="awaiting_user",
        stage=f"Phase {active['ordinal']} of {total} ready to play",
    )


async def _restore_after_regenerate(project: dict) -> None:
    """Return a project to the state a Regenerate interrupted.

    Regenerating one lesson re-opens the project so the worker picks the lesson
    up. Once it is done there is no active phase to close, and the project must
    go back to whatever it was: finished, or parked mid-course. Getting this
    wrong would silently hand the listener a phase they never asked for.
    """
    pid = int(project["id"])
    phases = await db.get_phases(pid)

    if not phases or all(p["status"] == "complete" for p in phases):
        await _phase_finalize(project)   # idempotent; recomputes from the lessons
        return

    total = len(phases)
    delivered = sum(1 for p in phases if p["status"] == "complete")
    await _rollup(pid)
    await db.set_project(
        pid, status="awaiting_user", phase="awaiting_user",
        stage=f"Phase {delivered} of {total} ready to play",
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
