"""Narrava Studio API — narration authoring.

Mounted at ``/studio`` on the consolidated app (see phansora/main.py). Three jobs:

  * turn a prompt into a timed narration script      POST /script/generate
  * turn pasted prose into a timed narration script  POST /script/segment
  * read a book and say which chapters to narrate    POST /ebook/analyze  (+ poll)
    then adapt the chosen ones                       POST /script/from-chapters

Everything returns the same script shape the editor already speaks — see segment.py.

Ebook parsing is Book Alchemy's (`products/book_alchemy/parsers.py`): it already handles
pdf/epub/mobi/azw/docx/txt/markdown/html and tags every block with the chapter it came
from, which is exactly the structure this needs. Its heavyweight analysis pipeline is NOT
reused: choosing chapters needs a condensed view of the book, not a full comprehension pass.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from phansora.shared.ai.deepseek_reasoner import DeepSeekReasoner  # noqa: E402

from .jobs import JobRegistry  # noqa: E402
from . import prompts  # noqa: E402
from .segment import script_from_beats, segment_script, word_count  # noqa: E402

logger = logging.getLogger("phansora.studio")

# Extension -> the format name book_alchemy's parser dispatches on.
EBOOK_FORMATS = {
    ".pdf": "pdf", ".epub": "epub", ".mobi": "mobi", ".azw": "mobi", ".azw3": "mobi",
    ".docx": "docx", ".txt": "txt", ".md": "markdown", ".markdown": "markdown",
    ".html": "html", ".htm": "html",
}

# Chapters shorter than this are front/back matter (title pages, dedications, colophons) —
# summarizing them wastes a model call and pollutes the ranking with noise.
MIN_CHAPTER_WORDS = 150
# Bound on concurrent summary calls: enough to finish a long book quickly, low enough not to
# trip provider rate limits when several users analyse at once.
SUMMARY_CONCURRENCY = 6
MAX_CHAPTERS = 120

app = FastAPI(
    title="Narrava Studio API",
    description="Narration authoring: prompt-to-script, prose-to-beats, and ebook chapter analysis.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"],
)
app.state.jobs = JobRegistry()


# ------------------------------------------------------------------------ models
class ScriptGenerateRequest(BaseModel):
    prompt: str = Field(min_length=3)
    style: str = "documentary"
    tone: Optional[str] = None
    target_duration_sec: Optional[float] = None


class ScriptSegmentRequest(BaseModel):
    script_text: str = Field(min_length=1)
    title: Optional[str] = None
    words_per_minute: Optional[int] = None


class FromChaptersRequest(BaseModel):
    book_title: Optional[str] = None
    chapters: list[dict[str, Any]] = Field(default_factory=list)
    target_duration_sec: Optional[float] = None


# ------------------------------------------------------------------------ routes
@app.get("/")
def root() -> dict:
    return {"product": "narrava_studio", "status": "ok", "version": app.version}


@app.get("/health")
def health() -> dict:
    reasoner = DeepSeekReasoner.reasoning()
    return {
        "status": "ok",
        "reasoning_model": reasoner.model,
        "reasoning_mode": reasoner.reasons,
        "fast_model": DeepSeekReasoner.fast().model,
        "deepseek_key": bool(os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_CHAT_API_KEY")),
    }


@app.post("/script/segment")
def script_segment(req: ScriptSegmentRequest) -> dict:
    """Pure text -> beats. No model involved, so it stays instant and free."""
    script = segment_script(
        req.script_text,
        title=req.title or "",
        wpm=req.words_per_minute or prompts.WORDS_PER_MINUTE,
    )
    if not script["segments"]:
        raise HTTPException(status_code=400, detail="No narration text found.")
    return {"script": script}


@app.post("/script/generate")
async def script_generate(req: ScriptGenerateRequest) -> dict:
    """A prompt -> a narration script, written by the reasoning model."""
    data = await _beats_from_model(
        system=prompts.SCRIPT_SYSTEM,
        user=prompts.script_user(
            req.prompt, style=req.style, tone=req.tone,
            target_duration_sec=req.target_duration_sec,
        ),
    )
    return {"script": script_from_beats(data["beats"], title=data["title"])}


@app.post("/script/from-chapters")
async def script_from_chapters(req: FromChaptersRequest) -> dict:
    """Chosen chapter summaries -> a narration script covering them as one piece."""
    chapters = [c for c in req.chapters if (c.get("summary") or "").strip()]
    if not chapters:
        raise HTTPException(status_code=400, detail="Select at least one chapter with a summary.")
    data = await _beats_from_model(
        system=prompts.FROM_CHAPTERS_SYSTEM,
        user=prompts.from_chapters_user(
            req.book_title or "", chapters, target_duration_sec=req.target_duration_sec,
        ),
        fallback_title=req.book_title or "",
    )
    return {"script": script_from_beats(data["beats"], title=data["title"])}


@app.post("/ebook/analyze")
async def ebook_analyze(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
) -> dict:
    """Submit a book for chapter analysis. Returns a job id to poll — this takes minutes."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    fmt = EBOOK_FORMATS.get(ext)
    if not fmt:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported ebook format '{ext or 'unknown'}'. Supported: "
                   + ", ".join(sorted({e.lstrip('.') for e in EBOOK_FORMATS})),
        )

    # Persist the upload before returning: the request body is gone once we respond, and
    # the job reads it long after. Cleaned up by the job itself.
    fd, path = tempfile.mkstemp(suffix=ext, prefix="studio_ebook_")
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := await file.read(1024 * 1024):
                out.write(chunk)
    except Exception:
        os.unlink(path)
        raise

    book_title = (title or os.path.splitext(file.filename or "")[0] or "Untitled").strip()
    job = app.state.jobs.submit(lambda handle: _run_ebook_analysis(handle, path, fmt, book_title))
    return {"job_id": job.id, "status": job.status}


@app.get("/ebook/jobs/{job_id}")
def ebook_job(job_id: str) -> dict:
    job = app.state.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired.")
    return job.to_dict()


# ------------------------------------------------------------------------ internals
async def _beats_from_model(*, system: str, user: str, fallback_title: str = "") -> dict:
    """Ask the reasoning model for {title, beats[]} and validate it before use.

    A model can return the right JSON with the wrong contents (empty beats, a string where
    a list belongs). Catching that here means the caller only ever sees a usable script.
    """
    client = DeepSeekReasoner.reasoning()
    try:
        data = await client.chat_json(system=system, user=user, max_output_tokens=4000)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Studio script generation failed")
        raise HTTPException(status_code=502, detail=f"Narration model failed: {exc}") from exc

    if isinstance(data, list):  # model returned a bare beat list
        data = {"beats": data}
    beats = data.get("beats") if isinstance(data, dict) else None
    if isinstance(beats, str):
        beats = [beats]
    beats = [b for b in (beats or []) if isinstance(b, str) and b.strip()]
    if not beats:
        raise HTTPException(status_code=502, detail="The narration model returned no usable text.")

    title = (data.get("title") if isinstance(data, dict) else None) or fallback_title
    return {"title": str(title or "").strip(), "beats": beats}


def _chapters_from_doc(doc) -> list[dict]:
    """Group the parser's blocks into chapters, preserving reading order.

    Blocks carry a best-effort `chapter` from their nearest heading; consecutive blocks
    sharing one belong together. A book with no headings at all yields a single chapter,
    which is correct — there is nothing to choose between.
    """
    chapters: list[dict] = []
    for block in doc.blocks:
        name = (getattr(block, "chapter", None) or "").strip()
        text = (block.text or "").strip()
        if not text:
            continue
        if chapters and chapters[-1]["title"] == name:
            chapters[-1]["parts"].append(text)
        else:
            chapters.append({"title": name, "parts": [text]})

    out = []
    for i, ch in enumerate(chapters):
        body = "\n\n".join(ch["parts"])
        out.append({
            "index": i,
            "title": ch["title"] or f"Section {i + 1}",
            "text": body,
            "word_count": word_count(body),
        })
    return out


async def _run_ebook_analysis(handle, path: str, fmt: str, book_title: str) -> dict:
    from phansora.products.book_alchemy.parsers import parse_source

    try:
        handle.progress(5, "Reading the file")
        doc = await asyncio.to_thread(
            parse_source, source_format=fmt, path=path, title_hint=book_title,
        )

        chapters = _chapters_from_doc(doc)
        substantive = [c for c in chapters if c["word_count"] >= MIN_CHAPTER_WORDS][:MAX_CHAPTERS]
        if not substantive:
            raise RuntimeError(
                "No readable chapters were found. If this is a scanned PDF it has no "
                "extractable text and needs OCR first."
            )
        if len(chapters) > MAX_CHAPTERS:
            logger.warning("Book had %d chapters; analysing the first %d", len(chapters), MAX_CHAPTERS)

        # Condense each chapter with the FAST model — bulk work, no judgement. Bounded
        # concurrency keeps a 60-chapter book to about a minute without hammering the API.
        handle.progress(15, f"Reading {len(substantive)} chapters")
        fast = DeepSeekReasoner.fast()
        gate = asyncio.Semaphore(SUMMARY_CONCURRENCY)
        done = 0

        async def summarize(ch: dict) -> dict:
            nonlocal done
            async with gate:
                try:
                    summary = await fast.chat(
                        system=prompts.CHAPTER_SUMMARY_SYSTEM,
                        user=prompts.chapter_summary_user(ch["title"], ch["text"]),
                        max_output_tokens=400,
                    )
                except Exception as exc:  # noqa: BLE001
                    # One bad chapter must not sink the book; it just ranks poorly.
                    logger.warning("Chapter %s summary failed: %s", ch["index"], exc)
                    summary = ""
            done += 1
            handle.progress(15 + int(65 * done / len(substantive)), f"Read {done}/{len(substantive)} chapters")
            return {**ch, "summary": summary.strip()}

        summarized = await asyncio.gather(*(summarize(c) for c in substantive))

        handle.progress(85, "Choosing chapters to narrate")
        ranked = await _rank_chapters(book_title, summarized)

        return {
            "book_title": ranked.get("suggested_title") or book_title,
            "chapter_count": len(chapters),
            "analyzed_count": len(summarized),
            "chapters": ranked["chapters"],
        }
    finally:
        # The upload is temp state; drop it whether the job succeeded or not.
        try:
            os.unlink(path)
        except OSError:
            pass


async def _rank_chapters(book_title: str, summarized: list[dict]) -> dict:
    """One reasoning call over the condensed book. Degrades to 'nothing recommended' rather
    than failing the whole analysis — the summaries are still useful on their own."""
    payload = [
        {"index": c["index"], "title": c["title"], "summary": c["summary"], "word_count": c["word_count"]}
        for c in summarized
    ]
    verdicts: dict[int, dict] = {}
    suggested_title = ""
    try:
        data = await DeepSeekReasoner.reasoning().chat_json(
            system=prompts.CHAPTER_RANK_SYSTEM,
            user=prompts.chapter_rank_user(book_title, payload),
            max_output_tokens=4000,
        )
        if isinstance(data, dict):
            suggested_title = str(data.get("suggested_title") or "").strip()
            for row in data.get("chapters") or []:
                if isinstance(row, dict) and isinstance(row.get("index"), int):
                    verdicts[row["index"]] = row
    except Exception as exc:  # noqa: BLE001
        logger.warning("Chapter ranking failed, returning unranked chapters: %s", exc)

    chapters = []
    for c in summarized:
        v = verdicts.get(c["index"], {})
        chapters.append({
            "index": c["index"],
            "title": c["title"],
            "summary": c["summary"],
            "word_count": c["word_count"],
            "recommended": bool(v.get("recommended")),
            "why": str(v.get("why") or "").strip(),
        })
    return {"suggested_title": suggested_title, "chapters": chapters}
