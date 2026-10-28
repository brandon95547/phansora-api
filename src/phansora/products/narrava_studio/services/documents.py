"""An imported document, read: its chapters, what each is about, and which would carry a video.

The narration form's Import tab sends research or an eBook here. Parsing is Book Alchemy's
(``products/book_alchemy/parsers.py``), which already reads pdf/epub/mobi/docx/txt/markdown/
html and tags every block with the heading it sits under. Each chapter is condensed by
DeepSeek, then one more call says which chapters would carry a video and why. The result is
what the editor shows as "the topics that were found", and what a narration is later
written from (``from_chapters``).

Slow by nature — minutes for a book — so it runs as a job (services/jobs.py) and is polled.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from . import llm
from .jobs import JobHandle

logger = logging.getLogger("narrava-studio.documents")

# Extension -> the format name Book Alchemy's parser dispatches on.
FORMATS = {
    ".pdf": "pdf", ".epub": "epub", ".mobi": "mobi", ".azw": "mobi", ".azw3": "mobi",
    ".docx": "docx", ".txt": "txt", ".md": "markdown", ".markdown": "markdown",
    ".html": "html", ".htm": "html",
}

# Shorter than this is front or back matter (a title page, a dedication, a colophon):
# condensing it costs a call and puts noise among the topics.
MIN_CHAPTER_WORDS = 150
# Enough at once to read a long book in about a minute; few enough not to trip the
# provider's rate limit when several people import at the same time.
SUMMARY_CONCURRENCY = 6
MAX_CHAPTERS = 120
# A chapter's text as sent to be condensed. A long chapter's opening says what it is about.
SUMMARY_INPUT_CHARS = 12000

SUMMARY_SYSTEM = (
    "You condense one chapter of a document so a video maker can decide whether it belongs "
    "in a narrated video. Report only what the chapter actually contains: what happens or "
    "is argued, who and what it involves, and what is concrete enough to show on screen. "
    "Four sentences at most. Plain prose, no preamble."
)

RANK_SYSTEM = (
    "You help a video maker choose which chapters of a document to turn into narrated "
    "video. You are given every chapter in order, condensed. Judge each on whether it would "
    "carry a video: a story, a turn, a concrete stake, something that can be SHOWN. "
    "Recommend the ones that would, in reading order. Do not recommend everything: a "
    "document that is mostly setup or apparatus should yield only its strongest few. "
    "Be specific about why: 'covers the trial and the verdict' beats 'interesting chapter'."
    '\n\nReturn JSON: {"suggested_title": str, "chapters": [{"index": int, '
    '"recommended": bool, "why": str}, ...]}, one entry per chapter given, same indexes.'
)


def word_count(text: str) -> int:
    return len((text or "").split())


def chapters_from_doc(doc: Any) -> list[dict]:
    """The parser's blocks grouped into chapters, in reading order.

    Blocks carry the heading they sit under; consecutive blocks under one heading belong
    together. A document with no headings at all is one chapter, which is right: there is
    nothing to choose between.
    """
    grouped: list[dict] = []
    for block in getattr(doc, "blocks", None) or []:
        name = (getattr(block, "chapter", None) or "").strip()
        text = (getattr(block, "text", None) or "").strip()
        if not text:
            continue
        if grouped and grouped[-1]["title"] == name:
            grouped[-1]["parts"].append(text)
        else:
            grouped.append({"title": name, "parts": [text]})
    out = []
    for i, ch in enumerate(grouped):
        body = "\n\n".join(ch["parts"])
        out.append({"index": i, "title": ch["title"] or f"Section {i + 1}", "text": body, "word_count": word_count(body)})
    return out


def _summary_user(title: str, text: str) -> str:
    body = text[:SUMMARY_INPUT_CHARS]
    cut = "\n\n[chapter truncated for length]" if len(text) > SUMMARY_INPUT_CHARS else ""
    return f"Chapter: {title or 'Untitled'}\n\n{body}{cut}"


def _rank_user(title: str, summarized: list[dict]) -> str:
    lines = [f"Document: {title or 'Untitled'}", f"Chapters: {len(summarized)}", ""]
    for c in summarized:
        lines += [
            f"[{c['index']}] {c['title'] or 'Untitled'}",
            f"    words: {c.get('word_count', 0)}",
            f"    {c.get('summary', '').strip() or '(no summary available)'}",
            "",
        ]
    return "\n".join(lines)


def rank(title: str, summarized: list[dict]) -> dict:
    """Which chapters would carry a video, and why. Degrades to nothing recommended rather
    than failing the reading: the summaries are still the topics on their own."""
    verdicts: dict[int, dict] = {}
    suggested = ""
    try:
        data = llm.deepseek_json(RANK_SYSTEM, _rank_user(title, summarized), max_output_tokens=4000)
        if isinstance(data, dict):
            suggested = str(data.get("suggested_title") or "").strip()
            for row in data.get("chapters") or []:
                if isinstance(row, dict) and isinstance(row.get("index"), int):
                    verdicts[row["index"]] = row
    except llm.ProviderRejected:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ranking a document's chapters failed; returning them unranked: %s", exc)
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
    return {"suggested_title": suggested, "chapters": chapters}


async def read(handle: JobHandle, path: str, fmt: str, title: str) -> dict:
    """The job: parse, condense each chapter, rank. Removes the upload when it is done."""
    from phansora.products.book_alchemy.parsers import parse_source

    try:
        handle.progress(5, "Reading the file")
        doc = await asyncio.to_thread(parse_source, source_format=fmt, path=path, title_hint=title)
        chapters = chapters_from_doc(doc)
        substantive = [c for c in chapters if c["word_count"] >= MIN_CHAPTER_WORDS][:MAX_CHAPTERS]
        if not substantive:
            raise RuntimeError(
                "No readable text was found. If this is a scanned PDF it has no "
                "extractable text and needs OCR first."
            )

        handle.progress(15, f"Reading {len(substantive)} chapters")
        gate = asyncio.Semaphore(SUMMARY_CONCURRENCY)
        done = 0

        async def condense(ch: dict) -> dict:
            nonlocal done
            async with gate:
                try:
                    summary = await asyncio.to_thread(
                        llm.deepseek_text, SUMMARY_SYSTEM, _summary_user(ch["title"], ch["text"]), max_output_tokens=400,
                    )
                except llm.ProviderRejected:
                    raise
                except Exception as exc:  # noqa: BLE001 — one bad chapter must not sink the document
                    logger.warning("Condensing chapter %s failed: %s", ch["index"], exc)
                    summary = ""
            done += 1
            handle.progress(15 + int(65 * done / len(substantive)), f"Read {done} of {len(substantive)} chapters")
            return {**ch, "summary": (summary or "").strip()}

        summarized = await asyncio.gather(*(condense(c) for c in substantive))

        handle.progress(85, "Choosing the topics")
        ranked = await asyncio.to_thread(rank, title, summarized)
        return {
            "book_title": ranked["suggested_title"] or title,
            "chapter_count": len(chapters),
            "analyzed_count": len(summarized),
            "chapters": ranked["chapters"],
        }
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
