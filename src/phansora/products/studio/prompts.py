"""Prompts for Narrava Studio narration.

Two audiences, deliberately different:

  * ``deepseek-chat`` does the bulk work (condensing a chapter). Cheap, fast, mechanical.
  * ``deepseek-reasoner`` does the judgement (which chapters carry a video, how the
    narration should be shaped). Slow and expensive, so it is only ever handed a condensed
    view of the book, never raw chapter text.

Every structured prompt asks for JSON *in the prompt body*, because the reasoning model has
no JSON mode — see shared/ai/deepseek_reasoner.py.
"""
from __future__ import annotations

from .segment import WORDS_PER_MINUTE

_VOICE = (
    "You write narration for documentary video: spoken aloud, second-person-free, "
    "plain and concrete. No headings, no bullet points, no stage directions, no 'in this "
    "video'. Never invent specifics (names, dates, numbers) that were not given to you."
)

# --------------------------------------------------------------------- write from a prompt
SCRIPT_SYSTEM = (
    _VOICE
    + " Divide the narration into BEATS. A beat is one continuous thought a narrator "
    "delivers without pausing — usually two to four sentences. Beat boundaries are a "
    "pacing decision: break where the viewer needs a moment, or where the picture would cut."
    '\n\nReturn: {"title": str, "beats": [str, ...]}'
)


def script_user(prompt: str, *, style: str = "documentary", tone: str | None = None,
                target_duration_sec: float | None = None) -> str:
    parts = [f"Subject of the video:\n{prompt.strip()}", f"\nStyle: {style}"]
    if tone:
        parts.append(f"Tone: {tone}")
    if target_duration_sec:
        words = int((target_duration_sec / 60) * WORDS_PER_MINUTE)
        parts.append(
            f"Target length: about {int(target_duration_sec)} seconds when read aloud, "
            f"which is roughly {words} words in total. Treat this as a budget for the whole "
            "narration, not per beat."
        )
    return "\n".join(parts)


# ------------------------------------------------------------------ condense one chapter
CHAPTER_SUMMARY_SYSTEM = (
    "You condense one chapter of a book so an editor can decide whether it belongs in a "
    "video. Report only what the chapter actually contains. State what happens or is "
    "argued, who and what it involves, and what is concrete enough to show on screen. "
    "Four sentences at most. Plain prose, no preamble."
)


def chapter_summary_user(title: str, text: str, *, limit: int = 12000) -> str:
    body = text[:limit]
    truncated = "\n\n[chapter truncated for length]" if len(text) > limit else ""
    return f"Chapter: {title or 'Untitled'}\n\n{body}{truncated}"


# ----------------------------------------------------------- choose chapters to narrate
CHAPTER_RANK_SYSTEM = (
    "You are helping an editor choose which chapters of a book to turn into narrated video. "
    "You are given every chapter in order, condensed. Judge each on whether it would carry a "
    "video: is there a story, a turn, a concrete stake, something that can be SHOWN? "
    "Recommend the ones that would, in reading order. Do not recommend everything — a book "
    "that is mostly setup or apparatus should yield only its strongest few chapters. "
    "Be specific about why: 'covers the trial and the verdict' beats 'interesting chapter'."
    '\n\nReturn: {"suggested_title": str, "chapters": [{"index": int, "recommended": bool, '
    '"why": str}, ...]} — one entry per chapter given, same indexes, same order.'
)


def chapter_rank_user(book_title: str, summaries: list[dict]) -> str:
    lines = [f"Book: {book_title or 'Untitled'}", f"Chapters: {len(summaries)}", ""]
    for item in summaries:
        lines.append(f"[{item['index']}] {item['title'] or 'Untitled'}")
        lines.append(f"    words: {item.get('word_count', 0)}")
        lines.append(f"    {item.get('summary', '').strip() or '(no summary available)'}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------- write narration from chosen chapters
FROM_CHAPTERS_SYSTEM = (
    _VOICE
    + " You are adapting chosen chapters of a book into narration. Stay grounded in the "
    "supplied summaries — this is an adaptation, not an invention. Carry the through-line "
    "across chapters so it plays as one piece, not a list of recaps."
    "\n\nDivide the narration into BEATS: one continuous thought each, usually two to four "
    "sentences, broken where the picture would cut."
    '\n\nReturn: {"title": str, "beats": [str, ...]}'
)


def from_chapters_user(book_title: str, chapters: list[dict], *,
                       target_duration_sec: float | None = None) -> str:
    lines = [f"Book: {book_title or 'Untitled'}", "", "Chapters to adapt, in order:", ""]
    for ch in chapters:
        lines.append(f"— {ch.get('title') or 'Untitled'}")
        lines.append(f"  {(ch.get('summary') or '').strip()}")
        lines.append("")
    if target_duration_sec:
        words = int((target_duration_sec / 60) * WORDS_PER_MINUTE)
        lines.append(
            f"Target length: about {int(target_duration_sec)} seconds read aloud "
            f"(~{words} words total across all beats)."
        )
    return "\n".join(lines)
