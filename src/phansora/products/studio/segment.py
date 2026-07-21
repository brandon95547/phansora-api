"""Prose -> the timed-beat narration script the Studio speaks.

    {title, full_text, estimated_duration_sec, segments: [{start_sec, end_sec, text}]}

This MIRRORS ``assets/js/admin/studio/util/script.js`` in the Node app: the editor segments
locally so the Write tab works offline, and the API segments so AI/ebook output arrives in
the same shape. The two must stay in step — same constants, same rules — or a script would
re-time differently depending on which side produced it.

Timing is an estimate from word count. Real durations only exist once audio is synthesized.
"""
from __future__ import annotations

import re
from typing import Any

WORDS_PER_MINUTE = 150  # unhurried documentary narration
MIN_BEAT_WORDS = 18     # below this a beat reads as a fragment
MIN_BEAT_SEC = 1.5      # a beat still needs to be clickable on the timeline

# Abbreviations that end in a period mid-sentence. Splitting after these is the single most
# common way a naive splitter mangles narration ("Dr. Reed" -> two beats).
ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "st", "jr", "sr", "prof", "rev", "gen", "col", "sgt", "lt", "capt",
    "no", "fig", "vol", "ch", "pp", "ed", "est", "approx", "vs", "etc", "inc", "ltd", "co",
}

# Fixed-width lookbehind only — Python's `re` rejects variable-width. The optional closing
# quote/bracket is captured instead, and healing happens in a second pass below.
_BOUNDARY = re.compile(r'(?<=[.!?])(?P<close>["\'\)\]]*)\s+(?=["\'\(\[]*[A-Z0-9])')
_INITIALS = re.compile(r"^([a-z]\.)+[a-z]$")


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def speaking_seconds(text: str, wpm: int = WORDS_PER_MINUTE) -> float:
    return max(MIN_BEAT_SEC, (word_count(text) / max(1, wpm)) * 60)


def _dangling_abbreviation(piece: str) -> bool:
    """True when `piece` ends on something that is not really a sentence end: a known
    abbreviation, or an initial like "J." — both legitimately followed by a capital."""
    tail = re.findall(r"\S+$", (piece or "").strip())
    if not tail:
        return False
    last = re.sub(r'[)"\'\]]+$', "", tail[0].lower())
    if not last.endswith("."):
        return False
    stem = last[:-1]
    return stem in ABBREVIATIONS or len(stem) == 1 and stem.isalpha() or bool(_INITIALS.match(stem))


def sentences(paragraph: str) -> list[str]:
    flat = re.sub(r"\s+", " ", paragraph or "").strip()
    if not flat:
        return []

    raw: list[str] = []
    last = 0
    for m in _BOUNDARY.finditer(flat):
        raw.append(flat[last:m.end("close")].strip())
        last = m.end()
    tail = flat[last:].strip()
    if tail:
        raw.append(tail)
    raw = [s for s in raw if s]

    # Heal splits that landed after an abbreviation.
    out: list[str] = []
    for piece in raw:
        if out and _dangling_abbreviation(out[-1]):
            out[-1] += f" {piece}"
        else:
            out.append(piece)
    return out or [flat]


def _beats(text: str) -> list[str]:
    """Group sentences into beats. A blank line always ends a beat — the writer's own
    pacing beats an algorithm — and within a paragraph sentences accumulate until the beat
    can stand on its own."""
    beats: list[str] = []
    for para in [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]:
        buf: list[str] = []
        for sentence in sentences(para):
            buf.append(sentence)
            if word_count(" ".join(buf)) >= MIN_BEAT_WORDS:
                beats.append(" ".join(buf))
                buf = []
        if buf:
            tail = " ".join(buf)
            # A short tail merges back rather than becoming a stub — unless it is the
            # paragraph's only content, which the writer clearly meant to stand alone.
            if beats and word_count(tail) < MIN_BEAT_WORDS / 2:
                beats[-1] += f" {tail}"
            else:
                beats.append(tail)
    return beats


def _time(beats: list[str], wpm: int) -> tuple[list[dict], float]:
    cursor = 0.0
    segments: list[dict] = []
    for text in beats:
        dur = speaking_seconds(text, wpm)
        segments.append({
            "start_sec": round(cursor, 2),
            "end_sec": round(cursor + dur, 2),
            "text": text,
        })
        cursor += dur
    return segments, round(cursor, 2)


def segment_script(text: str, *, title: str = "", wpm: int = WORDS_PER_MINUTE) -> dict[str, Any]:
    segments, total = _time(_beats(text), wpm)
    return {
        "title": (title or "").strip() or "Untitled narration",
        "full_text": "\n\n".join(s["text"] for s in segments),
        "estimated_duration_sec": total,
        "segments": segments,
    }


def script_from_beats(beats: list[str], *, title: str = "", wpm: int = WORDS_PER_MINUTE) -> dict[str, Any]:
    """Build the script from beats the MODEL already divided, keeping its boundaries.

    Used when the LLM returns a beat list: its paragraphing is a deliberate editorial
    choice about pacing, so re-segmenting it would throw away the better decision.
    """
    cleaned = [re.sub(r"\s+", " ", b).strip() for b in beats if b and b.strip()]
    segments, total = _time(cleaned, wpm)
    return {
        "title": (title or "").strip() or "Untitled narration",
        "full_text": "\n\n".join(s["text"] for s in segments),
        "estimated_duration_sec": total,
        "segments": segments,
    }
