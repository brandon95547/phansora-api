"""Script generation and segmentation.

Two entry points:
  - ``generate_script`` — prompt -> narrator-formatted script (via the LLM).
  - ``segment_script`` — any script text -> timed beats with per-beat keywords.

Segmentation is pure/deterministic (no LLM): it splits the narration into
sentence-level beats and estimates each beat's start/end from its word count at a
words-per-minute pace. Those timings are what let the timeline place each media
clip at the moment the narration talks about it.
"""
from __future__ import annotations

import re
import uuid
from typing import List

from .. import config
from ..models import Script, ScriptSegment
from . import llm

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
# Split on sentence-ending punctuation followed by whitespace. Good enough for
# narration prose; abbreviations are rare in scripts and a stray split is harmless.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Short, common words that never make useful media-search terms.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "as", "at", "by", "from", "into", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "their", "there", "here",
    "we", "you", "they", "he", "she", "his", "her", "our", "your", "them", "us",
    "not", "no", "so", "if", "then", "than", "when", "while", "which", "who",
    "what", "how", "why", "will", "would", "can", "could", "may", "might", "have",
    "has", "had", "do", "does", "did", "about", "over", "under", "up", "down",
    "out", "very", "just", "more", "most", "some", "any", "all", "one", "also",
}


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _estimate_seconds(text: str, wpm: int) -> float:
    words = _word_count(text)
    if words == 0:
        return 0.0
    return round(words / max(1, wpm) * 60.0, 2)


def extract_keywords(text: str, limit: int = 5) -> List[str]:
    """Heuristic keywords for media search: proper nouns first, then salient words."""
    tokens = _WORD_RE.findall(text or "")
    proper: List[str] = []
    other: List[str] = []
    seen = set()
    for i, tok in enumerate(tokens):
        low = tok.lower()
        if low in _STOPWORDS or len(tok) < 3:
            continue
        if low in seen:
            continue
        seen.add(low)
        # Capitalised mid-sentence -> likely a proper noun / named entity.
        if tok[0].isupper() and i > 0:
            proper.append(tok)
        else:
            other.append(tok)
    ordered = proper + other
    return ordered[:limit]


def segment_script(
    full_text: str,
    *,
    title: str = "",
    wpm: int | None = None,
    source: str = "provided",
) -> Script:
    """Split narration text into timed beats with keywords."""
    settings = config.get_settings()
    wpm = wpm or settings.narrava_words_per_minute
    text = (full_text or "").strip()

    raw_sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]

    # Merge very short fragments into the previous sentence so a beat is long
    # enough to warrant its own clip (avoids one-word beats like "Right.").
    beats: List[str] = []
    for sentence in raw_sentences:
        if beats and _word_count(sentence) < 4:
            beats[-1] = f"{beats[-1]} {sentence}"
        else:
            beats.append(sentence)

    segments: List[ScriptSegment] = []
    cursor = 0.0
    for index, beat in enumerate(beats):
        dur = _estimate_seconds(beat, wpm)
        segments.append(
            ScriptSegment(
                id=f"seg_{uuid.uuid4().hex[:8]}",
                index=index,
                text=beat,
                start_sec=round(cursor, 2),
                end_sec=round(cursor + dur, 2),
                keywords=extract_keywords(beat),
            )
        )
        cursor += dur

    if not title:
        title = _derive_title(text)

    return Script(
        title=title,
        full_text=text,
        segments=segments,
        estimated_duration_sec=round(cursor, 2),
        source="prompt" if source == "prompt" else "provided",
    )


def _derive_title(text: str) -> str:
    first = _SENTENCE_RE.split(text.strip(), maxsplit=1)[0] if text.strip() else "Untitled Video"
    words = first.split()
    return " ".join(words[:8]) or "Untitled Video"


_SCRIPT_SYSTEM = (
    "You are a professional video narration writer. Given a topic, write a spoken "
    "narration script the way a seasoned documentary narrator would deliver it: "
    "clear, engaging, in flowing prose. Rules:\n"
    "- Output ONLY the words to be spoken. No scene directions, no camera notes, "
    "no speaker labels, no markdown, no headings, no bracketed cues.\n"
    "- Write in short-to-medium sentences that read naturally aloud.\n"
    "- Open with a hook and give the piece a clear beginning, middle and end.\n"
    "- Do not address 'the viewer' with meta commentary about the video itself."
)


def generate_script(
    prompt: str,
    *,
    style: str = "documentary",
    tone: str | None = None,
    target_duration_sec: int | None = None,
    wpm: int | None = None,
) -> Script:
    """Prompt -> narrator-formatted script, then segmented into timed beats."""
    settings = config.get_settings()
    wpm = wpm or settings.narrava_words_per_minute

    instructions = [f"Topic / brief: {prompt.strip()}", f"Narration style: {style}."]
    if tone:
        instructions.append(f"Tone: {tone}.")
    if target_duration_sec:
        target_words = int(target_duration_sec / 60.0 * wpm)
        instructions.append(
            f"Target length: about {target_duration_sec} seconds of narration "
            f"(~{target_words} words). Stay close to this length."
        )
    user = "\n".join(instructions)

    body = llm.generate_text(_SCRIPT_SYSTEM, user, max_output_tokens=2500).strip()
    body = _strip_artifacts(body)

    return segment_script(body, wpm=wpm, source="prompt")


def _strip_artifacts(text: str) -> str:
    """Remove markdown/label artifacts an LLM sometimes adds despite instructions."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        # Drop a leading "Narrator:" / "Title:" style label.
        stripped = re.sub(r"^(narrator|voiceover|vo|title)\s*:\s*", "", stripped, flags=re.I)
        # Drop surrounding markdown emphasis / heading markers.
        stripped = re.sub(r"^#{1,6}\s*", "", stripped)
        stripped = stripped.strip("*_` ")
        lines.append(stripped)
    return "\n".join(lines).strip()
