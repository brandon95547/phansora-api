"""
provenance.py — verbatim grounding for dossier content.

The rule this module enforces: every passage that reaches the dossier body must be a
literal span of a source document. Not a faithful summary of one, not a careful
paraphrase of one — a substring.

Why it has to be code rather than prompt wording. The organizer used to ask the model to
"preserve the source's original meaning and framing", and the model's answer to that was
still generated prose: it decided the words. An instruction to preserve is a hope. Asking
the model to quote and then CHECKING the quote against the source is a guarantee, and the
difference matters most exactly where paraphrase is most tempting — contested claims,
where a reworded sentence quietly becomes a different assertion.

So the model is demoted from author to selector. It says WHERE content belongs; this
module decides what the words are, by copying them out of the source.

    passage from the LLM  ->  locate() in the source  ->  source[start:end]

Anything that cannot be located is not a paraphrase to be tidied up. It is text with no
origin, and it is dropped.

Matching is deliberately forgiving about presentation and strict about content. Curly
quotes, en dashes, non-breaking spaces, line wrapping and letter case all differ between
what a model echoes back and what a PDF extractor produced, and none of them change what
was said. Word choice does, so nothing here tolerates a changed word.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# Presentation-only differences. Folding these lets a passage match its source across the
# curly/straight quote split, the three dash widths, and non-breaking spaces — all of which
# survive a round trip through an LLM inconsistently.
_FOLD = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "―": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", "​": "",
    "ﬁ": "fi", "ﬂ": "fl",   # ligatures, common in PDF extraction
}

_WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)

# A sentence ends at .!? plus any closing quotes/brackets, followed by whitespace. Kept
# deliberately simple: it only ever decides where to EXTEND a span that is already
# grounded, so a missed boundary costs a slightly longer excerpt, never a wrong one.
_SENT_END_RE = re.compile(r"[.!?]['\"’”)\]]*(?=\s)")

_SOURCE_HEADER_RE = re.compile(r"^===\s*SOURCE:.*?===\s*$", re.MULTILINE)


def normalize_with_map(text: str) -> Tuple[str, List[int]]:
    """
    Fold `text` for matching and return the index map back to the original.

    ``index_map[i]`` is the offset in `text` of the character that produced
    ``normalized[i]`` — which is what makes it possible to find a passage in folded space
    and then slice the ORIGINAL, so the dossier carries the source's real characters
    rather than this module's flattened version of them.
    """
    out: List[str] = []
    index_map: List[int] = []
    at_space = True  # leading whitespace collapses away entirely

    for i, ch in enumerate(text):
        folded = _FOLD.get(ch, ch)
        if folded == "":
            continue
        if folded.isspace():
            if at_space:
                continue
            out.append(" ")
            index_map.append(i)
            at_space = True
            continue
        at_space = False
        for c in folded.lower():          # a ligature folds to two characters
            out.append(c)
            index_map.append(i)

    while out and out[-1] == " ":
        out.pop()
        index_map.pop()

    return "".join(out), index_map


def locate(passage: str, source: str, anchor_words: int = 6) -> Optional[Tuple[int, int]]:
    """
    Find `passage` inside `source`. Returns (start, end) in ORIGINAL `source` offsets.

    Two attempts, in order of how much they trust the model:

    1. The whole passage, folded. This is the case where the model quoted properly and
       only presentation drifted.
    2. Head and tail anchors. A model that echoes a long passage will sometimes drop or
       reflow something in the middle while getting the opening and closing words exactly
       right. If both ends are found in order, the span between them is real source text —
       whatever happened in the middle, the characters we return come from the source.

    There is no third attempt. Fuzzy whole-passage matching is what would let a genuinely
    reworded sentence through, which is the one thing this module exists to prevent.
    """
    norm_src, map_src = normalize_with_map(source)
    norm_pas, _ = normalize_with_map(passage)
    if not norm_pas or not norm_src:
        return None

    pos = norm_src.find(norm_pas)
    if pos != -1:
        return map_src[pos], map_src[pos + len(norm_pas) - 1] + 1

    words = _WORD_RE.findall(norm_pas)
    # Too short to anchor safely: with few words, a head/tail pair can match somewhere it
    # does not belong. Short passages have to match outright or not at all.
    if len(words) < anchor_words * 2:
        return None

    head = " ".join(words[:anchor_words])
    tail = " ".join(words[-anchor_words:])

    h = norm_src.find(head)
    if h == -1:
        return None
    t = norm_src.find(tail, h + len(head))
    if t == -1:
        return None

    return map_src[h], map_src[t + len(tail) - 1] + 1


def _block_bounds(source: str, start: int, end: int) -> Tuple[int, int]:
    """
    The paragraph containing the span, and never across a source header.

    Sentence snapping is allowed to reach outward, so it needs a wall to stop at. A blank
    line is that wall; a ``=== SOURCE: x ===`` line is a harder one, because reaching past
    it would pull another document's words into this document's excerpt.
    """
    lo = source.rfind("\n\n", 0, start)
    lo = 0 if lo == -1 else lo + 2
    hi = source.find("\n\n", end)
    hi = len(source) if hi == -1 else hi

    for m in _SOURCE_HEADER_RE.finditer(source):
        if m.end() <= start:
            lo = max(lo, m.end())
        if m.start() >= end:
            hi = min(hi, m.start())
            break

    return lo, hi


def snap_to_sentences(source: str, start: int, end: int) -> Tuple[int, int]:
    """
    Grow a span outward to whole sentences, staying inside its paragraph.

    A span that starts mid-sentence reads as a fragment and, worse, can strip the clause
    that qualified it — "according to the report," or "if the estimate holds," sitting just
    before the part the model picked. Widening to the sentence keeps the qualifier attached
    to the claim it qualifies.
    """
    lo, hi = _block_bounds(source, start, end)
    start = max(start, lo)
    end = min(max(end, start), hi)

    new_start = lo
    for m in _SENT_END_RE.finditer(source, lo, start):
        new_start = m.end()
    while new_start < start and source[new_start].isspace():
        new_start += 1

    m = _SENT_END_RE.search(source, max(end - 1, lo), hi)
    new_end = m.end() if m else hi

    return new_start, min(new_end, hi)


def ground_passage(passage: str, source: str, snap: bool = True) -> Optional[str]:
    """
    Return the verbatim source text behind `passage`, or None if it has no origin.

    None is the important return value: it means the model produced words that are not in
    the source, and the caller's job is to drop that section rather than repair it.
    """
    span = locate(passage, source)
    if span is None:
        return None
    start, end = span
    if snap:
        start, end = snap_to_sentences(source, start, end)
    return source[start:end].strip() or None


def is_grounded(passage: str, source: str) -> bool:
    """Whether `passage` can be traced to `source` at all. Used by the audit."""
    return locate(passage, source) is not None
