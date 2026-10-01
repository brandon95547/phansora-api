"""Turn parsed blocks into source-referenced chunks.

Each chunk aggregates consecutive blocks up to ``max_chars`` and carries merged
provenance (chapter/section/page range + char offsets into the normalized text)
so every downstream concept, session and validation can point back to exactly
where it came from. Oversized single blocks are split with the existing
``txt_to_voice.utils.chunking.chunk_text`` helper.

A chunk also carries ``teachable``. Blocks the parser classified as apparatus —
contents, indexes, running heads, copyright pages — are kept (nothing is deleted
from a source the reader paid to convert) but marked, and the pipeline neither
indexes nor narrates them. Prose and apparatus never share a chunk, so the mark
is always unambiguous.

A chunk boundary is also the only place a lesson can begin or end (lessons are
runs of whole chunks), so where chunks end is where lessons can stop. Two rules
follow from that, both below: a chunk does not end on half a sentence, and it
ends on a chapter change when it is big enough to.
"""
from __future__ import annotations

import logging
from typing import Optional

from phansora.shared.utils.chunking import chunk_text  # reuse existing splitter

from .parsers import Block, ParsedDoc
from .sentences import ends_complete, last_sentence_end

log = logging.getLogger("book_alchemy.chunking")

# The most of a source the apparatus filter may claim before its verdict is
# thrown away wholesale. A book really can be 20% front matter; one that reads as
# 60% apparatus means the heuristic has misfired on this document's layout, and
# narrating some contents pages is a far better failure than silently dropping
# half the book the reader uploaded.
APPARATUS_MAX_SHARE = 0.25

# The longest unfinished tail a chunk will hand on to the next one. A PDF is read
# a page at a time, so a paragraph that runs over the page arrives as two blocks
# and a chunk that fills up between them used to end mid-sentence — and a lesson
# that ended on that chunk ended there too. Anything longer than this is not a
# sentence caught by a page break (it is verse, a table, a list with no full
# stops), and is left where it is.
CARRY_MAX_CHARS = 600

# How full a chunk must be before a change of chapter closes it. Without a floor,
# a document with a heading every hundred words would become hundreds of tiny
# chunks, each one a model call in the analyze phase.
CHAPTER_FLUSH_SHARE = 0.5


def build_chunks(doc: ParsedDoc, *, max_chars: int = 4000) -> list[dict]:
    blocks = _apply_apparatus_verdict(doc.blocks)

    chunks: list[dict] = []
    ordinal = 0
    char_cursor = 0

    pending: list[Block] = []
    pending_len = 0

    def flush(next_block: Optional[Block] = None) -> None:
        """Close the pending chunk. Given the block that did not fit, an
        unfinished last sentence is held back to open the next chunk with it."""
        nonlocal ordinal, char_cursor, pending, pending_len
        if not pending:
            return
        text = "\n\n".join(b.text for b in pending).strip()
        carried: Optional[Block] = None
        if next_block is not None and _same_run(pending[-1], next_block):
            text, tail = _hold_back_unfinished(text)
            if tail:
                last = pending[-1]
                carried = Block(
                    text=tail, chapter=last.chapter, section=last.section,
                    page_start=last.page_end, page_end=last.page_end, kind=last.kind,
                )
        if text:
            chunks.append(_make_chunk(ordinal, text, pending, char_cursor))
            ordinal += 1
            char_cursor += len(text) + 2
        pending = [carried] if carried else []
        pending_len = len(carried.text) + 2 if carried else 0

    for block in blocks:
        btext = (block.text or "").strip()
        if not btext:
            continue
        # A chunk is entirely teachable or entirely not; never a blend.
        if pending and pending[0].kind != block.kind:
            flush()
        # A new chapter is the best stopping point a source has, so let it be a
        # boundary rather than something buried mid-chunk.
        if (
            pending
            and _chapter_key(block) != _chapter_key(pending[-1])
            and pending_len >= max_chars * CHAPTER_FLUSH_SHARE
        ):
            flush()
        if len(btext) > max_chars:
            flush()
            for piece in chunk_text(btext, max_chars):
                piece = piece.strip()
                if not piece:
                    continue
                sub = Block(
                    text=piece, chapter=block.chapter, section=block.section,
                    page_start=block.page_start, page_end=block.page_end,
                    kind=block.kind,
                )
                chunks.append(_make_chunk(ordinal, piece, [sub], char_cursor))
                ordinal += 1
                char_cursor += len(piece) + 2
            continue

        if pending_len + len(btext) > max_chars:
            flush(next_block=block)
        pending.append(block)
        pending_len += len(btext) + 2

    flush()
    return chunks


def _chapter_key(block: Block) -> str:
    return (block.chapter or "").strip().casefold()


def _same_run(last: Block, nxt: Block) -> bool:
    """Whether ``nxt`` carries straight on from ``last``: same kind, same chapter.

    Only then may a tail move forward. Across a chapter change it would open the
    new chapter's chunk with the old chapter's words (and its chapter label),
    and across a kind change it would put prose in an apparatus chunk.
    """
    return last.kind == nxt.kind and _chapter_key(last) == _chapter_key(nxt)


def _hold_back_unfinished(text: str) -> tuple[str, str]:
    """``(text up to its last finished sentence, the unfinished tail)``.

    The tail is empty — and the text untouched — when the text already ends on a
    sentence, has no finished sentence to fall back to, or trails off for longer
    than CARRY_MAX_CHARS.
    """
    if ends_complete(text):
        return text, ""
    cut = last_sentence_end(text)
    tail = text[cut:].strip()
    if cut <= 0 or not tail or len(tail) > CARRY_MAX_CHARS:
        return text, ""
    return text[:cut].rstrip(), tail


def _apply_apparatus_verdict(blocks: list[Block]) -> list[Block]:
    """Honor the parser's apparatus marks, unless there are implausibly many.

    Returns the blocks unchanged when the share is sane, and a copy with every
    mark cleared when it is not. See APPARATUS_MAX_SHARE.
    """
    total = sum(len(b.text or "") for b in blocks)
    if not total:
        return blocks
    flagged = sum(len(b.text or "") for b in blocks if b.kind == "apparatus")
    share = flagged / total
    if share <= APPARATUS_MAX_SHARE:
        if flagged:
            log.info(
                "Apparatus filter: %s of %s chars (%.1f%%) marked as front/back matter",
                flagged, total, share * 100,
            )
        return blocks

    log.warning(
        "Apparatus filter would drop %.1f%% of this source (%s of %s chars), which is "
        "above the %.0f%% ceiling — keeping everything. The document's layout probably "
        "does not match the reference-line heuristics.",
        share * 100, flagged, total, APPARATUS_MAX_SHARE * 100,
    )
    return [
        Block(
            text=b.text, chapter=b.chapter, section=b.section,
            page_start=b.page_start, page_end=b.page_end, kind="prose",
        )
        for b in blocks
    ]


def _make_chunk(ordinal: int, text: str, blocks: list[Block], char_start: int) -> dict:
    chapters = [b.chapter for b in blocks if b.chapter]
    sections = [b.section for b in blocks if b.section]
    pages = [p for b in blocks for p in (b.page_start, b.page_end) if p is not None]
    return {
        "ordinal": ordinal,
        "text": text,
        "chapter": _first(chapters),
        "section": _first(sections),
        "page_start": min(pages) if pages else None,
        "page_end": max(pages) if pages else None,
        "char_start": char_start,
        "char_end": char_start + len(text),
        "teachable": blocks[0].kind != "apparatus",
    }


def _first(values: list[str]) -> Optional[str]:
    return values[0] if values else None
