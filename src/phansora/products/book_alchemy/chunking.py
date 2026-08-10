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
"""
from __future__ import annotations

import logging
from typing import Optional

from phansora.shared.utils.chunking import chunk_text  # reuse existing splitter

from .parsers import Block, ParsedDoc

log = logging.getLogger("book_alchemy.chunking")

# The most of a source the apparatus filter may claim before its verdict is
# thrown away wholesale. A book really can be 20% front matter; one that reads as
# 60% apparatus means the heuristic has misfired on this document's layout, and
# narrating some contents pages is a far better failure than silently dropping
# half the book the reader uploaded.
APPARATUS_MAX_SHARE = 0.25


def build_chunks(doc: ParsedDoc, *, max_chars: int = 4000) -> list[dict]:
    blocks = _apply_apparatus_verdict(doc.blocks)

    chunks: list[dict] = []
    ordinal = 0
    char_cursor = 0

    pending: list[Block] = []
    pending_len = 0

    def flush() -> None:
        nonlocal ordinal, char_cursor, pending, pending_len
        if not pending:
            return
        text = "\n\n".join(b.text for b in pending).strip()
        if text:
            chunks.append(_make_chunk(ordinal, text, pending, char_cursor))
            ordinal += 1
            char_cursor += len(text) + 2
        pending = []
        pending_len = 0

    for block in blocks:
        btext = (block.text or "").strip()
        if not btext:
            continue
        # A chunk is entirely teachable or entirely not; never a blend.
        if pending and pending[0].kind != block.kind:
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
            flush()
        pending.append(block)
        pending_len += len(btext) + 2

    flush()
    return chunks


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
