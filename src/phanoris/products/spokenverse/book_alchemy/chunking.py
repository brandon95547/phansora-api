"""Turn parsed blocks into source-referenced chunks.

Each chunk aggregates consecutive blocks up to ``max_chars`` and carries merged
provenance (chapter/section/page range + char offsets into the normalized text)
so every downstream concept, session and validation can point back to exactly
where it came from. Oversized single blocks are split with the existing
``txt_to_voice.utils.chunking.chunk_text`` helper.
"""
from __future__ import annotations

from typing import Optional

from phanoris.shared.utils.chunking import chunk_text  # reuse existing splitter

from .parsers import Block, ParsedDoc


def build_chunks(doc: ParsedDoc, *, max_chars: int = 4000) -> list[dict]:
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

    for block in doc.blocks:
        btext = (block.text or "").strip()
        if not btext:
            continue
        if len(btext) > max_chars:
            flush()
            for piece in chunk_text(btext, max_chars):
                piece = piece.strip()
                if not piece:
                    continue
                sub = Block(
                    text=piece, chapter=block.chapter, section=block.section,
                    page_start=block.page_start, page_end=block.page_end,
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
    }


def _first(values: list[str]) -> Optional[str]:
    return values[0] if values else None
