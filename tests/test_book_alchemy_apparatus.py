"""Tests for the front/back-matter filter (book_alchemy/parsers.py + chunking.py).

A Bible course opened by reciting how many chapters each book has, because every
line in the uploaded file was treated as material to teach. These tests pin the
two halves of the fix: reference lines are recognized as navigation, and the
recognition is refused wholesale when it claims implausibly much of a source.

The second half matters more than the first. A missed table of contents is a
paragraph of tedious audio; a false positive at scale silently deletes a book the
reader paid to convert, with nothing in the output to say so.
"""

from __future__ import annotations

from phansora.products.book_alchemy.chunking import APPARATUS_MAX_SHARE, build_chunks
from phansora.products.book_alchemy.parsers import (
    Block,
    ParsedDoc,
    _is_reference_line,
    _running_heads,
    classify_block,
)


# ── Single lines: what counts as navigation ──────────────────────────────────

def test_contents_lines_are_reference_lines():
    assert _is_reference_line("Genesis .......... 1")
    assert _is_reference_line("The Book of Job          842")
    assert _is_reference_line("12")
    assert _is_reference_line("Page 340")
    assert _is_reference_line("xiv")
    assert _is_reference_line("Abraham, 14, 22, 31")
    assert _is_reference_line("covenant, 88-90, 145, 200")


def test_prose_is_not_a_reference_line():
    assert not _is_reference_line(
        "In the beginning God created the heaven and the earth."
    )
    # Ends in a number, but it is a sentence and it is long.
    assert not _is_reference_line(
        "The kingdom was divided after the death of Solomon in 931."
    )
    assert not _is_reference_line("And God said, Let there be light: and there was light.")


def test_a_sentence_ending_in_a_year_survives():
    """The single most likely false positive in a history book."""
    assert not _is_reference_line("The temple was destroyed in 586.")


# ── Blocks: a table of contents versus a chapter ─────────────────────────────

TOC = "\n".join([
    "Genesis .......... 1",
    "Exodus ........... 62",
    "Leviticus ........ 110",
    "Numbers .......... 145",
    "Deuteronomy ...... 199",
])

PROSE = (
    "In the beginning God created the heaven and the earth.\n"
    "And the earth was without form, and void; and darkness was upon the face "
    "of the deep.\n"
    "And the Spirit of God moved upon the face of the waters."
)


def test_a_contents_block_is_apparatus():
    assert classify_block(TOC) == "apparatus"


def test_a_prose_block_is_not():
    assert classify_block(PROSE) == "prose"


def test_a_copyright_page_is_apparatus():
    assert classify_block(
        "Copyright 2019 by the publisher. All rights reserved. No part of this "
        "book may be reproduced in any form without written permission."
    ) == "apparatus"


def test_two_reference_lines_alone_are_not_enough():
    """Below APPARATUS_MIN_LINES a couple of matches prove nothing — a heading and
    a verse number would otherwise take a real passage with them."""
    assert classify_block("Chapter 12\nverse 4") == "prose"


def test_a_mostly_prose_block_survives_a_stray_reference_line():
    assert classify_block(PROSE + "\nGenesis 1") == "prose"


# ── Running heads ────────────────────────────────────────────────────────────

def test_repeated_page_heads_are_found_despite_varying_page_numbers():
    pages = [f"THE BOOK OF GENESIS {n}\nSome real content on this page." for n in range(1, 11)]
    heads = _running_heads(pages)
    assert "THE BOOK OF GENESIS #" in heads


def test_a_short_document_has_no_running_heads():
    """Under RUNNING_HEAD_MIN_PAGES there is no evidence of repetition."""
    assert _running_heads(["A heading\nbody"] * 3) == set()


def test_a_line_appearing_on_a_few_pages_is_not_a_running_head():
    pages = ["Chapter opening\nbody text here"] * 2 + ["different\nbody text here"] * 18
    assert "Chapter opening" not in _running_heads(pages)


# ── The guard: a misfire must not gut the book ───────────────────────────────

def _doc(blocks: list[Block]) -> ParsedDoc:
    return ParsedDoc(title="t", blocks=blocks)


def test_apparatus_chunks_are_marked_not_deleted():
    """Nothing is removed from a source the reader paid to convert; the pipeline
    reads the mark and skips it. Keeping the text is what makes the decision
    auditable after the fact."""
    chunks = build_chunks(_doc([
        Block(text=TOC, kind="apparatus"),
        Block(text=PROSE * 20, kind="prose"),
    ]))
    assert any(not c["teachable"] for c in chunks)
    assert any(c["teachable"] for c in chunks)
    assert any(TOC in c["text"] for c in chunks)


def test_prose_and_apparatus_never_share_a_chunk():
    """The mark is per-chunk, so a blended chunk would be a lie either way."""
    chunks = build_chunks(_doc([
        Block(text="Genesis .......... 1", kind="apparatus"),
        Block(text="Exodus ........... 62", kind="apparatus"),
        Block(text=PROSE, kind="prose"),
        Block(text="Index entries, 4, 9", kind="apparatus"),
    ]))
    for chunk in chunks:
        assert (PROSE in chunk["text"]) != (not chunk["teachable"])


def test_an_implausible_apparatus_share_is_refused_wholesale():
    """More than APPARATUS_MAX_SHARE means the heuristic misfired on this
    document's layout. Narrating some contents pages beats dropping half a book.
    """
    chunks = build_chunks(_doc([
        Block(text=TOC * 10, kind="apparatus"),
        Block(text=PROSE, kind="prose"),
    ]))
    assert all(c["teachable"] for c in chunks)


def test_a_plausible_apparatus_share_is_honored():
    """Same shape as the test above, under the ceiling instead of over it."""
    blocks = [Block(text=TOC, kind="apparatus")]
    blocks += [Block(text=PROSE, kind="prose") for _ in range(30)]
    flagged = sum(len(b.text) for b in blocks if b.kind == "apparatus")
    assert flagged / sum(len(b.text) for b in blocks) < APPARATUS_MAX_SHARE

    chunks = build_chunks(_doc(blocks))
    assert any(not c["teachable"] for c in chunks)


def test_an_all_apparatus_source_is_kept():
    """A document that reads as nothing but reference lines is either a genuine
    reference work or a total misfire. Either way, dropping everything would
    leave the reader a course with no content and no explanation."""
    chunks = build_chunks(_doc([Block(text=TOC, kind="apparatus")]))
    assert chunks
    assert all(c["teachable"] for c in chunks)
