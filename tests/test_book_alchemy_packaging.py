"""Tests for the packaging filter: the document as an object, never as material.

A course narrated its source's own front matter — the filename the PDF shipped
under, the GeoCities address it could be downloaded from, its redistribution
terms, the author's two email addresses, and his degrees, employer and published
papers. It then framed the whole course around his credentials: "so you have
someone who is both scientifically trained and deeply interested in the occult,
and that synthesis is what we will be working with."

Every word of that was in the source, so no fidelity check could catch it. Four
guards now stand in the way, and these tests pin the two that are deterministic
plus the boundary that decides where material begins:

  parse time      _mark_front_matter    where the front-matter run ends
  parse time      classify_block        blocks shaped like a notice or address
  index time      prompts.PACKAGING     the rule, shared by all three prompts
  script time     scrub_apparatus       the floor: no address ever reaches TTS

The floor matters most. A session that fails validation on every attempt is
still written out and still narrated, so the model-level guards have no stop of
their own.
"""

from __future__ import annotations

import asyncio

import pytest

from phansora.products.book_alchemy import prompts
from phansora.products.book_alchemy.parsers import _is_reference_line, classify_block
from phansora.products.book_alchemy.pipeline import (
    FRONT_MATTER_MAX_SHARE,
    _mark_front_matter,
)
from phansora.products.book_alchemy.validation import scrub_apparatus


# ── The real front matter, as the parser would hand it over ──────────────────

DISTRIBUTION = (
    "This file may be freely uploaded anywhere, as long as it is complete and "
    "unmodified. You may share it, but do not alter it."
)

CONTACT = "\n".join([
    "The author can be contacted at:",
    "dondeg@compuserve.com",
    "ddegraci@med.wayne.edu",
    "http://www.geocities.com/ddegraci/index.html",
])

MATERIAL = (
    "An out-of-body experience begins where ordinary sleep ends. The mind "
    "remains awake while the body's own signals fall away, and what is left is "
    "an awareness with no obvious location. "
)


# ── classify_block: a notice is not prose ────────────────────────────────────

def test_a_distribution_notice_is_apparatus():
    """No publisher, so no copyright page — the terms are written as sentences.
    That shape is why the original filter walked straight past them."""
    assert classify_block(DISTRIBUTION) == "apparatus"


def test_a_contact_block_is_apparatus():
    assert classify_block(CONTACT) == "apparatus"


def test_prose_about_sharing_ideas_is_not_apparatus():
    """The likeliest false positive: material that happens to discuss sharing."""
    assert classify_block(
        "The teaching was shared freely among the students, and each was expected "
        "to pass it on without alteration to whoever came after."
    ) == "prose"


# ── _is_reference_line: an address line carries no prose ─────────────────────

def test_a_bare_address_is_a_reference_line():
    assert _is_reference_line("dondeg@compuserve.com")
    assert _is_reference_line("dondeg@compuserve.com (preferred)")
    assert _is_reference_line("http://www.geocities.com/ddegraci/index.html")
    assert _is_reference_line("www.geocities.com/ddegraci")


def test_a_sentence_mentioning_an_address_is_still_prose():
    """The guard that keeps this from eating material: a real sentence about an
    address is long, and the packaging rule in the prompts is what handles it."""
    assert not _is_reference_line(
        "He asked every student to write to the address given in the lesson "
        "before the end of the week, without exception."
    )


# ── scrub_apparatus: the floor ───────────────────────────────────────────────

LEAKED = (
    "The file is available in Adobe Acrobat format, under the filename DO_OBE.PDF. "
    "An out-of-body experience begins where ordinary sleep ends. "
    "If you need to reach the author, he can be contacted at dondeg@compuserve.com. "
    "The mind remains awake while the body's own signals fall away. "
    "Information is available at http://www.geocities.com/ddegraci/index.html."
)


def test_the_scrub_removes_addresses_and_filenames():
    clean, removed = scrub_apparatus(LEAKED)
    assert len(removed) == 3
    assert "DO_OBE.PDF" not in clean
    assert "compuserve.com" not in clean
    assert "geocities" not in clean


def test_the_scrub_keeps_the_teaching_around_them():
    """Sentence granularity, not clause surgery — the lesson has to still read."""
    clean, _ = scrub_apparatus(LEAKED)
    assert "An out-of-body experience begins where ordinary sleep ends." in clean
    assert "The mind remains awake while the body's own signals fall away." in clean


def test_a_clean_script_is_returned_untouched():
    """Identity, not a rebuild: the common case must not reflow the narration."""
    script = "First paragraph, teaching.\n\nSecond paragraph, teaching."
    clean, removed = scrub_apparatus(script)
    assert removed == []
    assert clean is script


# ── prompts: writer and checker must be handed the same rule ─────────────────

def test_all_three_prompts_carry_the_same_packaging_rule():
    """The trap this module's docstring already documents for already_taught: a
    writer told to skip something the checker still expects is marked down for an
    omission nobody asked for, and the retry loop puts the packaging back in."""
    for prompt in (prompts.ANALYZE_SYSTEM, prompts.SCRIPT_SYSTEM, prompts.VALIDATION_SYSTEM):
        assert prompts.PACKAGING in prompt


def test_the_checker_is_told_not_to_call_skipped_packaging_an_omission():
    assert "NEVER mark it omitted" in prompts.VALIDATION_SYSTEM


# ── _mark_front_matter: where the material begins ────────────────────────────

class _StubClient:
    """Returns one canned boundary verdict, and records that it was asked once."""

    def __init__(self, verdict):
        self.verdict = verdict
        self.calls = 0

    async def chat_json(self, **kwargs):
        self.calls += 1
        return self.verdict


def _chunks(*texts: str) -> list[dict]:
    out = []
    cursor = 0
    for i, text in enumerate(texts):
        out.append({
            "ordinal": i, "text": text, "chapter": None, "section": None,
            "page_start": None, "page_end": None,
            "char_start": cursor, "char_end": cursor + len(text),
            "teachable": True,
        })
        cursor += len(text) + 2
    return out


def _run(client, chunks):
    return asyncio.run(_mark_front_matter(client, chunks))


def test_the_leading_run_is_marked_not_teachable():
    chunks = _chunks(DISTRIBUTION + "\n" + CONTACT, MATERIAL * 20, MATERIAL * 20)
    out = _run(_StubClient({"first_material_ordinal": 1, "first_material_sentence": None}), chunks)
    assert [c["teachable"] for c in out] == [False, True, True]
    # Marked, never deleted — the reader paid to convert this file.
    assert any("compuserve" in c["text"] for c in out)


def test_a_straddling_chunk_is_split_at_the_first_material_sentence():
    """The case that actually broke the course: front matter and the opening of
    the real material inside one 4000-char chunk, where no whole-chunk verdict
    is right."""
    first = MATERIAL.strip().split(". ")[0] + "."
    chunks = _chunks(CONTACT + "\n\n" + MATERIAL * 12, MATERIAL * 20)
    out = _run(
        _StubClient({"first_material_ordinal": 0, "first_material_sentence": first}),
        chunks,
    )
    assert len(out) == 3
    assert out[0]["teachable"] is False and "compuserve" in out[0]["text"]
    assert out[1]["teachable"] is True and out[1]["text"].startswith(first)
    assert "compuserve" not in out[1]["text"]
    # Every later phase indexes by ordinal, and the split added a chunk.
    assert [c["ordinal"] for c in out] == [0, 1, 2]


def test_a_document_that_opens_on_material_is_left_alone():
    """The common case, and the answer the prompt asks for when unsure."""
    chunks = _chunks(MATERIAL * 20, MATERIAL * 20)
    out = _run(_StubClient({"first_material_ordinal": 0, "first_material_sentence": None}), chunks)
    assert all(c["teachable"] for c in out)


def test_an_implausible_boundary_is_refused_wholesale():
    """Same reasoning as APPARATUS_MAX_SHARE: a classifier claiming most of a
    source has misfired, and narrating a preface beats eating the work."""
    chunks = _chunks(MATERIAL * 20, MATERIAL * 20, MATERIAL * 20)
    out = _run(_StubClient({"first_material_ordinal": 2, "first_material_sentence": None}), chunks)
    assert all(c["teachable"] for c in out)


def test_a_single_chunk_source_is_never_touched():
    """One chunk IS the whole source; a verdict could leave a course with no
    material at all, so the question is not even asked."""
    client = _StubClient({"first_material_ordinal": 1, "first_material_sentence": None})
    out = _run(client, _chunks(MATERIAL))
    assert client.calls == 0
    assert all(c["teachable"] for c in out)


def test_a_failed_boundary_check_keeps_everything():
    """Never fail a paid job over this — the course is only duller without it."""

    class _Broken:
        async def chat_json(self, **kwargs):
            raise RuntimeError("deepseek down")

    chunks = _chunks(CONTACT, MATERIAL * 20)
    out = _run(_Broken(), chunks)
    assert all(c["teachable"] for c in out)


def test_the_share_ceiling_is_looser_than_the_regex_filters():
    """A contiguous run from position zero is a much narrower claim than a
    scattered line-shape verdict, so it is trusted further — but still bounded."""
    assert 0.25 < FRONT_MATTER_MAX_SHARE < 0.5
