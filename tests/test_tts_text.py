"""Tests for spoken-form normalization (shared/utils/tts_text.py)."""

from __future__ import annotations

import pytest

from phansora.shared.utils.chunking import chunk_text
from phansora.shared.utils.tts_text import normalize_for_tts


# ── The bug that motivated this: abbreviation periods read as sentence ends ───

def test_abbreviation_period_no_longer_splits_a_sentence():
    """"D.C. drew" is mid-sentence, so the chunkers must not see a full stop there."""
    raw = "The march on Washington, D.C. drew thousands."
    assert "D.C. drew" in raw
    out = normalize_for_tts(raw)
    assert out == "The march on Washington, D C drew thousands."
    # One sentence in, one sentence out — no interior full stop for a chunker to split on.
    assert out.count(".") == 1


def test_real_sentence_boundary_is_preserved():
    """Dropping every abbreviation period would run two sentences together."""
    out = normalize_for_tts("He moved to D.C. The next year he left.")
    assert out == "He moved to D C. The next year he left."


def test_ambiguous_case_keeps_its_period():
    """"the D.C. Metro" is unknowable; keeping the period matches today's behavior."""
    out = normalize_for_tts("The D.C. Metro opened in 1976.")
    assert out == "The D C. Metro opened in 1976."


# ── Lexicons ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Dr. King spoke.", "Doctor King spoke."),
        ("Mr. and Mrs. Smith", "Mister and Missus Smith"),
        ("Sen. Adams and Gov. Reed", "Senator Adams and Governor Reed"),
        ("Martin Luther King Jr. led", "Martin Luther King Junior led"),
    ],
)
def test_titles(raw, expected):
    assert normalize_for_tts(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("on Aug. 28", "on August 28"),
        ("by Sept. 1", "by September 1"),
        ("since Jan. 3", "since January 3"),
    ],
)
def test_months(raw, expected):
    assert normalize_for_tts(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("at 3 p.m. sharp", "at 3 PM sharp"),
        ("from 9 a.m. daily", "from 9 AM daily"),
        ("apples, e.g. Fuji", "apples, for example Fuji"),
        ("that is, i.e. this", "that is, that is this"),
        ("Ali vs. Frazier", "Ali versus Frazier"),
    ],
)
def test_phrases(raw, expected):
    assert normalize_for_tts(raw) == expected


def test_etc_can_end_a_sentence():
    assert normalize_for_tts("pears, etc. Then we left.") == "pears, et cetera. Then we left."


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("the U.S. delegation", "the United States delegation"),
        ("the U.K. and E.U.", "the United Kingdom and European Union."),
        ("the F.B.I. report", "the F B I report"),
    ],
)
def test_initialisms(raw, expected):
    assert normalize_for_tts(raw) == expected


def test_unknown_initialism_is_spelled_not_collapsed():
    """"US" invites the engine to say the word "us"; spaced letters cannot be."""
    assert normalize_for_tts("the A.B. test") == "the A B test"


def test_middle_initial():
    assert normalize_for_tts("John F. Kennedy spoke") == "John F Kennedy spoke"


def test_saint_versus_street():
    assert normalize_for_tts("near St. Louis") == "near Saint Louis"
    assert normalize_for_tts("on Main St. today") == "on Main Street today"


def test_number_only_before_a_digit():
    assert normalize_for_tts("No. 3 finished") == "number 3 finished"
    # A bare "No." is the word "no" and must be left alone.
    assert "number" not in normalize_for_tts("She said No. Firmly.")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("rose 5%", "rose 5 percent"),
        ("hit 20°", "hit 20 degrees"),
        ("R&B charts", "R and B charts"),
    ],
)
def test_symbols(raw, expected):
    assert normalize_for_tts(raw) == expected


def test_bare_percent_is_left_alone():
    """Only a digit-anchored % is unambiguous."""
    assert "percent" not in normalize_for_tts("the % sign")


def test_dollars_are_left_alone():
    """Documented non-goal: a half-fix reads worse than no fix."""
    assert "$1.5 million" in normalize_for_tts("It cost $1.5 million.")


# ── Structural guarantees ────────────────────────────────────────────────────

def test_newlines_survive():
    """Both chunkers split on blank lines and single newlines to handle verse."""
    raw = "Verse one\nverse two\n\nSecond stanza after Dr. Smith."
    out = normalize_for_tts(raw)
    assert out == "Verse one\nverse two\n\nSecond stanza after Doctor Smith."


def test_runs_of_spaces_collapse():
    assert normalize_for_tts("too   many    spaces") == "too many spaces"


@pytest.mark.parametrize(
    "raw",
    [
        "The march on Washington, D.C. drew thousands. Dr. King spoke at 3 p.m. on Aug. 28.",
        "He moved to D.C. The next year he left.",
        "The D.C. Metro opened in 1976.",
        "John F. Kennedy met the U.S. delegation, e.g. the U.N. envoy, etc. It went well.",
        "Rates rose 5% to 20°, No. 3 on Main St. near St. Louis.",
    ],
)
def test_idempotent(raw):
    """Applied at both the document and engine hooks, so it runs more than once."""
    once = normalize_for_tts(raw)
    assert normalize_for_tts(once) == once


@pytest.mark.parametrize("raw", ["", "   ", "\n\n"])
def test_empty_input_is_passed_through(raw):
    assert normalize_for_tts(raw) == raw


# ── End to end: the chunker stops splitting mid-sentence ─────────────────────

def test_chunker_no_longer_breaks_at_an_abbreviation():
    """Chunks are synthesized separately and concatenated, so a mid-sentence split is
    an audible seam. Force a small max_chars to put the boundary in play."""
    raw = "The march on Washington, D.C. drew thousands of people from every state."
    before = chunk_text(raw, 40)
    after = chunk_text(normalize_for_tts(raw), 40)
    # Before: the split lands right after the abbreviation.
    assert any(c.endswith("D.C.") for c in before)
    # After: nothing ends on the abbreviation any more.
    assert not any(c.rstrip().endswith("D C") for c in after)
