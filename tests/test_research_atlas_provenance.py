"""Verbatim grounding for Dossier Nova, without an LLM.

Dossier Nova's promise is that the dossier contains only the sources' own words. That
promise lives in provenance.py, and these tests exist because the promise is only worth
what its REFUSALS are worth: matching a quote correctly is easy, and declining to match a
paraphrase is the part that keeps model wording off the page.

So the cases below are weighted toward what must be rejected — a reworded sentence, a
softened one, a single changed word, an editorial addition — since each of those is a
different claim wearing the source's clothes.
"""
import pytest

from phansora.products.research_atlas.provenance import (
    ground_passage,
    is_grounded,
    normalize_with_map,
)

# Curly quotes, an em dash and hard line wrapping — what real PDF extraction produces, and
# what a model reliably fails to echo back byte for byte.
SOURCE = """=== SOURCE: report.pdf ===

The committee met on 4 July 2024. According to the preliminary findings, the
system failed under load. Investigators said the cause was “not yet
established” — a phrase repeated throughout.

A second paragraph discusses funding. The budget rose by 12% in the same
period, though the report notes this figure excludes contractor costs.
"""

FULL_SENTENCE = "According to the preliminary findings, the\nsystem failed under load."


# --- what must be accepted -------------------------------------------------------

def test_exact_quote_is_restored_to_its_whole_sentence():
    # The model quoted only the tail. Snapping puts the attribution clause back, because
    # "according to the preliminary findings" is what makes the claim a report of a
    # finding rather than a statement of fact.
    assert ground_passage("The system failed under load.", SOURCE) == FULL_SENTENCE


def test_mid_sentence_selection_widens_to_the_sentence():
    assert ground_passage("failed under load", SOURCE) == FULL_SENTENCE


def test_presentation_drift_is_forgiven():
    # Straight quotes for curly, collapsed line wrap, different case.
    got = ground_passage('Investigators said the cause was "not yet established"', SOURCE)
    assert got is not None
    # ...and what comes back carries the SOURCE's characters, not the model's rendering.
    assert "“" in got and "”" in got


def test_qualifier_after_the_claim_is_kept():
    got = ground_passage("The budget rose by 12%", SOURCE)
    assert got is not None and "excludes contractor costs" in got


def test_anchor_path_recovers_a_span_that_drifted_in_the_middle():
    src = ("The inquiry concluded that oversight had been inadequate for a period of "
           "several years, and that responsibility rested with the board.")
    # Model dropped "for a period of" but got both ends exactly right.
    got = ground_passage(
        "The inquiry concluded that oversight had been inadequate several years, "
        "and that responsibility rested with the board.", src)
    assert got is not None and "for a period of" in got


# --- what must be refused: the actual guarantee ----------------------------------

@pytest.mark.parametrize("passage, why", [
    ("The system collapsed when placed under heavy load.", "paraphrase"),
    ("The committee met on 5 August 2024.", "invented fact"),
    ("The failure was clearly foreseeable and negligent.", "editorial addition"),
    ("Investigators said the cause was undetermined", "softened wording"),
    ("The budget fell by 12% in the same period", "one word changed, meaning inverted"),
    ("", "empty"),
])
def test_ungrounded_text_is_refused(passage, why):
    assert ground_passage(passage, SOURCE) is None, f"{why} must not be grounded"


def test_short_near_miss_is_refused_rather_than_anchored():
    src = ("The inquiry concluded that oversight had been inadequate for a period of "
           "several years, and that responsibility rested with the board.")
    # Too few words to anchor safely — a head/tail pair could land somewhere it does not
    # belong, so a short passage has to match outright or not at all.
    assert ground_passage("oversight was bad", src) is None


def test_snapping_never_crosses_into_another_document():
    two = """=== SOURCE: a.pdf ===

Alpha reported the reactor was stable.

=== SOURCE: b.pdf ===

Beta reported the reactor was failing.
"""
    got = ground_passage("the reactor was stable", two)
    assert got is not None
    assert "Beta" not in got and "b.pdf" not in got


# --- the index map, which is what lets the original characters come back ---------

def test_index_map_is_one_to_one_and_in_range():
    norm, index_map = normalize_with_map(SOURCE)
    assert len(norm) == len(index_map)
    assert all(0 <= i < len(SOURCE) for i in index_map)


def test_is_grounded_agrees_with_ground_passage():
    assert is_grounded("The system failed under load.", SOURCE)
    assert not is_grounded("The system was sabotaged.", SOURCE)
