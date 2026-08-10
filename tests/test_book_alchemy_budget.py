"""Tests for lesson length and lesson count (book_alchemy/pipeline.py).

These pin the change that turned Book Alchemy from a re-voicing into a course.
The old rule was `min_words = source_words * 0.9`, which made compression
structurally impossible: 780,000 words of Bible could not become fewer than about
700,000 narrated ones, so the course ran ~108 hours and spent most of it reciting
genealogies name by name.

The new rule budgets from how many distinct ideas the analyze phase indexed. The
property that matters is not any particular ratio — those are tuning numbers —
but that the ratio RESPONDS to information density: repetitive source compresses
hard, dense source barely compresses at all.
"""

from __future__ import annotations

import pytest

from phansora.products.book_alchemy.pipeline import (
    BUDGET_UNDERSHOOT,
    DEPTHS,
    MAX_LESSON_WORDS,
    Depth,
    _lesson_budget,
    lesson_word_budget,
    max_source_words_per_lesson,
    resolve_depth,
)

STANDARD = DEPTHS["standard"]
COMPREHENSIVE = DEPTHS["comprehensive"]
OVERVIEW = DEPTHS["overview"]


def ratio(concept_count: int, source_words: int, depth: Depth) -> float:
    """Midpoint of the budget as a share of the source."""
    lo, hi = lesson_word_budget(
        concept_count=concept_count, source_words=source_words, depth=depth
    )
    return ((lo + hi) / 2) / source_words


# ── The property the whole change rests on ───────────────────────────────────

def test_repetitive_source_compresses_far_harder_than_dense_source():
    """Two lessons, same source length, different numbers of ideas in them.

    A genealogy indexes as one or two concepts however many names it lists; a
    page of argument indexes as a dozen. The first should compress hard and the
    second barely — and no fixed density band can express that difference.
    """
    genealogy = ratio(2, 2100, STANDARD)
    argument = ratio(14, 2100, STANDARD)
    assert genealogy < argument
    assert argument / genealogy > 2


def test_a_dense_lesson_is_not_compressed_into_a_summary():
    """You cannot teach fourteen distinct ideas in two hundred words. The budget
    has to grow with the idea count even at a compressing depth."""
    lo, _ = lesson_word_budget(concept_count=14, source_words=2100, depth=STANDARD)
    assert lo >= 14 * 50


def test_a_sparse_lesson_is_allowed_to_be_short():
    """The old floor (0.9 x source) is exactly what this must no longer do."""
    lo, hi = lesson_word_budget(concept_count=1, source_words=3000, depth=STANDARD)
    assert hi < 3000 * 0.9


# ── The clamps: neither failure mode may run away ────────────────────────────

def test_the_ceiling_stops_a_course_being_invented_around_the_material():
    """The original failure this product had: a 334-word letter blown into ~1,290
    words across four lessons. However many concepts get indexed, the lesson
    cannot outgrow its own source by more than the depth allows."""
    _, hi = lesson_word_budget(concept_count=99, source_words=334, depth=STANDARD)
    assert hi <= 334 * STANDARD.ceiling_share * 1.25 + 60


def test_the_floor_stops_a_thin_index_producing_a_stub():
    """If the analyze phase came back with nothing, the lesson still has to teach
    the segment rather than emit two sentences about it."""
    lo, _ = lesson_word_budget(concept_count=0, source_words=3000, depth=STANDARD)
    assert lo >= int(int(3000 * STANDARD.floor_share) * BUDGET_UNDERSHOOT)


def test_min_is_always_below_max():
    for depth in DEPTHS.values():
        for concepts in (0, 1, 5, 50):
            for words in (60, 500, 3000, 20000):
                lo, hi = lesson_word_budget(
                    concept_count=concepts, source_words=words, depth=depth
                )
                assert 0 < lo < hi


def test_a_tiny_source_still_gets_a_workable_budget():
    lo, hi = lesson_word_budget(concept_count=1, source_words=40, depth=STANDARD)
    assert lo >= 60 and hi > lo


# ── Depth ordering ───────────────────────────────────────────────────────────

def test_depths_are_ordered_by_how_much_they_compress():
    dense = 12
    assert (
        ratio(dense, 2100, OVERVIEW)
        < ratio(dense, 2100, STANDARD)
        < ratio(dense, 2100, COMPREHENSIVE)
    )


def test_comprehensive_still_runs_at_about_parity_or_above():
    """It is the escape hatch for a reader who wants the whole text taught back,
    so it must not quietly compress."""
    assert ratio(12, 2100, COMPREHENSIVE) >= 0.85


def test_resolve_depth_falls_back_rather_than_failing():
    assert resolve_depth({"depth": "overview"}) is OVERVIEW
    assert resolve_depth({"depth": "OverView"}) is OVERVIEW
    # Projects created before the option existed, and clients sending nonsense.
    assert resolve_depth({}) is DEPTHS["standard"]
    assert resolve_depth(None) is DEPTHS["standard"]
    assert resolve_depth({"depth": "nonsense"}) is DEPTHS["standard"]
    assert resolve_depth({"voice": "default"}) is DEPTHS["standard"]


# ── Lesson count, which must be computed in narration words ─────────────────

def test_lesson_count_falls_as_compression_rises():
    """The bug this guards: _lesson_budget used to divide SOURCE words, so a
    compressing depth produced the same number of much shorter lessons instead of
    fewer full ones."""
    _, comprehensive, _ = _lesson_budget(780_000, COMPREHENSIVE)
    _, standard, _ = _lesson_budget(780_000, STANDARD)
    _, overview, _ = _lesson_budget(780_000, OVERVIEW)
    assert overview < standard < comprehensive


def test_a_long_book_becomes_a_listenable_number_of_lessons():
    """A Bible-sized source at the default depth. The old code gave 372 lessons
    and ~108 hours of audio."""
    _, suggested, _ = _lesson_budget(780_000, STANDARD)
    hours = suggested * 14 / 60          # TARGET_LESSON_MINUTES per lesson
    assert 20 <= hours <= 50


def test_a_source_that_fits_one_sitting_is_one_lesson():
    """Decided in code, never by the model: a short work that touches five topics
    is still one lesson."""
    assert _lesson_budget(500, STANDARD) == (1, 1, 1)
    assert _lesson_budget(0, STANDARD) == (1, 1, 1)


def test_a_long_book_is_never_one_lesson():
    minimum, _, _ = _lesson_budget(780_000, COMPREHENSIVE)
    assert minimum > 1


def test_lesson_counts_are_ordered_and_positive():
    for depth in DEPTHS.values():
        for words in (0, 100, 5_000, 100_000, 780_000):
            lo, mid, hi = _lesson_budget(words, depth)
            assert 1 <= lo <= mid <= hi


# ── The source-word cap the segmentation prompt is given ────────────────────

def test_the_segmentation_cap_is_expressed_in_source_words():
    """Segmentation counts source segments, so handing it the narration cap would
    cut every lesson into fragments at a compressing depth."""
    assert max_source_words_per_lesson(STANDARD) > MAX_LESSON_WORDS
    assert max_source_words_per_lesson(COMPREHENSIVE) < max_source_words_per_lesson(STANDARD)


def test_the_segmentation_cap_matches_the_planning_ratio():
    assert max_source_words_per_lesson(STANDARD) == pytest.approx(
        MAX_LESSON_WORDS / STANDARD.planning_ratio, rel=0.01
    )
