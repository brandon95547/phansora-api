"""Tests for delivery-phase grouping (products/book_alchemy/phases.py).

A phase is what the listener asks for one at a time, so the two things that must
never break are: every lesson lands in exactly one phase, and a phase is roughly
a sitting's worth of listening. Everything else is a preference.
"""

from __future__ import annotations

from phansora.products.book_alchemy.phases import estimate_seconds, plan_phases

CAP = 3 * 3600


def lessons(start, end, seconds, chapter=None):
    return [
        {"ordinal": i, "seconds": seconds, "chapter": chapter}
        for i in range(start, end + 1)
    ]


def covered(phases):
    out = []
    for p in phases:
        out += list(range(p["session_start"], p["session_end"] + 1))
    return out


def total_of(phases):
    return sum(p["est_seconds"] for p in phases)


# ── The invariant everything else depends on ─────────────────────────────────

def test_every_lesson_lands_in_exactly_one_phase():
    """Phases tile the course: no lesson skipped, none delivered twice."""
    src = lessons(1, 85, 850)
    phases = plan_phases(src, cap_seconds=CAP)
    assert covered(phases) == [l["ordinal"] for l in src]
    assert total_of(phases) == sum(l["seconds"] for l in src)


def test_phase_ranges_are_contiguous_and_ordered():
    phases = plan_phases(lessons(1, 40, 900), cap_seconds=CAP)
    assert [p["ordinal"] for p in phases] == list(range(1, len(phases) + 1))
    for prev, nxt in zip(phases, phases[1:]):
        assert nxt["session_start"] == prev["session_end"] + 1


# ── The case that actually ships: a source with no chapters at all ───────────
# pdf, pasted text and OCR carry no chapter (see parsers._parse_pdf), and pdf is
# the likeliest format for a long book. This is the primary path, not a fallback.

def test_no_chapter_data_still_cuts_near_the_cap():
    phases = plan_phases(lessons(1, 85, 850), cap_seconds=CAP)
    assert len(phases) > 1
    for p in phases:
        assert p["est_seconds"] <= CAP + 900   # at most one lesson of overshoot


def test_no_chapter_data_labels_by_lesson_number():
    phases = plan_phases(lessons(1, 85, 850), cap_seconds=CAP)
    assert phases[0]["label"].startswith("Lessons ")


# ── Chapter awareness, where the format provides it ─────────────────────────

def test_short_chapters_merge_up_to_the_cap():
    """Twenty 25-minute chapters must not become twenty phases."""
    src = [
        {"ordinal": i, "seconds": 1500, "chapter": f"Chapter {i}"}
        for i in range(1, 21)
    ]
    phases = plan_phases(src, cap_seconds=CAP)
    assert len(phases) < 20
    assert all(p["est_seconds"] <= CAP + 1500 for p in phases)
    assert "–" in phases[0]["label"]           # spans several chapters


def test_phase_prefers_to_close_on_a_chapter_boundary():
    """A short opening chapter is its own phase rather than bleeding into part one."""
    src = (
        lessons(1, 4, 2400, "Introduction")
        + lessons(5, 20, 2400, "Part One")
        + lessons(21, 23, 2400, "Conclusion")
    )
    phases = plan_phases(src, cap_seconds=CAP)
    assert phases[0]["label"] == "Introduction"
    assert phases[0]["session_end"] == 4
    assert phases[-1]["label"] == "Conclusion"


def test_one_chapter_larger_than_the_cap_is_split():
    """A seven-hour chapter cannot be one phase; it splits at lesson boundaries."""
    phases = plan_phases(lessons(1, 30, 840, "Part One"), cap_seconds=CAP)
    assert len(phases) > 1
    assert all(p["label"] == "Part One" for p in phases)
    assert covered(phases) == list(range(1, 31))


# ── Shape edges ──────────────────────────────────────────────────────────────

def test_a_single_lesson_is_a_single_phase():
    phases = plan_phases(lessons(1, 1, 900), cap_seconds=CAP)
    assert len(phases) == 1
    assert phases[0]["label"] == "Lesson 1"


def test_a_course_under_the_cap_is_never_split():
    phases = plan_phases(lessons(1, 6, 900), cap_seconds=CAP)
    assert len(phases) == 1


def test_a_runt_tail_folds_into_the_previous_phase():
    """A 45-minute final phase is a click for its own sake."""
    src = lessons(1, 13, 2700)          # 4 x 45min = 3h exactly, then one over
    phases = plan_phases(src, cap_seconds=CAP)
    assert phases[-1]["est_seconds"] > CAP * 0.30
    assert covered(phases) == list(range(1, 14))


def test_empty_course_yields_no_phases():
    assert plan_phases([], cap_seconds=CAP) == []


# ── Duration estimation ──────────────────────────────────────────────────────

def test_estimate_prefers_measured_audio_over_a_script():
    assert estimate_seconds(source_words=2000, script="a " * 500, audio_seconds=999) == 999


def test_estimate_prefers_a_script_over_source_words():
    # 300 narration words at 150 wpm is two minutes, regardless of source length.
    assert estimate_seconds(source_words=99999, script="word " * 300) == 120


def test_estimate_accounts_for_narration_running_longer_than_source():
    """Reading the cap off TARGET_LESSON_MINUTES would undershoot by ~25%; the
    density midpoint is what keeps a 3-hour phase actually about 3 hours."""
    # 2100 source words is the "14 minute" lesson target, but narration expands it.
    assert estimate_seconds(source_words=2100) > 14 * 60


def test_estimate_is_never_negative():
    assert estimate_seconds(source_words=-5) == 0
