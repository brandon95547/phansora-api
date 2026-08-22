"""Group a course's lessons into user-gated delivery phases.

A twenty-hour course used to arrive in one run. It now arrives in batches the
listener asks for, and this module decides where those batches fall.

The listener's unit is a sitting, so the cap is a DURATION rather than a lesson
count; the book's own structure decides where inside that budget the cut lands,
when the source has any structure to offer. Most sources do not — pdf, pasted
text and OCR carry no chapter at all (see parsers.py: _parse_pdf records page
numbers and nothing else), and pdf is the likeliest format for a long book. So
the cap alone has to produce a sane result, and the code below is written so that
is the SAME code path rather than a fallback branch: with no chapters, no cut
point is preferred over any other and the cap simply decides by itself.

Nothing here touches the model, the database or the source text. Phases are a
pure function of the lesson plan, which is why planning them costs nothing and
happens exactly once, at the end of the curriculum step.

NAMING TRAP: ``book_alchemy_projects.phase`` is the internal pipeline cursor
(uploaded -> analyze -> ... -> complete). The phases here are DELIVERY phases —
what the listener means by "Phase 3 of 7". Different things.
"""
from __future__ import annotations

import os
from typing import Optional

# Import direction is one-way ON PURPOSE: this module reads pipeline's listening
# constants, so pipeline must NOT import this one at module level or the two
# deadlock on each other at import time. pipeline._phase_curriculum imports it
# inside the function instead. Do not "tidy" that into a top-level import.
from .pipeline import DEFAULT_DEPTH, DEPTHS, WORDS_PER_MINUTE

# The cap the product promises: roughly one long sitting's worth of listening.
# Env-overridable purely so an end-to-end test can force several phases out of a
# short source without a twenty-hour run.
PHASE_TARGET_SECONDS = int(os.getenv("BOOK_ALCHEMY_PHASE_TARGET_SECONDS", str(3 * 3600)))

# How full a phase must already be before it may cut in the middle of a chapter.
# Only reachable when one chapter alone runs past the cap.
PHASE_SPLIT_FLOOR = 0.75

# A trailing runt is not worth a click of its own; fold it into its predecessor.
PHASE_MERGE_TAIL = 0.30

# A lesson does not run for as long as its source would take to read: it teaches
# the ideas the source contains and drops the repetition and enumeration, so the
# narration is a fraction of the source at every depth but "comprehensive". That
# fraction is the depth's planning_ratio, and it is a PARAMETER here rather than
# a constant because two projects at different depths get genuinely different
# durations from the same source.
#
# Do NOT estimate from TARGET_LESSON_MINUTES instead. That constant is derived
# from narration pace, but the caller passes SOURCE words, so the two only agree
# when the ratio happens to be 1.0.
DEFAULT_NARRATION_RATIO = (
    DEPTHS.get(DEFAULT_DEPTH) or DEPTHS["standard"]
).planning_ratio


def estimate_seconds(
    *,
    source_words: int = 0,
    script: str = "",
    audio_seconds: Optional[int] = None,
    narration_ratio: Optional[float] = None,
) -> int:
    """Best available duration for one lesson: measured beats scripted beats planned.

    At planning time only the last branch can fire — no script exists yet, so the
    depth's ``planning_ratio`` is the only thing standing between a source word
    count and a duration. The earlier two are here because the same estimate is
    re-derived for display as lessons land, and a real number should always win
    over a guess; neither of them needs the ratio at all.
    """
    if audio_seconds:
        return int(audio_seconds)
    if script and script.strip():
        return int(len(script.split()) * 60.0 / WORDS_PER_MINUTE)
    ratio = DEFAULT_NARRATION_RATIO if narration_ratio is None else max(0.0, narration_ratio)
    return int(max(0, source_words) * ratio * 60.0 / WORDS_PER_MINUTE)


def plan_phases(lessons: list[dict], *, cap_seconds: int = PHASE_TARGET_SECONDS) -> list[dict]:
    """Contiguous groups of lessons, each roughly at or under ``cap_seconds``.

    ``lessons`` is in ordinal order; each carries ``{ordinal, seconds, chapter}``.
    ``chapter`` may be None throughout, which is the intended behavior for pdf
    and plain text rather than a degraded one.

    A phase only ever closes at a lesson boundary, and preferentially at one where
    the chapter changes. Because a single lesson is capped at MAX_LESSON_WORDS of
    narration (~20 minutes of audio), a phase can overshoot the cap by at most one
    lesson. That is the honest contract.
    """
    groups: list[list[dict]] = []
    cur: list[dict] = []
    cur_sec = 0

    for lesson in lessons:
        sec = max(0, int(lesson.get("seconds") or 0))
        if cur:
            new_chapter = _key(lesson) != _key(cur[-1])
            would_overflow = cur_sec + sec > cap_seconds
            # Two reasons to close a phase here: the book itself turns a page and
            # this lesson would not have fitted anyway, or a single chapter has
            # outgrown the cap and the phase is full enough that cutting inside it
            # beats running on indefinitely.
            if would_overflow and (new_chapter or cur_sec >= cap_seconds * PHASE_SPLIT_FLOOR):
                groups.append(cur)
                cur, cur_sec = [], 0
        cur.append(lesson)
        cur_sec += sec
    if cur:
        groups.append(cur)

    # A ten-minute final phase is a click for its own sake. Fold it back, unless
    # doing so would make its neighbor absurd.
    if len(groups) > 1:
        tail = _secs(groups[-1])
        prev = _secs(groups[-2])
        if tail < cap_seconds * PHASE_MERGE_TAIL and prev + tail <= cap_seconds * 1.5:
            groups[-2].extend(groups.pop())

    return [
        {
            "ordinal": i,
            "label": _label(g),
            "session_start": g[0]["ordinal"],
            "session_end": g[-1]["ordinal"],
            "est_seconds": _secs(g),
        }
        for i, g in enumerate(groups, start=1)
    ]


def _secs(group: list[dict]) -> int:
    return sum(max(0, int(item.get("seconds") or 0)) for item in group)


def _key(lesson: dict) -> Optional[str]:
    chapter = (lesson.get("chapter") or "").strip()
    return chapter.casefold() or None


def _label(group: list[dict]) -> str:
    """What to call this phase in the list.

    The book's own words where it has them, lesson numbers where it does not —
    never an invented course-style name, for the same reason the segmentation
    prompt forbids them. A chapter that spans several phases yields the same
    label on each; the UI disambiguates with "Phase 2 of 7".
    """
    names: list[str] = []
    for lesson in group:
        chapter = (lesson.get("chapter") or "").strip()
        if chapter and chapter not in names:
            names.append(chapter)

    if not names:
        first, last = group[0]["ordinal"], group[-1]["ordinal"]
        return f"Lesson {first}" if first == last else f"Lessons {first}–{last}"
    if len(names) == 1:
        return names[0][:120]
    return f"{names[0]} – {names[-1]}"[:160]
