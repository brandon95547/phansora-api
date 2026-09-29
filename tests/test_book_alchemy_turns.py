"""Several people's books on one worker.

The failure that prompted this: on 2026-09-27 a new user's first book failed while
recording its first lesson. The investigation turned up how the worker treats
several books at once. It held ONE book for its whole delivery phase (~2.5 h),
ordered by book age, so a new upload waited behind every phase of every older book
with nothing to listen to. One failed recording in a first phase failed the whole
book with no retry, and the error it left was blank.

Pinned here: a lane hands a book back after a turn, a recording is retried before
the book is given up on, a lesson is recorded as soon as it is written, and no
failure is ever recorded without a name.
"""

from __future__ import annotations

import asyncio

import pytest

from phansora.products.book_alchemy import pipeline, worker
from phansora.products.book_alchemy.audio import VoiceServiceError


# ── The worker's turn ─────────────────────────────────────────────────────────

class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Book:
    """The one project a turn works on: its phase after each step."""

    def __init__(self, phases):
        self.phases = list(phases)
        self.step = 0

    @property
    def phase(self):
        return self.phases[min(self.step, len(self.phases) - 1)]


@pytest.fixture
def turn(monkeypatch):
    """_take_turn against a fake database and a fake clock; returns a runner."""
    clock = Clock()
    monkeypatch.setattr(worker.time, "monotonic", clock)
    monkeypatch.setattr(worker, "TURN_SECONDS", 300.0)
    monkeypatch.setattr(worker, "_stop", asyncio.Event())
    failed = []

    async def noop(*_a, **_k):
        return None

    async def fail(project_id, proj, message):
        failed.append(message)

    monkeypatch.setattr(worker.db, "renew_lease", noop)
    monkeypatch.setattr(worker.db, "set_project", noop)
    monkeypatch.setattr(worker, "_cleanup_source", lambda proj: None)
    monkeypatch.setattr(worker, "_fail", fail)

    async def run(book: Book, step_seconds: float, error: Exception | None = None):
        async def get_project(pid):
            return {"id": pid, "phase": book.phase, "source_path": None}

        async def run_step(proj, client):
            if error is not None:
                raise error
            clock.now += step_seconds
            book.step += 1

        monkeypatch.setattr(worker.db, "get_project", get_project)
        monkeypatch.setattr(worker.pipeline, "run_step", run_step)
        return await worker._take_turn(7, client=None, owner="host:1/1")

    run.failed = failed
    return run


def test_a_turn_ends_once_its_time_is_used(turn):
    """Short steps (concept extraction) run until the turn's time is used, then the
    book goes back — it does not keep the lane until the phase is done."""
    book = Book(["analyze"] * 50)
    steps = asyncio.run(turn(book, step_seconds=120))
    assert steps == 3            # 0 -> 120 -> 240 -> 360: the third step crosses 300
    assert book.step == 3


def test_a_step_longer_than_the_turn_still_completes(turn):
    """A recording is never cut short: a lesson that takes longer than a turn is
    one whole step, and then the book goes back."""
    steps = asyncio.run(turn(Book(["audio"] * 10), step_seconds=600))
    assert steps == 1


def test_a_turn_ends_when_the_book_parks(turn):
    steps = asyncio.run(turn(Book(["audio", "audio", "awaiting_user"]), step_seconds=10))
    assert steps == 2


def test_a_crash_without_a_message_is_still_named(turn):
    """str(RuntimeError()) is "" — which is exactly what project 61 was left with."""
    asyncio.run(turn(Book(["audio"]), step_seconds=10, error=RuntimeError()))
    assert turn.failed == ["RuntimeError"]


def test_a_lane_hands_the_book_back_even_when_the_turn_blows_up(monkeypatch):
    stop = asyncio.Event()
    monkeypatch.setattr(worker, "_stop", stop)
    monkeypatch.setattr(worker, "IDLE_SLEEP", 0)
    released = []
    claims = iter([{"id": 5, "phase": "audio"}])

    async def claim(owner, lease):
        row = next(claims, None)
        if row is None:
            stop.set()
        return row

    async def release(pid):
        released.append(pid)

    async def boom(pid, client, owner):
        raise RuntimeError("database went away")

    monkeypatch.setattr(worker.db, "claim_next_project", claim)
    monkeypatch.setattr(worker.db, "release_lease", release)
    monkeypatch.setattr(worker, "_take_turn", boom)

    asyncio.run(asyncio.wait_for(worker._lane(1, client=None), timeout=2))
    assert released == [5]


def test_the_shipped_defaults_share_the_worker():
    """More than one lane, and a turn measured in minutes — not a whole phase."""
    assert worker.CONCURRENCY > 1
    assert 60 <= worker.TURN_SECONDS <= 900


# ── Record a lesson as soon as it is written ─────────────────────────────────

def row(ordinal):
    return {"ordinal": ordinal}


def test_the_lesson_just_written_is_recorded_before_the_next_is_written():
    assert pipeline._record_before_writing(row(1), row(2)) is True


def test_nothing_written_means_write():
    assert pipeline._record_before_writing(None, row(1)) is False


def test_the_last_written_lessons_are_recorded():
    assert pipeline._record_before_writing(row(6), None) is True


def test_a_reopened_earlier_lesson_is_written_before_later_ones_are_recorded():
    """A Regenerate on lesson 3 while 4-6 wait to be recorded: 3 is what the
    listener hears next, so it is rewritten first."""
    assert pipeline._record_before_writing(row(4), row(3)) is False


# ── A failed recording is retried before the book is ─────────────────────────

@pytest.fixture
def lesson(monkeypatch, tmp_path):
    """_record_lesson with the database and the voice service faked."""
    saved = []
    calls = []

    async def noop(*_a, **_k):
        return None

    async def set_session(session_id, **fields):
        saved.append(fields)

    async def stage_label(pid, detail):
        return detail

    monkeypatch.setattr(pipeline.db, "set_project", noop)
    monkeypatch.setattr(pipeline.db, "set_session", set_session)
    monkeypatch.setattr(pipeline, "_stage_label", stage_label)
    monkeypatch.setattr(pipeline, "_LESSON_RETRY_WAIT_S", 0)
    monkeypatch.setattr(pipeline, "_LESSON_ATTEMPTS", 3)
    monkeypatch.setattr(
        pipeline, "session_audio_path", lambda uid, pid, ordinal, ext: tmp_path / f"{ordinal}.{ext}"
    )

    async def run(outcomes):
        """Each outcome is an exception to raise or a duration to return."""
        script = iter(outcomes)

        async def render(**kwargs):
            calls.append(kwargs)
            outcome = next(script)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        monkeypatch.setattr(pipeline, "render_script_to_audio", render)
        project = {"id": 61, "user_id": 14, "options": {}}
        sess = {"id": 2602, "ordinal": 1, "script": "Gastric cancer, in brief."}
        await pipeline._record_lesson(project, sess, 6, [])

    run.saved = saved
    run.calls = calls
    return run


def test_a_failed_render_is_retried(lesson):
    asyncio.run(lesson([VoiceServiceError(500, "Generating that audio failed."), 1234]))
    assert len(lesson.calls) == 2
    assert lesson.saved[-1]["status"] == "complete"
    assert lesson.saved[-1]["audio_seconds"] == 1234


def test_a_render_that_always_fails_still_fails_the_lesson(lesson):
    with pytest.raises(RuntimeError):
        asyncio.run(lesson([RuntimeError(), RuntimeError(), RuntimeError()]))
    assert len(lesson.calls) == 3
    assert lesson.saved == []


def test_a_bad_request_is_not_retried(lesson):
    with pytest.raises(VoiceServiceError):
        asyncio.run(lesson([VoiceServiceError(400, "Unknown voice.")]))
    assert len(lesson.calls) == 1
