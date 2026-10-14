"""Where a book's time went, and the two places it was simply waiting.

A course is as slow as its slowest machine, and that is the voice: one GPU,
one lesson at a time. Nothing here makes the voice faster. What these pin is the
waiting AROUND it, which was most of what a listener sat through before hearing
anything, and a fifth of every phase after that:

  * Analysis indexed one excerpt per model call, end to end — over half an hour
    for an ordinary book before lesson 1 could be planned. The calls are
    independent, so they now run side by side and are stored in source order.
  * Scripts and recordings alternated, so the voice idled while every script was
    written. The next script is now written while the current lesson records.

Neither changes what a course contains, and the tests below hold that too: the
same excerpts are indexed, in the same order, and lessons are still written and
recorded in ordinal order.
"""

from __future__ import annotations

import asyncio

import pytest

from phansora.products.book_alchemy import cover, pipeline


# ── Analysis: side by side, stored in order ──────────────────────────────────

def chunk(ordinal, *, teachable=True):
    return {
        "id": 100 + ordinal, "ordinal": ordinal, "chapter": None,
        "text": f"excerpt-{ordinal}", "teachable": teachable,
    }


class Indexer:
    """A model that takes a moment per excerpt, and notes how many it had at once."""

    def __init__(self, *, fail_first=(), fail_always=(), apparatus=(), slow_first=False):
        self.calls = []
        self.live = 0
        self.peak = 0
        self.fail_first = set(fail_first)
        self.fail_always = set(fail_always)
        self.apparatus = set(apparatus)
        self.slow_first = slow_first

    async def chat_json(self, *, system, user, max_output_tokens):
        ordinal = int(user.rsplit("-", 1)[1])
        first_try = ordinal not in self.calls
        self.calls.append(ordinal)
        self.live += 1
        self.peak = max(self.peak, self.live)
        try:
            # The earliest excerpt answers LAST, so anything stored in arrival
            # order would come out reversed.
            await asyncio.sleep(0.03 - ordinal * 0.002 if self.slow_first else 0.005)
            if ordinal in self.fail_always or (ordinal in self.fail_first and first_try):
                raise RuntimeError("DeepSeek HTTP 429: slow down")
            if ordinal in self.apparatus:
                return {"teachable": False}
            return {"concepts": [{"title": f"idea-{ordinal}", "body": "…"}]}
        finally:
            self.live -= 1


@pytest.fixture
def analyze(monkeypatch):
    """_phase_analyze against a book held in memory; returns a runner."""
    state = {"stored": [], "cleared": [], "apparatus": [], "project": {}}

    async def run(chunks, client, *, cursor=0, concurrency=8):
        async def count_chunks(pid):
            return len(chunks)

        async def get_chunks_in_range(pid, start, stop):
            return [c for c in chunks if start <= c["ordinal"] < stop]

        async def delete_concepts_for_chunks(pid, ids):
            state["cleared"].append(list(ids))

        async def insert_concepts(pid, concepts):
            state["stored"].extend(c["source_chunk_ids"][0] for c in concepts)

        async def set_chunk_teachable(chunk_id, teachable):
            state["apparatus"].append(chunk_id)

        async def set_project(pid, **fields):
            state["project"].update(fields)

        monkeypatch.setattr(pipeline, "ANALYZE_CONCURRENCY", concurrency)
        monkeypatch.setattr(pipeline.prompts, "analyze_user", lambda text, chapter=None: text)
        monkeypatch.setattr(pipeline.db, "count_chunks", count_chunks)
        monkeypatch.setattr(pipeline.db, "get_chunks_in_range", get_chunks_in_range)
        monkeypatch.setattr(pipeline.db, "delete_concepts_for_chunks", delete_concepts_for_chunks)
        monkeypatch.setattr(pipeline.db, "insert_concepts", insert_concepts)
        monkeypatch.setattr(pipeline.db, "set_chunk_teachable", set_chunk_teachable)
        monkeypatch.setattr(pipeline.db, "set_project", set_project)
        await pipeline._phase_analyze({"id": 7, "analyze_cursor": cursor}, client)
        return state

    return run


def test_a_batch_of_excerpts_is_indexed_at_once(analyze):
    client = Indexer()
    state = asyncio.run(analyze([chunk(n) for n in range(20)], client))
    assert client.peak == 8
    assert sorted(client.calls) == list(range(8))
    assert state["project"]["analyze_cursor"] == 8


def test_what_comes_back_is_stored_in_source_order(analyze):
    """get_concepts_for_chunks reads concepts back by id and calls it source order."""
    client = Indexer(slow_first=True)
    state = asyncio.run(analyze([chunk(n) for n in range(8)], client))
    assert state["stored"] == [100 + n for n in range(8)]


def test_the_last_batch_stops_at_the_end_of_the_book(analyze):
    client = Indexer()
    state = asyncio.run(analyze([chunk(n) for n in range(11)], client, cursor=8))
    assert sorted(client.calls) == [8, 9, 10]
    assert state["project"]["analyze_cursor"] == 11


def test_front_matter_costs_no_model_call(analyze):
    chunks = [chunk(0, teachable=False), chunk(1), chunk(2, teachable=False), chunk(3)]
    client = Indexer()
    state = asyncio.run(analyze(chunks, client))
    assert sorted(client.calls) == [1, 3]
    assert state["project"]["analyze_cursor"] == 4


def test_an_excerpt_that_fails_in_the_crowd_is_tried_again_alone(analyze):
    """A provider that throttles a burst must not cost the course the excerpt."""
    client = Indexer(fail_first={2, 5})
    state = asyncio.run(analyze([chunk(n) for n in range(8)], client))
    assert client.calls.count(2) == 2 and client.calls.count(5) == 2
    assert state["stored"] == [100 + n for n in range(8)]


def test_an_excerpt_that_keeps_failing_is_skipped_not_fatal(analyze):
    client = Indexer(fail_always={3})
    state = asyncio.run(analyze([chunk(n) for n in range(6)], client))
    assert state["stored"] == [100, 101, 102, 104, 105]
    assert state["project"]["analyze_cursor"] == 6


def test_the_models_verdict_on_apparatus_still_stands(analyze):
    client = Indexer(apparatus={1})
    state = asyncio.run(analyze([chunk(n) for n in range(3)], client))
    assert state["apparatus"] == [101]
    assert state["stored"] == [100, 102]


def test_a_batch_run_twice_does_not_store_its_ideas_twice(analyze):
    """An interrupted batch is run again from the top; its leftovers go first."""
    state = asyncio.run(analyze([chunk(n) for n in range(4)], Indexer()))
    assert state["cleared"] == [[100, 101, 102, 103]]


def test_one_at_a_time_is_still_available(analyze):
    client = Indexer()
    state = asyncio.run(analyze([chunk(n) for n in range(4)], client, concurrency=1))
    assert client.peak == 1
    assert state["project"]["analyze_cursor"] == 1


def test_the_shipped_default_indexes_several_at_once():
    assert pipeline.ANALYZE_CONCURRENCY > 1


# ── The next script is written while this lesson records ─────────────────────

def lesson(ordinal, status):
    return {"id": 2600 + ordinal, "ordinal": ordinal, "status": status}


@pytest.fixture
def sessions_step(monkeypatch):
    """_phase_sessions with the two halves of a lesson replaced by timers."""
    events = []
    state = {"project": {}, "record_error": None, "write_error": None}

    async def run(lessons, *, write_ahead=True):
        async def get_sessions(pid):
            return lessons

        async def work_ceiling(pid):
            return None

        async def next_session_needing(pid, statuses, ceiling):
            return next((l for l in lessons if l["status"] in statuses), None)

        async def set_project(pid, **fields):
            state["project"].update(fields)

        async def record(project, sess, total, sessions):
            events.append(("record:start", sess["ordinal"]))
            await asyncio.sleep(0.03)
            if state["record_error"]:
                raise state["record_error"]
            events.append(("record:end", sess["ordinal"]))

        async def write(project, client, sess, sessions, *, announce=True):
            events.append(("write:start", sess["ordinal"], announce))
            await asyncio.sleep(0.01)
            if state["write_error"]:
                raise state["write_error"]
            events.append(("write:end", sess["ordinal"]))

        monkeypatch.setattr(pipeline, "WRITE_AHEAD", write_ahead)
        monkeypatch.setattr(pipeline.db, "get_sessions", get_sessions)
        monkeypatch.setattr(pipeline.db, "work_ceiling", work_ceiling)
        monkeypatch.setattr(pipeline.db, "next_session_needing", next_session_needing)
        monkeypatch.setattr(pipeline.db, "set_project", set_project)
        monkeypatch.setattr(pipeline, "_record_lesson", record)
        monkeypatch.setattr(pipeline, "_write_lesson", write)
        await pipeline._phase_sessions({"id": 7}, client=None)

    run.events = events
    run.state = state
    return run


def test_the_first_lesson_has_to_be_written_before_anything_records(sessions_step):
    asyncio.run(sessions_step([lesson(1, "pending"), lesson(2, "pending")]))
    assert sessions_step.events == [("write:start", 1, True), ("write:end", 1)]


def test_the_next_script_is_written_while_this_lesson_records(sessions_step):
    step = sessions_step
    asyncio.run(step([lesson(1, "validated"), lesson(2, "pending"), lesson(3, "pending")]))
    # Lesson 2's script is under way, and finished, before lesson 1's audio is.
    assert step.events.index(("write:start", 2, False)) < step.events.index(("record:end", 1))
    assert step.events.index(("write:end", 2)) < step.events.index(("record:end", 1))
    # One ahead and no further.
    assert not [e for e in step.events if e[0].startswith("write") and e[1] == 3]


def test_the_progress_line_stays_on_the_recording(sessions_step):
    """The listener is waiting for audio; the script behind it is not news."""
    asyncio.run(sessions_step([lesson(1, "validated"), lesson(2, "pending")]))
    assert ("write:start", 2, False) in sessions_step.events


def test_the_last_lesson_of_a_batch_records_alone(sessions_step):
    asyncio.run(sessions_step([lesson(1, "complete"), lesson(2, "validated")]))
    assert sessions_step.events == [("record:start", 2), ("record:end", 2)]


def test_a_reopened_earlier_lesson_is_still_written_first(sessions_step):
    """Regenerate on lesson 1 while lesson 2 waits to be recorded: 1 is what the
    listener hears next, so it is rewritten before anything else happens."""
    asyncio.run(sessions_step([lesson(1, "pending"), lesson(2, "validated")]))
    assert sessions_step.events == [("write:start", 1, True), ("write:end", 1)]


def test_switched_off_they_alternate_as_before(sessions_step):
    asyncio.run(sessions_step([lesson(1, "validated"), lesson(2, "pending")], write_ahead=False))
    assert sessions_step.events == [("record:start", 1), ("record:end", 1)]


def test_a_failed_recording_fails_the_step_once_the_writer_has_finished(sessions_step):
    """No task is left writing to a book the worker has already given up on."""
    sessions_step.state["record_error"] = RuntimeError("txt-to-audio HTTP 500")
    with pytest.raises(RuntimeError, match="txt-to-audio"):
        asyncio.run(sessions_step([lesson(1, "validated"), lesson(2, "pending")]))
    assert ("write:end", 2) in sessions_step.events


def test_a_failed_script_still_fails_the_step(sessions_step):
    sessions_step.state["write_error"] = RuntimeError("DeepSeek call failed after retries")
    with pytest.raises(RuntimeError, match="DeepSeek"):
        asyncio.run(sessions_step([lesson(1, "validated"), lesson(2, "pending")]))
    assert ("record:end", 1) in sessions_step.events


def test_writing_ahead_ships_switched_on():
    assert pipeline.WRITE_AHEAD is True


# ── A course's cover and author ──────────────────────────────────────────────

@pytest.mark.parametrize("raw", [
    "Three Initiates",
    "George Washington Carey, Inez Eudora Perry",
    "  Marcus   Aurelius ; ",
    "José Ortega y Gasset",
])
def test_a_persons_name_is_kept(raw):
    assert cover.clean_author(raw) == " ".join(raw.split()).strip(" ;,")


@pytest.mark.parametrize("raw", [
    None, "", "   ", "Administrator", "user", "Microsoft Office User", "Unknown",
    "calibre 6.4", "jane@example.com", "C:\\Users\\jane", "home/jane", "12345", "x" * 200,
])
def test_what_no_person_typed_is_not_printed_as_an_author(raw):
    assert cover.clean_author(raw) is None


def test_only_the_cover_this_module_wrote_is_ever_served(tmp_path):
    (tmp_path / cover.COVER_NAME).write_bytes(b"jpeg")
    (tmp_path / "source.pdf").write_bytes(b"pdf")
    assert cover.cover_file(tmp_path, cover.COVER_NAME) == tmp_path / cover.COVER_NAME
    assert cover.cover_file(tmp_path, "source.pdf") is None
    assert cover.cover_file(tmp_path, "../other/cover.jpg") is None
    assert cover.cover_file(tmp_path, None) is None


def test_no_stored_cover_means_no_cover(tmp_path):
    assert cover.cover_file(tmp_path, cover.COVER_NAME) is None


@pytest.mark.parametrize("fmt", ["text", "url", "docx", "markdown", "mobi"])
def test_sources_with_no_cover_to_find_are_left_alone(tmp_path, fmt):
    src = tmp_path / "source.bin"
    src.write_bytes(b"anything")
    assert cover.extract(fmt, str(src), tmp_path) == {}
    assert list(tmp_path.iterdir()) == [src]


def test_a_file_that_cannot_be_read_costs_the_course_nothing(tmp_path):
    """Never raises: a book without a cover is still a book."""
    broken = tmp_path / "source.pdf"
    broken.write_bytes(b"this is not a pdf")
    assert cover.extract("pdf", str(broken), tmp_path) == {}
    assert cover.extract("epub", str(tmp_path / "missing.epub"), tmp_path) == {}


def test_a_cover_is_stored_small_enough_to_be_a_thumbnail(tmp_path):
    image_lib = pytest.importorskip("PIL.Image")
    from io import BytesIO

    buf = BytesIO()
    image_lib.new("RGBA", (2400, 3600), (90, 60, 200, 255)).save(buf, "PNG")
    dest = tmp_path / cover.COVER_NAME
    assert cover._save_cover(buf.getvalue(), dest) is True
    with image_lib.open(dest) as stored:
        assert stored.format == "JPEG"
        assert stored.width <= cover.COVER_MAX_WIDTH and stored.height <= cover.COVER_MAX_HEIGHT


def test_a_publishers_mark_is_not_a_cover(tmp_path):
    image_lib = pytest.importorskip("PIL.Image")
    from io import BytesIO

    buf = BytesIO()
    image_lib.new("RGB", (48, 48), (0, 0, 0)).save(buf, "PNG")
    dest = tmp_path / cover.COVER_NAME
    assert cover._save_cover(buf.getvalue(), dest) is False
    assert not dest.exists()


# ── What the dashboard is told about a course ────────────────────────────────

def project_row(**overrides):
    row = {
        "id": 61, "name": "The Kybalion", "source_format": "epub", "status": "processing",
        "phase": "sessions", "stage": "Phase 1 of 2 · recording lesson 3 of 20",
        "progress": 61, "validation_status": "pending", "total_audio_seconds": 0,
        "options": {},
    }
    row.update(overrides)
    return row


def test_a_course_carries_its_byline_and_says_whether_it_has_a_cover():
    from phansora.products.book_alchemy import server as ba_server

    wire = ba_server._ba_project_wire(project_row(options={
        "author": "Three Initiates", "cover": cover.COVER_NAME,
        "voice": "v_arthur", "depth": "overview", "instruct_text": "calm",
    }))
    assert wire["author"] == "Three Initiates"
    assert wire["has_cover"] is True
    assert (wire["voice"], wire["depth"], wire["instruct_text"]) == ("v_arthur", "overview", "calm")


def test_a_course_made_before_covers_existed_still_wires():
    """Options arrive as a JSON string from the database, and older rows have neither key."""
    from phansora.products.book_alchemy import server as ba_server

    wire = ba_server._ba_project_wire(project_row(options='{"voice": "default"}'))
    assert wire["author"] is None and wire["has_cover"] is False
    assert wire["depth"] == pipeline.DEFAULT_DEPTH

    assert ba_server._ba_project_wire(project_row(options=None))["has_cover"] is False


def test_a_renamed_course_keeps_the_words_its_listener_typed():
    from phansora.products.book_alchemy import server as ba_server

    assert ba_server._ba_course_name("  Course:\tHermetic\n Basics  ") == "Course: Hermetic Basics"
    assert ba_server._ba_course_name("\x00\x07") == ""
    assert len(ba_server._ba_course_name("x" * 500)) == 200
