"""
Who gets the voice model when a book and a person want it at the same time.

The reported failure was "Couldn't reach the voice service" in Narrava Studio while a Book
Alchemy job was running. Book Alchemy calls the same endpoint over HTTP, once per chunk, so
a book is hundreds of requests arriving through the door a person is also standing in.

What is pinned here is the guarantee, not the tuning: batch is refused the last slot, so an
interactive caller can never find the model entirely taken by a book. That is a stronger
property than "batch waits longer", and it is the one that makes starvation impossible
rather than unlikely.
"""

import asyncio

import pytest

from phansora.products.spokenverse import admission

# Captured at import, BEFORE the autouse fixture below shortens them. Read live, these
# would be whatever the fixture last set, and the assertion would pass while saying
# nothing — a test that measures its own scaffolding.
SHIPPED = {
    "slots": admission._SLOTS,
    "batch_slots": admission._BATCH_SLOTS,
    "interactive_wait": admission._INTERACTIVE_WAIT_S,
    "batch_wait": admission._BATCH_WAIT_S,
    "batch_max_yield": admission._BATCH_MAX_YIELD_S,
}


@pytest.fixture(autouse=True)
def fresh_gate(monkeypatch):
    """
    A gate per test — the module-level one is process-wide by design.

    The waits are shortened too, and that is not cosmetic: batch waits fifteen minutes in
    production before it gives up, which is correct there and would have this suite sitting
    silently for a quarter of an hour. The real defaults are asserted separately, below.
    """
    monkeypatch.setattr(admission, "_gate", admission._Gate())
    monkeypatch.setattr(admission, "_turns", admission._Turns())
    monkeypatch.setattr(admission, "_INTERACTIVE_WAIT_S", 0.2)
    monkeypatch.setattr(admission, "_BATCH_WAIT_S", 0.2)
    yield


@pytest.mark.asyncio
async def test_a_slot_is_held_for_the_duration_of_the_work():
    assert admission.stats()["in_flight"] == 0
    async with admission.slot("interactive"):
        assert admission.stats()["in_flight"] == 1
    assert admission.stats()["in_flight"] == 0


@pytest.mark.asyncio
async def test_batch_can_never_take_the_last_slot():
    """The guarantee. With two slots, one book holds one and cannot have the other."""
    held = asyncio.Event()
    release = asyncio.Event()

    async def book():
        async with admission.slot("batch"):
            held.set()
            await release.wait()

    task = asyncio.create_task(book())
    await asyncio.wait_for(held.wait(), timeout=2)

    # A second book is refused rather than queued behind the first, because the batch
    # allowance is one slot fewer than the total.
    with pytest.raises(admission.Busy):
        async with admission.slot("batch"):
            pass

    # And the person walks straight in, with a book already running.
    async with admission.slot("interactive"):
        assert admission.stats()["in_flight"] == 2

    release.set()
    await task


@pytest.mark.asyncio
async def test_interactive_waits_rather_than_failing_when_a_slot_frees_up():
    """Refusal is the last resort, not the first answer — a short wait is still served."""
    release = asyncio.Event()

    async def holder():
        async with admission.slot("interactive"):
            await release.wait()

    a = asyncio.create_task(holder())
    b = asyncio.create_task(holder())
    await asyncio.sleep(0.05)

    async def latecomer():
        async with admission.slot("interactive"):
            return "served"

    late = asyncio.create_task(latecomer())
    await asyncio.sleep(0.05)
    assert not late.done()          # waiting, not refused

    release.set()
    assert await asyncio.wait_for(late, timeout=2) == "served"
    await asyncio.gather(a, b)


@pytest.mark.asyncio
async def test_busy_carries_a_retry_after_that_is_never_zero():
    """A Retry-After of 0 invites a client to hammer a service that just said it was full."""
    release = asyncio.Event()

    async def holder():
        async with admission.slot("interactive"):
            await release.wait()

    tasks = [asyncio.create_task(holder()) for _ in range(admission._SLOTS)]
    await asyncio.sleep(0.05)

    with pytest.raises(admission.Busy) as exc:
        async with admission.slot("interactive"):
            pass
    assert exc.value.retry_after >= 5

    release.set()
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_a_slot_is_returned_even_when_the_work_raises():
    """A leaked slot is a permanent reduction in capacity — worse than the original error."""
    with pytest.raises(ValueError):
        async with admission.slot("batch"):
            raise ValueError("synthesis blew up")
    assert admission.stats()["in_flight"] == 0
    # And proves it is genuinely free: two more batch acquisitions in a row would deadlock
    # if the batch semaphore had leaked.
    for _ in range(3):
        async with admission.slot("batch"):
            pass


class TestPriorityHeader:
    """An unmarked caller is a browser until proven otherwise."""

    def test_batch_only_when_declared(self):
        assert admission.priority_from_headers({"x-phansora-priority": "batch"}) == "batch"
        assert admission.priority_from_headers({"x-phansora-priority": "BATCH"}) == "batch"
        assert admission.priority_from_headers({"x-phansora-priority": " batch "}) == "batch"

    def test_everything_else_is_interactive(self):
        for headers in ({}, {"x-phansora-priority": ""}, {"x-phansora-priority": "low"},
                        {"x-phansora-priority": "interactive"}, {"other": "batch"}):
            assert admission.priority_from_headers(headers) == "interactive"

    def test_a_header_bag_that_misbehaves_does_not_take_the_endpoint_down(self):
        class Hostile:
            def get(self, _key):
                raise RuntimeError("nope")
        assert admission.priority_from_headers(Hostile()) == "interactive"


def test_the_shipped_defaults_are_what_the_design_says():
    """
    The fixture above shortens the waits, so nothing else in this file would notice if the
    real numbers drifted. These are the ones that ship.
    """
    assert SHIPPED["slots"] >= 2, "one slot cannot reserve anything for interactive work"
    assert SHIPPED["batch_slots"] == SHIPPED["slots"] - 1, "batch must be capped below the total"
    # A person QUEUES rather than being turned away — the wait is for another person's
    # narration, which runs at full speed now that a book pauses for it. Bounded by the
    # proxy: nginx gives the API 300s, and the synthesis has to fit in what is left.
    assert 60 <= SHIPPED["interactive_wait"] <= 180, "queue, but answer before the proxy gives up"
    assert SHIPPED["batch_wait"] >= 300, "nobody is watching a book; let it wait"
    # Long enough that standing aside is the normal case, bounded so a busy box cannot
    # stop a course rendering altogether.
    assert 60 <= SHIPPED["batch_max_yield"] <= 3600, "a book yields for a while, not forever"


# ── Rule 3: standing aside ───────────────────────────────────────────────────
# The guarantee above gets a person a SLOT. It does not get them the GPU, and on one
# device two syntheses do not run twice as fast — measured on prod, a ~24s narration took
# 4m24s next to a book. So a book waits for the box rather than merely for a slot.


@pytest.mark.asyncio
async def test_a_book_does_not_start_while_a_person_is_on_the_box():
    """The free slot is deliberately left alone. This is the whole change."""
    took_slot = asyncio.Event()

    async def book():
        async with admission.slot("batch"):
            took_slot.set()

    async with admission.slot("interactive"):
        task = asyncio.create_task(book())
        await asyncio.sleep(0.05)
        # There IS a slot free — two total, one held. The old gate handed it straight to
        # the book, and the person's synthesis then ran at half speed alongside it.
        assert admission.stats()["in_flight"] == 1
        assert not took_slot.is_set(), "the book took the GPU out from under a person"
        assert admission.stats()["batch_yields"] == 1

    # ...and it resumes the moment they are done, with nothing to poke it.
    await asyncio.wait_for(task, timeout=2)
    assert took_slot.is_set()


@pytest.mark.asyncio
async def test_someone_still_queueing_is_enough_to_hold_a_book_back():
    """Waiting counts, not just running — by the time they are served it is too late."""
    release = asyncio.Event()

    async def holder():
        async with admission.slot("interactive"):
            await release.wait()

    holders = [asyncio.create_task(holder()) for _ in range(admission._SLOTS)]
    await asyncio.sleep(0.05)

    async def latecomer():
        async with admission.slot("interactive"):
            return "served"

    late = asyncio.create_task(latecomer())
    await asyncio.sleep(0.05)

    # Two being served and one in the queue: all three hold a book back.
    assert admission.stats()["interactive_present"] == admission._SLOTS + 1

    release.set()
    await asyncio.wait_for(late, timeout=2)
    await asyncio.gather(*holders)
    assert admission.stats()["interactive_present"] == 0


@pytest.mark.asyncio
async def test_a_book_is_never_starved_by_a_box_that_is_never_quiet(monkeypatch):
    """Standing aside is bounded. A course that never renders is the worse failure."""
    monkeypatch.setattr(admission, "_BATCH_MAX_YIELD_S", 0.1)
    took_slot = asyncio.Event()

    async def book():
        async with admission.slot("batch"):
            took_slot.set()

    # The person never leaves for the duration of this block.
    async with admission.slot("interactive"):
        await asyncio.wait_for(book(), timeout=2)

    assert took_slot.is_set()
    assert admission.stats()["batch_overrides"] == 1


@pytest.mark.asyncio
async def test_a_cancelled_caller_stops_holding_books_back():
    """A browser that disconnects mid-wait must not leave the count stuck above zero.

    It would be a quiet, permanent one: every later book stands aside for somebody who
    left, until the process restarts.
    """
    release = asyncio.Event()

    async def holder():
        async with admission.slot("interactive"):
            await release.wait()

    holders = [asyncio.create_task(holder()) for _ in range(admission._SLOTS)]
    await asyncio.sleep(0.05)

    waiter = asyncio.create_task(latecomer_that_gives_up())
    await asyncio.sleep(0.05)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    release.set()
    await asyncio.gather(*holders)
    assert admission.stats()["interactive_present"] == 0


async def latecomer_that_gives_up():
    async with admission.slot("interactive"):
        pass


# ── Rule 4: a book already running pauses at its next piece ───────────────────
# Rule 3 holds back the NEXT lesson; a lesson already running used to hold the engine for
# a whole chunk at a time, and a plain lock gave the handoff to whichever thread won — a
# book's four threads, usually. Reported from Narrava Studio as "The voice service is busy"
# while a Book Alchemy course rendered. These drive the engine's turns() the way the
# synthesis threads do: from threads, each carrying its request's ticket.

import threading
import time


def _piece(ticket, log, name, started=None, hold=0.0):
    """One engine piece from a worker thread, as the synthesis loop takes it."""
    def run():
        admission._TICKET.set(ticket)
        with admission.turn():
            log.append(name)
            if started is not None:
                started.set()
            time.sleep(hold)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _queued(count, timeout=2.0):
    """Wait until `count` pieces are queued for the engine."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with admission._turns._cond:
            if len(admission._turns._queue) >= count:
                return
        time.sleep(0.005)
    raise AssertionError(f"never saw {count} piece(s) queued")


BOOK, PERSON = 1, 0


def test_a_person_gets_the_engine_before_a_book_that_was_waiting_first():
    log, running = [], threading.Event()
    holder = _piece((BOOK, 1), log, "book-in-progress", started=running, hold=0.2)
    assert running.wait(2)

    book_next = _piece((BOOK, 1), log, "book-next")
    _queued(1)
    person = _piece((PERSON, 2), log, "person")
    _queued(2)

    for t in (holder, book_next, person):
        t.join(2)
    # The piece in progress finishes — nothing interrupts it — then the person, then the book.
    assert log == ["book-in-progress", "person", "book-next"]


def test_a_book_holds_back_while_a_person_is_present_even_with_nobody_queued():
    """The gap between two pieces of a narration is the one a book used to take."""
    log, done = [], threading.Event()
    admission._gate._enter_interactive()  # a narration mid-request, between pieces
    book = _piece((BOOK, 1), log, "book", started=done)

    time.sleep(0.3)
    assert log == [], "a book must not take the gap between a person's pieces"

    admission._gate._leave_interactive()
    # Woken by the person leaving, not by its next poll.
    assert done.wait(0.5), "a paused book resumes the moment the person is served"
    book.join(2)
    assert admission.stats()["batch_pauses"] == 1


def test_people_are_served_in_the_order_they_arrived():
    """One narration finishes before the next starts, instead of both at half speed."""
    log, running = [], threading.Event()
    holder = _piece((PERSON, 0), log, "first", started=running, hold=0.2)
    assert running.wait(2)

    later = _piece((PERSON, 9), log, "arrived-later")
    _queued(1)
    earlier = _piece((PERSON, 3), log, "arrived-earlier")  # queued second, arrived first
    _queued(2)

    for t in (holder, later, earlier):
        t.join(2)
    assert log == ["first", "arrived-earlier", "arrived-later"]


def test_a_book_is_not_stopped_forever_by_a_box_that_is_never_quiet(monkeypatch):
    monkeypatch.setattr(admission, "_BATCH_MAX_YIELD_S", 0.2)
    log, done = [], threading.Event()
    admission._gate._enter_interactive()  # and never leaves
    try:
        book = _piece((BOOK, 1), log, "book", started=done)
        assert done.wait(2.5), "a book that yields forever never finishes the course"
        book.join(2)
        assert admission.stats()["batch_piece_overrides"] == 1
    finally:
        admission._gate._leave_interactive()


def test_the_engine_is_handed_back_when_a_piece_raises():
    def failing():
        with admission.turn():
            raise RuntimeError("synthesis blew up")

    with pytest.raises(RuntimeError):
        failing()

    log, done = [], threading.Event()
    _piece((PERSON, 1), log, "next", started=done)
    assert done.wait(1), "a failed piece must not keep the engine"


@pytest.mark.asyncio
async def test_the_request_ticket_reaches_the_synthesis_threads():
    """to_thread copies the context — the ticket needs no plumbing through the pipeline."""
    async with admission.slot("batch"):
        rank, _ = await asyncio.to_thread(admission._TICKET.get)
        assert rank == BOOK
    async with admission.slot("interactive"):
        rank, _ = await asyncio.to_thread(admission._TICKET.get)
        assert rank == PERSON
    assert admission._TICKET.get() is None
