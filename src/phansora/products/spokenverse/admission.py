"""
Who gets to use the voice model, and who waits.

WHY THIS EXISTS. There is one TTS model, loaded once, in one process, and every product
reaches it through the same endpoint — including Book Alchemy, whose worker calls back
into this API over HTTP for every chunk of a book. A book is hundreds of those requests.
Before this, they and a person waiting on a narration in Narrava Studio were the same kind
of caller, arriving through the same door, with nothing deciding between them. A book could
take every slot and hold them for as long as the book took.

THREE RULES, and the third is the one that makes the difference people can feel:

  1. A bounded number of syntheses run at once. More than the model can do concurrently is
     not throughput, it is the same work in a worse order plus the memory to hold it.

  2. BATCH CAN NEVER TAKE THE LAST SLOT. Not "batch waits longer" — batch is refused the
     final slot outright, so there is always one reserved for somebody watching a spinner.
     A priority queue would only reorder the wait; this makes starvation impossible rather
     than unlikely, which is the property worth having when the alternative is a person
     staring at a page that looks broken.

  3. BATCH STANDS ASIDE WHILE ANYONE INTERACTIVE IS HERE — waiting or running, not just
     running. Rule 2 gets a person a slot; it does not get them the GPU. There is one
     device, and two syntheses sharing it do not run twice as fast, they each run at about
     half speed. Measured on prod: a narration that takes ~24 seconds on a quiet box took
     4m24s alongside a book. So a book waits between lessons rather than running straight
     through. A lesson takes about fourteen minutes and a narration about twenty-five
     seconds, so standing aside costs the book a fraction of a percent and gives the person
     back an order of magnitude — the most lopsided trade in the system.

  4. A BOOK ALREADY RUNNING PAUSES AT THE NEXT PIECE. Rule 3 holds back the next lesson;
     it cannot touch one already in flight, and a lesson runs for many minutes. What made
     that window hurt was not the overlap itself but the lock inside the engine: it was
     held for a whole ~2,500-character chunk, a lesson fed it four chunks at once, and a
     plain lock is not first-come-first-served — so a narration kept losing the handoff
     and took minutes. The engine now takes turns() once per PIECE (the ~200 characters it
     synthesizes in one call, a few seconds of GPU), and a turn goes to a person before a
     book, people in the order they arrived. A book waits while anyone interactive is
     present, not merely queued: a narration's pieces come one after another, and between
     two of them it is briefly not in the queue at all — a gap a book would otherwise take,
     one piece at a time, which is the half-speed sharing this replaces. So a person waits
     at most for the piece a book is in the middle of, and the lesson resumes where it
     stopped. Bounded like rule 3: a box that is never quiet cannot stop a book forever.

A person who arrives while every slot is taken QUEUES rather than being turned away: with a
book pausing at its next piece, the wait is for another person's narration, which now runs
at full speed. Only a wait long enough to approach the proxy's read timeout is refused, with
a 503 and Retry-After — and the browser retries that on its own, so it is not a message
anyone is asked to act on. The failure this replaced was a request that simply never came
back, which the browser reports as the service being down.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import heapq
import itertools
import logging
import os
import threading
import time
from typing import AsyncIterator, Iterator, Literal, Optional

LOG = logging.getLogger("spokenverse.admission")

Priority = Literal["interactive", "batch"]

# How many syntheses may run at once. One model on one GPU: this is about not queueing work
# inside the model that is better queued outside it, where it can be measured and refused.
_SLOTS = max(1, int(os.getenv("TTS_MAX_CONCURRENT", "2")))

# Slots batch work may occupy. Always at least one fewer than the total, so an interactive
# caller can never find the model completely taken by a book.
_BATCH_SLOTS = max(1, _SLOTS - 1) if _SLOTS > 1 else 1

# How long a caller waits for a slot before being told to come back.
#
# Interactive QUEUES: the wait it is sitting out is another person's narration, which runs
# at full speed now that a book pauses for it (rule 4). The ceiling is the proxy, not
# patience — nginx gives the API 300s to answer, and the synthesis itself must fit in what
# is left, so a wait near that would come back as a gateway timeout instead of a clean 503.
# Batch waits long because nobody is watching.
_INTERACTIVE_WAIT_S = float(os.getenv("TTS_INTERACTIVE_WAIT_S", "120"))
_BATCH_WAIT_S = float(os.getenv("TTS_BATCH_WAIT_S", "900"))

# How long batch will stand aside before taking its turn anyway.
#
# Never forever. On a box that is never completely quiet — which is what a hundred users
# looks like — a book that yielded unconditionally would never finish, and "the course
# never rendered" is a worse outcome than "one narration was slow". After this it goes,
# still capped by _BATCH_SLOTS, so a person keeps their reserved slot regardless.
_BATCH_MAX_YIELD_S = float(os.getenv("TTS_BATCH_MAX_YIELD_S", "600"))

# Who is asking, carried from the request down to the engine's worker threads: (rank,
# arrival), rank 0 for a person and 1 for a book. create_task and asyncio.to_thread both
# copy the context, so the pipeline's chunk tasks and the threads they run in all see the
# request's ticket without it being threaded through every signature in between.
_TICKET: contextvars.ContextVar[Optional[tuple[int, int]]] = contextvars.ContextVar(
    "tts_ticket", default=None
)
_arrivals = itertools.count()


class Busy(Exception):
    """No slot became free in time. Carries the Retry-After the caller should honour."""

    def __init__(self, retry_after: int) -> None:
        super().__init__("The voice service is busy.")
        self.retry_after = retry_after


class _Gate:
    def __init__(self) -> None:
        self._all = asyncio.Semaphore(_SLOTS)
        # A second, smaller semaphore that ONLY batch takes. Holding both is what caps
        # batch at _BATCH_SLOTS while interactive is capped only by the real total.
        self._batch = asyncio.Semaphore(_BATCH_SLOTS)
        # Interactive callers WAITING OR RUNNING. Waiting counts: the point of rule 3 is to
        # get out of the way of somebody who has just arrived, and by the time they are
        # running it is already too late to have stayed off the GPU for them.
        self._interactive = 0
        # Clear while any interactive caller is present. Batch waits on this rather than
        # polling, so a book resumes the instant the last person is served instead of at
        # the top of some next tick.
        self._quiet = asyncio.Event()
        self._quiet.set()
        self.in_flight = 0
        # Observability, because "the book got slower" and "the book stopped" look the same
        # from outside and want very different responses.
        self.batch_yields = 0
        self.batch_overrides = 0

    def _enter_interactive(self) -> None:
        self._interactive += 1
        self._quiet.clear()

    def _leave_interactive(self) -> None:
        self._interactive = max(0, self._interactive - 1)
        if self._interactive == 0:
            self._quiet.set()
            # A book paused mid-lesson (rule 4) is waiting in a worker thread, not on
            # _quiet; tell it the box is clear so it resumes now rather than on its next poll.
            _turns.wake()

    def interactive_present(self) -> bool:
        return self._interactive > 0

    async def _stand_aside(self) -> None:
        """Wait for the box to go quiet before a book takes the GPU. Rule 3.

        Bounded by _BATCH_MAX_YIELD_S, after which it goes anyway — see that constant.

        There is a race here and it is the acceptable one: quiet can be signalled and an
        interactive caller arrive before the semaphore below is taken, which leaves one
        lesson overlapping one person. That is the same window as a lesson already in
        flight, and closing it would mean holding the gate across the acquire — turning a
        yield into a lock that a person then queues behind.
        """
        if self._quiet.is_set():
            return
        self.batch_yields += 1
        LOG.info("batch standing aside — %d interactive caller(s) ahead", self._interactive)
        try:
            await asyncio.wait_for(self._quiet.wait(), timeout=_BATCH_MAX_YIELD_S)
        except asyncio.TimeoutError:
            self.batch_overrides += 1
            LOG.warning(
                "batch stood aside %.0fs and is taking its turn — the box has not been "
                "quiet for that long, so a book would otherwise never finish",
                _BATCH_MAX_YIELD_S,
            )

    @contextlib.asynccontextmanager
    async def slot(self, priority: Priority) -> AsyncIterator[None]:
        wait = _INTERACTIVE_WAIT_S if priority == "interactive" else _BATCH_WAIT_S
        interactive = priority == "interactive"
        # The ticket the engine's turns() reads — taken on ARRIVAL, so people are served in
        # the order they came, not the order they happened to get a slot.
        ticket = _TICKET.set((0 if interactive else 1, next(_arrivals)))
        try:
            async with self._admitted(priority, wait, interactive):
                yield
        finally:
            _TICKET.reset(ticket)

    @contextlib.asynccontextmanager
    async def _admitted(self, priority: Priority, wait: float, interactive: bool) -> AsyncIterator[None]:
        held_batch = False
        # Counted BEFORE the wait, so a person queueing is already making the next lesson
        # stand aside rather than only registering once they are served.
        if interactive:
            self._enter_interactive()
        try:
            if not interactive:
                await self._stand_aside()
                await asyncio.wait_for(self._batch.acquire(), timeout=wait)
                held_batch = True
            await asyncio.wait_for(self._all.acquire(), timeout=wait)
        except asyncio.TimeoutError as exc:
            if held_batch:
                self._batch.release()
            if interactive:
                self._leave_interactive()
            # Round up, and never advise zero — a Retry-After of 0 is an invitation to
            # hammer a service that just said it was full.
            raise Busy(retry_after=max(5, int(wait))) from exc
        except BaseException:
            # A disconnect or a shutdown cancels the acquire. Without this the count never
            # comes back down and every later book stands aside for a caller that left.
            if held_batch:
                self._batch.release()
            if interactive:
                self._leave_interactive()
            raise

        self.in_flight += 1
        LOG.info("tts slot taken (%s) — %d/%d in flight", priority, self.in_flight, _SLOTS)
        try:
            yield
        finally:
            self.in_flight -= 1
            self._all.release()
            if held_batch:
                self._batch.release()
            if interactive:
                self._leave_interactive()


_gate = _Gate()


class _Turns:
    """The engine itself, one piece at a time, to whoever should have it next. Rule 4.

    Thread-level, because synthesis runs in worker threads (asyncio.to_thread) and the
    engine is one shared vLLM instance that must see one call at a time. This replaces the
    plain lock that used to serialize it: same exclusion, but the release goes to the right
    waiter instead of whichever thread wins the race — and it is taken per piece, so a book
    has a stopping point every few seconds instead of every chunk.

    Order: a person before a book, then arrival order of the REQUEST (so one person's
    narration finishes before the next person's starts, rather than the two interleaving at
    half speed each), then arrival of the piece.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._held = False
        self._queue: list[tuple[int, int, int]] = []
        self._pieces = itertools.count()
        # When the current pause began, None while books are running. One clock for all
        # books rather than one per piece: a per-piece deadline would let a book through
        # once per _BATCH_MAX_YIELD_S — a lesson a piece every ten minutes, which is stopped.
        self._paused_since: Optional[float] = None
        self._overriding = False
        self.batch_pauses = 0
        self.batch_overrides = 0

    def wake(self) -> None:
        with self._cond:
            self._cond.notify_all()

    @contextlib.contextmanager
    def take(self) -> Iterator[None]:
        # Unmarked work — the startup warm-up, a voice preview that never came through
        # slot() — ranks as a person, for the same reason an unmarked header does.
        rank, request = _TICKET.get() or (0, next(_arrivals))
        me = (rank, request, next(self._pieces))
        book = rank == 1
        # A book yields while anyone interactive is PRESENT, not just queued: between two
        # pieces of a narration there is a moment when it holds nothing and waits for
        # nothing, and a book allowed into that gap takes it, piece after piece.
        def overdue() -> bool:
            return (self._paused_since is not None
                    and time.monotonic() - self._paused_since >= _BATCH_MAX_YIELD_S)

        def ready() -> bool:
            if self._held or self._queue[0] != me:
                return False
            return not book or not _gate.interactive_present() or overdue()

        with self._cond:
            heapq.heappush(self._queue, me)
            try:
                while not ready():
                    if book and _gate.interactive_present() and self._paused_since is None:
                        self._paused_since = time.monotonic()
                        self.batch_pauses += 1
                        LOG.info("book paused between pieces — interactive work goes first")
                    # A book re-checks now and then as well as on notify: its deadline passes
                    # without anybody signalling it.
                    self._cond.wait(timeout=1.0 if book else None)
            except BaseException:
                self._queue.remove(me)
                heapq.heapify(self._queue)
                self._cond.notify_all()
                raise
            heapq.heappop(self._queue)
            self._held = True
            if book and self._paused_since is not None:
                if _gate.interactive_present():
                    # Still not quiet: going anyway, on the same terms as rule 3's override.
                    # The pause clock is left running so the rest of the lesson goes too,
                    # still behind any person already queued.
                    if not self._overriding:
                        self._overriding = True
                        self.batch_overrides += 1
                        LOG.warning(
                            "book paused %.0fs and is taking turns anyway — the box has not "
                            "been quiet for that long", _BATCH_MAX_YIELD_S,
                        )
                else:
                    self._paused_since = None
                    self._overriding = False
                    LOG.info("book resumed")
        try:
            yield
        finally:
            with self._cond:
                self._held = False
                self._cond.notify_all()


_turns = _Turns()


def slot(priority: Priority = "interactive"):
    """Hold a synthesis slot for the duration of the block, or raise Busy."""
    return _gate.slot(priority)


def turn():
    """Hold the engine for one piece of synthesis. Blocks its thread until it is this caller's
    turn; see _Turns. Use around each engine call, never around a whole request."""
    return _turns.take()


def stats() -> dict:
    return {
        "slots": _SLOTS,
        "batch_slots": _BATCH_SLOTS,
        "in_flight": _gate.in_flight,
        "interactive_present": _gate._interactive,
        "batch_yields": _gate.batch_yields,
        "batch_overrides": _gate.batch_overrides,
        "batch_pauses": _turns.batch_pauses,
        "batch_piece_overrides": _turns.batch_overrides,
    }


def priority_from_headers(headers) -> Priority:
    """
    Batch only when a caller says so. Defaulting to interactive is deliberate: an unmarked
    caller is a browser until proven otherwise, and the cost of being wrong that way is a
    book waiting slightly longer, not a person being starved.
    """
    value = ""
    try:
        value = (headers.get("x-phansora-priority") or "").strip().lower()
    except Exception:
        return "interactive"
    return "batch" if value == "batch" else "interactive"
