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

     What this cannot do is interrupt a lesson already in flight, and nothing here should:
     the work is most of the way to an audio file nobody would get back. So one lesson may
     still overlap one person. Shrinking THAT window is a question about how finely a
     lesson is cut into requests, not about scheduling, and it is not answered here.

And a caller that cannot be served soon is TOLD SO, quickly, with Retry-After. The failure
this replaces was a request that simply never came back — which the browser reports as a
network error, indistinguishable from the service being down. A fast, honest 503 is a
better answer than a slow one, and far better than none.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from typing import AsyncIterator, Literal

LOG = logging.getLogger("spokenverse.admission")

Priority = Literal["interactive", "batch"]

# How many syntheses may run at once. One model on one GPU: this is about not queueing work
# inside the model that is better queued outside it, where it can be measured and refused.
_SLOTS = max(1, int(os.getenv("TTS_MAX_CONCURRENT", "2")))

# Slots batch work may occupy. Always at least one fewer than the total, so an interactive
# caller can never find the model completely taken by a book.
_BATCH_SLOTS = max(1, _SLOTS - 1) if _SLOTS > 1 else 1

# How long a caller waits for a slot before being told to come back. Interactive is short
# because somebody is watching; batch is long because nobody is.
_INTERACTIVE_WAIT_S = float(os.getenv("TTS_INTERACTIVE_WAIT_S", "20"))
_BATCH_WAIT_S = float(os.getenv("TTS_BATCH_WAIT_S", "900"))

# How long batch will stand aside before taking its turn anyway.
#
# Never forever. On a box that is never completely quiet — which is what a hundred users
# looks like — a book that yielded unconditionally would never finish, and "the course
# never rendered" is a worse outcome than "one narration was slow". After this it goes,
# still capped by _BATCH_SLOTS, so a person keeps their reserved slot regardless.
_BATCH_MAX_YIELD_S = float(os.getenv("TTS_BATCH_MAX_YIELD_S", "600"))


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


def slot(priority: Priority = "interactive"):
    """Hold a synthesis slot for the duration of the block, or raise Busy."""
    return _gate.slot(priority)


def stats() -> dict:
    return {
        "slots": _SLOTS,
        "batch_slots": _BATCH_SLOTS,
        "in_flight": _gate.in_flight,
        "interactive_present": _gate._interactive,
        "batch_yields": _gate.batch_yields,
        "batch_overrides": _gate.batch_overrides,
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
