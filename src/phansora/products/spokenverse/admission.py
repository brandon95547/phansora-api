"""
Who gets to use the voice model, and who waits.

WHY THIS EXISTS. There is one TTS model, loaded once, in one process, and every product
reaches it through the same endpoint — including Book Alchemy, whose worker calls back
into this API over HTTP for every chunk of a book. A book is hundreds of those requests.
Before this, they and a person waiting on a narration in Narrava Studio were the same kind
of caller, arriving through the same door, with nothing deciding between them. A book could
take every slot and hold them for as long as the book took.

TWO RULES, and the second is the one that matters:

  1. A bounded number of syntheses run at once. More than the model can do concurrently is
     not throughput, it is the same work in a worse order plus the memory to hold it.

  2. BATCH CAN NEVER TAKE THE LAST SLOT. Not "batch waits longer" — batch is refused the
     final slot outright, so there is always one reserved for somebody watching a spinner.
     A priority queue would only reorder the wait; this makes starvation impossible rather
     than unlikely, which is the property worth having when the alternative is a person
     staring at a page that looks broken.

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
        self.in_flight = 0

    @contextlib.asynccontextmanager
    async def slot(self, priority: Priority) -> AsyncIterator[None]:
        wait = _INTERACTIVE_WAIT_S if priority == "interactive" else _BATCH_WAIT_S
        held_batch = False
        try:
            if priority == "batch":
                await asyncio.wait_for(self._batch.acquire(), timeout=wait)
                held_batch = True
            await asyncio.wait_for(self._all.acquire(), timeout=wait)
        except asyncio.TimeoutError as exc:
            if held_batch:
                self._batch.release()
            # Round up, and never advise zero — a Retry-After of 0 is an invitation to
            # hammer a service that just said it was full.
            raise Busy(retry_after=max(5, int(wait))) from exc

        self.in_flight += 1
        LOG.info("tts slot taken (%s) — %d/%d in flight", priority, self.in_flight, _SLOTS)
        try:
            yield
        finally:
            self.in_flight -= 1
            self._all.release()
            if held_batch:
                self._batch.release()


_gate = _Gate()


def slot(priority: Priority = "interactive"):
    """Hold a synthesis slot for the duration of the block, or raise Busy."""
    return _gate.slot(priority)


def stats() -> dict:
    return {
        "slots": _SLOTS,
        "batch_slots": _BATCH_SLOTS,
        "in_flight": _gate.in_flight,
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
