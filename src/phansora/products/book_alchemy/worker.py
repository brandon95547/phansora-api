#!/usr/bin/env python3
"""Book Alchemy background worker.

A standalone, durable processor for long-running Book Alchemy jobs. It is
deliberately a SEPARATE process from the FastAPI app so that:

  * job processing survives independently of the `--workers N` API,
  * there is no in-memory job state (Postgres is the source of truth),
  * a crash mid-book is recovered automatically — another worker reclaims the
    project once its lease expires and resumes from the last committed phase.

Run:
    python -m phansora.products.book_alchemy.worker
Deploy as the systemd unit `book-alchemy-worker.service` (single instance to
start; the SKIP LOCKED claim design already allows scaling to N workers later).

Several users' books share this process. It runs CONCURRENCY lanes, and a lane
holds a book for one TURN (about TURN_SECONDS of work, always at least one step)
before handing it back; the next claim goes to whichever waiting book was served
longest ago (db.claim_next_project). So books take turns, roughly a lesson at a
time, instead of one book holding the worker for a whole ~2.5 h delivery phase
while someone who has just uploaded waits behind it with nothing to listen to.

The voice itself is still one GPU: two lanes recording at once queue at the API's
admission gate (spokenverse/admission.py) and are served one after the other.
What the lanes buy is that the rest of a book — reading the source, extracting
concepts, writing scripts, all DeepSeek and CPU — goes on while another book records.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import time
from pathlib import Path

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from phansora.products.book_alchemy import db, pipeline, storage  # noqa: E402
from phansora.products.book_alchemy.deepseek_client import DeepSeekClient  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
)
log = logging.getLogger("book_alchemy.worker")

WORKER_ID = f"{socket.gethostname()}:{os.getpid()}"
LEASE_SECONDS = int(os.getenv("BOOK_ALCHEMY_LEASE_SECONDS", "600"))
IDLE_SLEEP = float(os.getenv("BOOK_ALCHEMY_IDLE_SLEEP", "5"))
# How many times one delivery phase may fail before the whole course is failed.
PHASE_MAX_ATTEMPTS = int(os.getenv("BOOK_ALCHEMY_PHASE_MAX_ATTEMPTS", "2"))
# Books worked on at once. Three: one recording, the others writing, with room
# for a long OCR parse that would otherwise hold a lane for an hour.
CONCURRENCY = max(1, int(os.getenv("BOOK_ALCHEMY_CONCURRENCY", "3")))
# How long a lane stays on one book before handing it back. Measured in time rather
# than steps because steps are wildly uneven — one concept extraction is ~15 s, one
# recorded lesson ~5-8 min — and a turn of "one step" would let a book in its
# analyze phase move one chunk per lesson another book records. A step is never cut
# short, so a recording always completes its lesson.
TURN_SECONDS = float(os.getenv("BOOK_ALCHEMY_TURN_SECONDS", "300"))

_stop = asyncio.Event()


def _cleanup_ocr_cache(source_path: Path) -> None:
    """Drop the resume cache that a scanned PDF leaves behind.

    It only exists to survive an interrupted OCR run; once parsing is behind us it
    is dead weight — one small file per page of the book."""
    cache = pipeline.ocr_cache_dir(source_path)
    try:
        if not cache.is_dir() or cache.name != "ocr_cache":
            return
        for entry in cache.iterdir():
            if entry.is_file():
                entry.unlink()
        cache.rmdir()
    except OSError:
        log.warning("Could not remove OCR cache %s", cache, exc_info=True)


def _cleanup_source(proj: dict) -> None:
    """Delete the uploaded source file (PDF/EPUB/etc.) once parsing is behind us.

    The source is read by exactly one step, `_phase_parse`, whose idempotency
    guard is "do chunks already exist" rather than "is the file still there" — so
    once the cursor is past it, nothing ever opens the file again. Dropping it
    early matters now that a project sits parked between delivery phases for days
    at a time: there is no reason to hold a 40 MB PDF for the life of a
    twenty-hour course. The rendered session audio in the same folder is left
    untouched. Best-effort: never raises, and only ever unlinks a real file inside
    the book_alchemy dir."""
    source_path = proj.get("source_path")
    if not source_path:
        return
    try:
        path = Path(source_path).resolve()
        base = storage.BASE_DIR.resolve()
        if base in path.parents:
            _cleanup_ocr_cache(path)
        if base in path.parents and path.is_file():
            path.unlink()
            log.info("Deleted source file for project %s: %s", proj.get("id"), path.name)
    except Exception:  # noqa: BLE001 — cleanup must never break the worker
        log.warning("Could not delete source file %s", source_path, exc_info=True)


async def _fail(project_id: int, proj: dict, message: str) -> None:
    """Give up on the current work.

    A course delivered in phases should not be binned wholesale because one
    lesson tripped: if the listener already has hours of finished, playable audio,
    only the phase that broke is failed and the project parks so they can retry
    it. A phase that keeps breaking escalates to a project-level failure after
    PHASE_MAX_ATTEMPTS, which is also what restores the credit-refund path in the
    dashboard's reconciler.
    """
    detail = message[:1000]
    active = await db.active_phase(project_id)
    delivered = [p for p in await db.get_phases(project_id) if p["status"] == "complete"]

    if active is not None and delivered:
        attempts = int(active["attempts"]) + 1
        if attempts < PHASE_MAX_ATTEMPTS:
            log.warning(
                "Project %s phase %s failed (attempt %s/%s): %s",
                project_id, active["ordinal"], attempts, PHASE_MAX_ATTEMPTS, message,
            )
            await db.set_phase(
                active["id"], status="failed", attempts=attempts, error_message=detail,
            )
            await db.set_project(
                project_id, status="awaiting_user", phase="awaiting_user",
                stage=f"Phase {active['ordinal']} could not be built",
            )
            return
        log.warning(
            "Project %s phase %s failed %s times; failing the project",
            project_id, active["ordinal"], attempts,
        )
        await db.set_phase(
            active["id"], status="failed", attempts=attempts, error_message=detail,
        )

    await db.set_project(
        project_id, status="failed", phase="failed", stage="Failed", error_message=detail,
    )
    # The upload survives a failure that happened before parsing stored anything.
    # A `failed` row is not really terminal — it is resumed by flipping status and
    # phase back — but only while the file it was made from still exists, and a
    # project that trips in its first phase is failed outright with no retry. That
    # combination is what turned a one-line NUL-byte bug into a lost book and a
    # re-upload. Once chunks exist, nothing opens the source again: delete it then.
    if await db.count_chunks(project_id) > 0:
        _cleanup_source(proj)


async def _heartbeat(project_id: int, owner: str) -> None:
    """Hold the lease while a single step runs.

    Renewing only BETWEEN steps was fine when every step was bounded in minutes,
    but parsing a scanned book is hours of work inside one step: the lease expired
    while the worker was still busy, leaving the row claimable by anyone. With one
    worker that was invisible; with two it is the same book read twice.
    """
    interval = max(30, LEASE_SECONDS // 3)
    while True:
        await asyncio.sleep(interval)
        try:
            await db.renew_lease(project_id, owner, LEASE_SECONDS)
        except Exception:  # noqa: BLE001 — a DB blip must not end the run
            log.warning("Lease renewal failed for project %s", project_id, exc_info=True)


async def _take_turn(project_id: int, client: DeepSeekClient, owner: str) -> int:
    """Drive one claimed project for a turn: steps until TURN_SECONDS is used up or
    the book has nothing more to do — done, failed, or parked waiting for the
    listener to ask for the next delivery phase. Returns the steps taken.

    Ending the turn does not end the book. The caller releases the lease, and the
    book is claimed again (by any lane) once the books waiting longer have had
    theirs; every phase is cursor-driven and resumes from what is in the database.
    """
    started = time.monotonic()
    steps = 0
    while not _stop.is_set():
        row = await db.get_project(project_id)
        if row is None:
            return steps
        proj = dict(row)
        if proj["phase"] in pipeline.IDLE_PHASES:
            # Done, failed, or parked between phases — all finished with the source
            # file, except the failure that never got as far as a chunk. See _fail.
            if proj["phase"] != "failed" or await db.count_chunks(project_id) > 0:
                _cleanup_source(proj)
            return steps
        if steps and time.monotonic() - started >= TURN_SECONDS:
            return steps
        if proj["phase"] not in ("uploaded", "parse"):
            # Idempotent and cheap once the file is gone, so it can run every turn
            # rather than needing to remember across them.
            _cleanup_source(proj)
        beat = asyncio.ensure_future(_heartbeat(project_id, owner))
        try:
            await pipeline.run_step(proj, client)
            steps += 1
        except pipeline.RetryableError as exc:
            # Interrupted, not broken. Leave status 'processing' and let the lease
            # drop: the next worker claims the row and resumes from what's on disk.
            log.warning("Project %s interrupted, will resume: %s", project_id, exc)
            await db.set_project(project_id, stage="Paused — resuming shortly")
            return steps
        except pipeline.TerminalError as exc:
            log.warning("Project %s failed (terminal): %s", project_id, exc)
            await _fail(project_id, proj, str(exc))
            return steps
        except Exception as exc:  # noqa: BLE001
            # Same reasoning as RetryableError, for the phases that don't name it
            # themselves: audio rendering loses its ffmpeg and its HTTP call to the
            # API in the same instant a restart arrives.
            if _stop.is_set() or pipeline.shutting_down():
                log.warning("Project %s interrupted by shutdown, will resume: %s", project_id, exc)
                await db.set_project(project_id, stage="Paused — resuming shortly")
                return steps
            log.exception("Project %s step crashed", project_id)
            # str() of some exceptions is empty; a blank error is what the dashboard
            # and the admin page would then show for the book.
            await _fail(project_id, proj, str(exc) or type(exc).__name__)
            return steps
        finally:
            beat.cancel()
        await db.renew_lease(project_id, owner, LEASE_SECONDS)
    return steps


async def _lane(lane: int, client: DeepSeekClient) -> None:
    """One of CONCURRENCY loops: claim the book whose turn it is, work it for a turn,
    hand it back."""
    # Distinct per lane so the lease column says which loop holds a book.
    owner = f"{WORKER_ID}/{lane}"
    while not _stop.is_set():
        try:
            row = await db.claim_next_project(owner, LEASE_SECONDS)
        except Exception:  # noqa: BLE001
            log.exception("claim failed; backing off")
            await _sleep_or_stop(IDLE_SLEEP)
            continue

        if row is None:
            await _sleep_or_stop(IDLE_SLEEP)
            continue

        pid = int(row["id"])
        log.info("Lane %s claimed project %s (phase=%s)", lane, pid, row["phase"])
        steps = 0
        try:
            steps = await _take_turn(pid, client, owner)
        except Exception:  # noqa: BLE001 — one book's bad turn must not end the lane
            log.exception("Lane %s: turn on project %s failed outside a step", lane, pid)
            await _sleep_or_stop(IDLE_SLEEP)
        finally:
            await db.release_lease(pid)
            log.info("Lane %s released project %s after %s step(s)", lane, pid, steps)


async def main() -> None:
    log.info(
        "Book Alchemy worker starting (id=%s, lease=%ss, lanes=%s, turn=%ss)",
        WORKER_ID, LEASE_SECONDS, CONCURRENCY, TURN_SECONDS,
    )
    client = DeepSeekClient.from_env()
    await db.get_pool()  # fail fast if DB/env is misconfigured

    await asyncio.gather(*(_lane(n, client) for n in range(1, CONCURRENCY + 1)))

    await db.close_pool()
    log.info("Book Alchemy worker stopped.")


async def _sleep_or_stop(seconds: float) -> None:
    try:
        await asyncio.wait_for(_stop.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass


def _on_signal() -> None:
    """Stop the loop AND tell the pipeline this is a shutdown.

    systemd kills the whole control group, so a step's child processes (Tesseract,
    ffmpeg) die at the same moment the worker is asked to stop. Without this flag
    the pipeline sees only "the child died" and fails a perfectly good book.
    """
    _stop.set()
    pipeline.signal_shutdown()


def _install_signals(loop: asyncio.AbstractEventLoop) -> None:
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _on_signal)
        except NotImplementedError:  # pragma: no cover (non-unix)
            pass


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _install_signals(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
