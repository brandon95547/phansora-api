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
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
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

_stop = asyncio.Event()


def _cleanup_source(proj: dict) -> None:
    """Delete the uploaded source file (PDF/EPUB/etc.) once a project reaches a
    terminal phase — whether the audio course succeeded or failed. The source is
    only needed while parsing; afterwards it just consumes disk and accumulates.
    The rendered session audio in the same folder is left untouched. Best-effort:
    never raises, and only ever unlinks a real file inside the book_alchemy dir."""
    source_path = proj.get("source_path")
    if not source_path:
        return
    try:
        path = Path(source_path).resolve()
        base = storage.BASE_DIR.resolve()
        if base in path.parents and path.is_file():
            path.unlink()
            log.info("Deleted source file for project %s: %s", proj.get("id"), path.name)
    except Exception:  # noqa: BLE001 — cleanup must never break the worker
        log.warning("Could not delete source file %s", source_path, exc_info=True)


async def _process_project(project_id: int, client: DeepSeekClient) -> None:
    """Drive one claimed project to a terminal phase, renewing its lease."""
    while not _stop.is_set():
        row = await db.get_project(project_id)
        if row is None:
            return
        proj = dict(row)
        if proj["phase"] in ("complete", "failed"):
            # Reached a terminal phase (course done or failed) — drop the source file.
            _cleanup_source(proj)
            return
        try:
            await pipeline.run_step(proj, client)
        except pipeline.TerminalError as exc:
            log.warning("Project %s failed (terminal): %s", project_id, exc)
            await db.set_project(
                project_id, status="failed", phase="failed",
                stage="Failed", error_message=str(exc)[:1000],
            )
            _cleanup_source(proj)
            return
        except Exception as exc:  # noqa: BLE001
            log.exception("Project %s step crashed", project_id)
            await db.set_project(
                project_id, status="failed", phase="failed",
                stage="Failed", error_message=str(exc)[:1000],
            )
            _cleanup_source(proj)
            return
        await db.renew_lease(project_id, WORKER_ID, LEASE_SECONDS)


async def main() -> None:
    log.info("Book Alchemy worker starting (id=%s, lease=%ss)", WORKER_ID, LEASE_SECONDS)
    client = DeepSeekClient.from_env()
    await db.get_pool()  # fail fast if DB/env is misconfigured

    while not _stop.is_set():
        try:
            row = await db.claim_next_project(WORKER_ID, LEASE_SECONDS)
        except Exception:  # noqa: BLE001
            log.exception("claim failed; backing off")
            await _sleep_or_stop(IDLE_SLEEP)
            continue

        if row is None:
            await _sleep_or_stop(IDLE_SLEEP)
            continue

        pid = int(row["id"])
        log.info("Claimed project %s (phase=%s)", pid, row["phase"])
        try:
            await _process_project(pid, client)
        finally:
            await db.release_lease(pid)
            log.info("Released project %s", pid)

    await db.close_pool()
    log.info("Book Alchemy worker stopped.")


async def _sleep_or_stop(seconds: float) -> None:
    try:
        await asyncio.wait_for(_stop.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass


def _install_signals(loop: asyncio.AbstractEventLoop) -> None:
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _stop.set)
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
