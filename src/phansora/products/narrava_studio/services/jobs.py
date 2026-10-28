"""In-process job registry for work too slow to hold an HTTP request open.

Reading a document takes minutes: parse it, condense every chapter, then one pass to say
which ones would carry a video. No proxy waits for that, so the caller submits and polls.

Jobs live in this process's memory. That is a deliberate trade: a restart loses in-flight
readings (the user imports again, and Phansora refunds the credit when the poll comes back
404) but costs no schema, no worker and no broker. Phansora mirrors id, status and result
into Postgres, so nothing a user keeps depends on this dict.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger("narrava-studio.jobs")

JOB_TTL_SECONDS = 60 * 60  # long enough for a slow poller, or a page reloaded mid-reading


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | running | done | failed
    progress: int = 0
    stage: str = "Queued"
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "result": self.result,
            "error": self.error,
        }


class JobHandle:
    """What running work uses to say where it has got to."""

    def __init__(self, job: Job) -> None:
        self._job = job

    def progress(self, percent: int, stage: str) -> None:
        self._job.progress = max(0, min(100, int(percent)))
        self._job.stage = stage
        self._job.updated_at = time.time()


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._tasks: set[asyncio.Task] = set()

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def submit(self, work: Callable[[JobHandle], Awaitable[dict[str, Any]]]) -> Job:
        """Start ``work`` in the background; it gets a handle for progress and returns the
        result. A failure is the job's, never the process's: its message becomes the job's
        error, which is what the editor shows."""
        self._evict_stale()
        job = Job(id=uuid.uuid4().hex)
        self._jobs[job.id] = job

        async def run() -> None:
            job.status, job.stage, job.updated_at = "running", "Starting", time.time()
            try:
                job.result = await work(JobHandle(job))
                job.status, job.progress, job.stage = "done", 100, "Complete"
            except Exception as exc:  # noqa: BLE001
                logger.exception("Narrava job %s failed", job.id)
                job.status, job.error = "failed", str(exc) or "The document could not be read."
            finally:
                job.updated_at = time.time()

        # A strong reference: asyncio holds only a weak one, and an unreferenced task can be
        # collected mid-flight, silently cancelling the job.
        task = asyncio.create_task(run())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    def _evict_stale(self) -> None:
        cutoff = time.time() - JOB_TTL_SECONDS
        for jid in [j.id for j in self._jobs.values() if j.updated_at < cutoff]:
            self._jobs.pop(jid, None)


jobs = JobRegistry()
