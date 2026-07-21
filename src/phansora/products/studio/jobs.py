"""In-process job registry for work too slow to hold an HTTP request open.

Reading a whole book takes minutes: parse, condense every chapter, then one reasoning pass.
No proxy will wait for that, so the caller submits and polls — the same shape Chrono Origin
uses (``services/job_manager.py``), simplified because this work is already async I/O and
needs no thread pool.

Jobs live in this process's memory. That is a deliberate trade: it means a restart loses
in-flight analyses (the user re-uploads) but costs no schema, no worker and no broker. The
Node side mirrors id + status into Postgres, so nothing a user *keeps* depends on this dict.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger("phansora.studio.jobs")

JOB_TTL_SECONDS = 60 * 60  # keep finished jobs long enough for a slow poller or a refresh


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
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._tasks: set[asyncio.Task] = set()

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def submit(self, work: Callable[["JobHandle"], Awaitable[dict[str, Any]]]) -> Job:
        """Start `work` in the background. It receives a handle for progress reporting and
        returns the result dict."""
        self._evict_stale()
        job = Job(id=uuid.uuid4().hex)
        self._jobs[job.id] = job

        async def run() -> None:
            job.status, job.stage, job.updated_at = "running", "Starting", time.time()
            try:
                job.result = await work(JobHandle(job))
                job.status, job.progress, job.stage = "done", 100, "Complete"
            except Exception as exc:  # noqa: BLE001 — the failure belongs to the job, not the process
                logger.exception("Studio job %s failed", job.id)
                job.status, job.error = "failed", str(exc)
            finally:
                job.updated_at = time.time()

        # Keep a strong reference: asyncio only holds a weak one, so an unreferenced task
        # can be garbage-collected mid-flight and silently cancel the job.
        task = asyncio.create_task(run())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    def _evict_stale(self) -> None:
        cutoff = time.time() - JOB_TTL_SECONDS
        for jid in [j.id for j in self._jobs.values() if j.updated_at < cutoff]:
            self._jobs.pop(jid, None)


class JobHandle:
    """What running work uses to report where it has got to."""

    def __init__(self, job: Job) -> None:
        self._job = job

    def progress(self, percent: int, stage: str) -> None:
        self._job.progress = max(0, min(100, int(percent)))
        self._job.stage = stage
        self._job.updated_at = time.time()
