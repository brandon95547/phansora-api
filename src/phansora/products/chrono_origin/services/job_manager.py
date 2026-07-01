"""In-memory async job manager for /trace runs.

Jobs are kept in process memory, keyed by UUID. Old jobs are evicted after
``JOB_TTL_SECONDS`` so the dict can't grow without bound.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import Executor
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

from ..models import TraceRequest
from ..pipeline.orchestrator import TraceOrchestrator

logger = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "done", "failed"]

# Keep finished jobs around so that slow front-end pollers / page refreshes
# can still pick up the result.
JOB_TTL_SECONDS = 60 * 60  # 1 hour


@dataclass
class Job:
    id: str
    status: JobStatus = "queued"
    progress: int = 0
    stage: str = "Queued"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.id,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class JobManager:
    def __init__(self, orchestrator: TraceOrchestrator, executor: Executor) -> None:
        self._orchestrator = orchestrator
        self._executor = executor
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    # ----------------------------------------------------------- public API
    def submit(self, req: TraceRequest) -> Job:
        self._evict_stale()
        job = Job(id=uuid.uuid4().hex)
        with self._lock:
            self._jobs[job.id] = job
        # Run in the shared executor so we don't block the event loop.
        self._executor.submit(self._run_job, job.id, req)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    # ------------------------------------------------------------- internal
    def _run_job(self, job_id: str, req: TraceRequest) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "running"
            job.started_at = time.time()
            job.stage = "Starting"
            job.updated_at = time.time()

        def on_progress(percent: int, stage: str) -> None:
            with self._lock:
                j = self._jobs.get(job_id)
                if j is None or j.status in ("done", "failed"):
                    return
                # Progress can never go backwards.
                j.progress = max(j.progress, max(0, min(99, int(percent))))
                if stage:
                    j.stage = stage
                j.updated_at = time.time()

        try:
            response = self._orchestrator.run(req, on_progress=on_progress)
            payload = response.model_dump()
            with self._lock:
                j = self._jobs.get(job_id)
                if j is None:
                    return
                j.status = "done"
                j.progress = 100
                j.stage = "Done"
                j.result = payload
                j.finished_at = time.time()
                j.updated_at = j.finished_at
        except Exception as exc:  # noqa: BLE001 - surface any failure
            logger.exception("Job %s failed", job_id)
            with self._lock:
                j = self._jobs.get(job_id)
                if j is None:
                    return
                j.status = "failed"
                j.error = str(exc) or exc.__class__.__name__
                j.finished_at = time.time()
                j.updated_at = j.finished_at

    def _evict_stale(self) -> None:
        cutoff = time.time() - JOB_TTL_SECONDS
        with self._lock:
            for jid in list(self._jobs.keys()):
                j = self._jobs[jid]
                if j.status in ("done", "failed") and (j.finished_at or j.updated_at) < cutoff:
                    self._jobs.pop(jid, None)
