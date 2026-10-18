from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from phansora.shared.auth import enforce_user_scope
from pydantic import BaseModel, Field, model_validator

from phansora.shared.errors import fail
from phansora.shared.paths import runtime_root
from .pipeline import run_pipeline

logger = logging.getLogger("research_atlas")


BASE_DIR = runtime_root()
TMP_DIR = BASE_DIR / "tmp" / "research_atlas"
TMP_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="research-atlas-api",
    version="2.0.0",
    dependencies=[Depends(enforce_user_scope)],
)


_JOBS: dict[str, dict] = {}
_JOBS_LOCK = asyncio.Lock()
_MAX_COMPLETED_JOBS = 400

# The running pipelines. The event loop keeps only a weak reference to a task, so one
# nothing else holds can be collected mid-run; this set is that other reference, and each
# task takes itself out when it ends.
_TASKS: set[asyncio.Task] = set()

# How many pipelines run at once. Each one keeps up to LLM_MAX_WORKERS calls in flight and
# holds its sources in memory, and the Node worker sends one job at a time by default; jobs
# past this wait as "queued", which the worker already polls through.
_MAX_RUNNING = max(1, int(os.getenv("RESEARCH_ATLAS_MAX_CONCURRENT_RUNS", "2")))
_RUN_SLOTS = asyncio.Semaphore(_MAX_RUNNING)
# Jobs queued or running before /run starts saying "busy". Far past what the worker sends;
# it bounds what a caller who keeps posting can make this process hold.
_MAX_OUTSTANDING = 20

# Request size caps. Generous: a source is the extracted text of a file the Node app
# accepts at up to 250 MB, and a long book or a transcribed recording runs to millions of
# characters. They exist so one request cannot hold an unbounded amount of memory, not to
# trim real research. The worker sends at most 15 sources (5 files and 10 articles).
MAX_SOURCES = 50
MAX_TEXT_CHARS = 20_000_000
MAX_TOTAL_CHARS = 40_000_000
MAX_LABEL_CHARS = 1_000


class SourceItem(BaseModel):
    """A single source document with label and text."""
    label: str = Field(max_length=MAX_LABEL_CHARS)
    text: str = Field(max_length=MAX_TEXT_CHARS)


class RunPipelineRequest(BaseModel):
    """What a caller may ask for: CONTENT, never a location on this server's disk.

    `input_path` and `toc_full_path` used to be accepted here and handed straight to
    `Path(...).read_text()`, with the TOC read back into the job result — so a request
    naming `/var/www/phansora-api/.env` returned its contents to the caller. Neither
    field was ever sent by the Node worker; both are gone, and the pipeline's working
    files are now always temp paths this module chooses (see _run_pipeline_job).
    """

    text: str | None = Field(default=None, max_length=MAX_TEXT_CHARS)
    max_chunk_chars: int | None = None
    sources: list[SourceItem] | None = Field(default=None, max_length=MAX_SOURCES)
    # The report prints this as its H1, so the caller's own name for the job ends up on
    # the cover. Optional: run_pipeline falls back to a generic title.
    title: str | None = Field(default=None, max_length=MAX_LABEL_CHARS)

    @model_validator(mode="after")
    def _within_total(self) -> "RunPipelineRequest":
        total = len(self.text or "") + sum(len(s.text) for s in self.sources or [])
        if total > MAX_TOTAL_CHARS:
            raise ValueError(f"The sources add up to more than {MAX_TOTAL_CHARS:,} characters.")
        return self


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _prune_finished_jobs() -> None:
    async with _JOBS_LOCK:
        finished = [
            (job_id, data)
            for job_id, data in _JOBS.items()
            if data.get("status") in {"completed", "failed"}
        ]
        if len(finished) <= _MAX_COMPLETED_JOBS:
            return

        finished.sort(key=lambda item: item[1].get("finished_at") or item[1].get("updated_at") or "")
        to_remove = len(finished) - _MAX_COMPLETED_JOBS
        for job_id, _ in finished[:to_remove]:
            _JOBS.pop(job_id, None)


async def _set_job_state(job_id: str, **kwargs) -> None:
    async with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        job.update(kwargs)
        job["updated_at"] = _utc_now_iso()


async def _run_pipeline_job(job_id: str, payload: RunPipelineRequest) -> None:
    async with _RUN_SLOTS:
        await _run_pipeline_job_now(job_id, payload)


async def _run_pipeline_job_now(job_id: str, payload: RunPipelineRequest) -> None:
    temp_input_path: Path | None = None
    temp_toc_path: Path | None = None
    # Both start unset: the caller no longer supplies paths, so every file this job
    # touches is one this module created under TMP_DIR.
    run_input_path = None
    run_toc_path = None
    run_sources = None

    await _set_job_state(job_id, status="processing", started_at=_utc_now_iso(), error=None)

    try:
        # If sources array is provided, convert to pipeline-ready format
        if payload.sources:
            run_sources = [
                {"label": s.label, "text": s.text}
                for s in payload.sources
                if s.text and s.text.strip()
            ]
            if not run_sources:
                raise ValueError("All provided sources are empty.")
        elif payload.text is not None:
            temp_input_path = TMP_DIR / f"api_input_{uuid.uuid4().hex}.txt"
            temp_input_path.write_text(payload.text, encoding="utf-8")
            run_input_path = str(temp_input_path)

        if not run_toc_path:
            temp_toc_path = TMP_DIR / f"api_toc_{uuid.uuid4().hex}.md"
            run_toc_path = str(temp_toc_path)

        kwargs = {}
        if (payload.title or "").strip():
            kwargs["title"] = payload.title.strip()
        result = await asyncio.to_thread(
            run_pipeline,
            input_path=run_input_path,
            toc_full_path=run_toc_path,
            max_chunk_chars=payload.max_chunk_chars,
            sources=run_sources,
            **kwargs,
        )

        toc_markdown = ""
        loss_report_markdown = ""
        resolved_toc_path = result.get("toc_full_path") or run_toc_path
        if resolved_toc_path:
            toc_path = Path(resolved_toc_path)
            if toc_path.exists():
                toc_markdown = toc_path.read_text(encoding="utf-8")
            loss_report_path = toc_path.parent / "loss_report.md"
            if loss_report_path.exists():
                loss_report_markdown = loss_report_path.read_text(encoding="utf-8")

        await _set_job_state(
            job_id,
            status="completed",
            finished_at=_utc_now_iso(),
            result={
                **result,
                "toc_markdown": toc_markdown,
                "loss_report_markdown": loss_report_markdown,
            },
        )
    except Exception as e:
        # The exception can carry a provider's response body or a path on this server;
        # the job's error is read back by the caller, so it gets a sentence and a
        # reference, and the log gets the traceback under the same reference.
        await _set_job_state(
            job_id,
            status="failed",
            finished_at=_utc_now_iso(),
            error=fail(500, "The research pipeline failed.", e, logger=logger,
                       context=f"Research Atlas job {job_id} failed").detail,
        )
    finally:
        if temp_input_path is not None:
            try:
                temp_input_path.unlink(missing_ok=True)
            except Exception:
                pass
        if temp_toc_path is not None:
            try:
                temp_toc_path.unlink(missing_ok=True)
            except Exception:
                pass
            # Also clean up the loss report written next to the temp TOC
            try:
                temp_loss_report = temp_toc_path.parent / "loss_report.md"
                temp_loss_report.unlink(missing_ok=True)
            except Exception:
                pass
        await _prune_finished_jobs()


@app.get("/health")
async def health() -> dict:
    async with _JOBS_LOCK:
        active_jobs = sum(1 for data in _JOBS.values() if data.get("status") in {"queued", "processing"})
    return {"ok": True, "active_jobs": active_jobs}


@app.post("/run")
async def run_pipeline_endpoint(payload: RunPipelineRequest) -> dict:
    if not payload.text and not payload.sources:
        raise HTTPException(status_code=400, detail="Provide `text` or `sources`.")

    job_id = uuid.uuid4().hex
    now = _utc_now_iso()

    async with _JOBS_LOCK:
        outstanding = sum(1 for data in _JOBS.values() if data.get("status") in {"queued", "processing"})
        if outstanding >= _MAX_OUTSTANDING:
            raise HTTPException(
                status_code=503,
                detail="Too many research jobs are already waiting. Try again shortly.",
                headers={"Retry-After": "60"},
            )
        _JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "result": None,
        }

    task = asyncio.create_task(_run_pipeline_job(job_id, payload))
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return {
        "ok": True,
        "job_id": job_id,
        "status": "queued",
    }


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str) -> dict:
    async with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")

        return {
            "ok": True,
            "job_id": job["job_id"],
            "status": job["status"],
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
            "started_at": job["started_at"],
            "finished_at": job["finished_at"],
            "error": job["error"],
        }


@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str) -> dict:
    async with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")

        status = job.get("status")
        if status == "failed":
            raise HTTPException(status_code=409, detail=job.get("error") or "Job failed.")
        if status != "completed":
            raise HTTPException(status_code=409, detail=f"Job is still {status}.")

        return {
            "ok": True,
            "job_id": job_id,
            "result": job.get("result") or {},
        }
