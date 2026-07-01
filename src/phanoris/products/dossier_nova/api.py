from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from phanoris.shared.paths import runtime_root
from .main import run_pipeline


BASE_DIR = runtime_root()
TMP_DIR = BASE_DIR / "tmp" / "dossier_nova"
TMP_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="tomeweaver-api", version="1.0.0")


_JOBS: dict[str, dict] = {}
_JOBS_LOCK = asyncio.Lock()
_MAX_COMPLETED_JOBS = 400


class SourceItem(BaseModel):
    """A single source document with label and text."""
    label: str
    text: str


class RunPipelineRequest(BaseModel):
    text: str | None = None
    input_path: str | None = None
    toc_full_path: str | None = None
    max_chunk_chars: int | None = None
    sources: list[SourceItem] | None = None


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
    temp_input_path: Path | None = None
    temp_toc_path: Path | None = None
    run_input_path = payload.input_path
    run_toc_path = payload.toc_full_path
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

        result = await asyncio.to_thread(
            run_pipeline,
            input_path=run_input_path,
            toc_full_path=run_toc_path,
            max_chunk_chars=payload.max_chunk_chars,
            sources=run_sources,
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
        await _set_job_state(
            job_id,
            status="failed",
            finished_at=_utc_now_iso(),
            error=f"Pipeline failed: {e}",
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
    if not payload.text and not payload.input_path and not payload.sources:
        raise HTTPException(status_code=400, detail="Provide `text`, `input_path`, or `sources`.")

    job_id = uuid.uuid4().hex
    now = _utc_now_iso()

    async with _JOBS_LOCK:
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

    asyncio.create_task(_run_pipeline_job(job_id, payload))
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
