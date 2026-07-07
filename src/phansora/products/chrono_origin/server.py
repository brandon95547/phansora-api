"""Chrono-Origin FastAPI entrypoint."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

import asyncio  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from .config import get_settings  # noqa: E402
from .models import ExpandRequest, ExpandResponse, TraceRequest, TraceResponse  # noqa: E402
from .pipeline.orchestrator import TraceOrchestrator  # noqa: E402
from .services.job_manager import JobManager  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
logger = logging.getLogger("chrono-origin")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if not settings.anthropic_api_key:
        logger.warning("ANTHROPIC_API_KEY is not set. /trace will fail until configured.")
    app.state.executor = ThreadPoolExecutor(max_workers=4)
    app.state.orchestrator = TraceOrchestrator()
    app.state.job_manager = JobManager(app.state.orchestrator, app.state.executor)
    yield
    app.state.executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="Chrono-Origin API",
    description="Trace the earliest known origin of a story, myth, or event using Claude grounded search.",
    version="0.2.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"name": "chrono-origin", "status": "ok", "version": "0.2.0"}


@app.get("/health")
def health():
    return {"status": "ok", "anthropic_configured": bool(get_settings().anthropic_api_key)}


def _ensure_configured() -> None:
    if not get_settings().anthropic_api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not configured.")


@app.post("/trace", response_model=TraceResponse)
async def trace(req: TraceRequest):
    """Synchronous trace - kept for backwards compatibility / quick CLI use."""
    _ensure_configured()
    loop = asyncio.get_running_loop()
    timeout = get_settings().chrono_request_timeout_s
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(app.state.executor, app.state.orchestrator.run, req),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Trace exceeded {timeout}s timeout.")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Trace failed")
        raise HTTPException(status_code=500, detail=f"Trace failed: {exc}")
    return result


# ---------------------------------------------------------------- async jobs
@app.post("/trace/jobs")
def submit_trace_job(req: TraceRequest):
    """Submit a trace as an async job. Returns a job id to poll."""
    _ensure_configured()
    job = app.state.job_manager.submit(req)
    return job.to_dict()


@app.get("/trace/jobs/{job_id}")
def get_trace_job(job_id: str):
    job = app.state.job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found or expired.")
    return job.to_dict()


# --------------------------------------------------------------------- expand
@app.post("/expand", response_model=ExpandResponse)
async def expand(req: ExpandRequest):
    """Expand a single timeline item into chronologically-ordered sub-events."""
    _ensure_configured()
    loop = asyncio.get_running_loop()
    timeout = get_settings().chrono_request_timeout_s
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(app.state.executor, app.state.orchestrator.expand, req),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail=f"Expand exceeded {timeout}s timeout.")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Expand failed")
        raise HTTPException(status_code=500, detail=f"Expand failed: {exc}")
    return result