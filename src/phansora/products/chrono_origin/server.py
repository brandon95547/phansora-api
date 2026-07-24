"""Chrono-Origin FastAPI entrypoint."""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

import asyncio  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from .config import get_settings  # noqa: E402
from .models import CacheKeyRequest, ExpandRequest, ExpandResponse, TraceRequest, TraceResponse  # noqa: E402
from .pipeline.orchestrator import TraceOrchestrator  # noqa: E402
from .services.cache import delete_cached, normalize_title  # noqa: E402
from .services.job_manager import JobManager  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s")
logger = logging.getLogger("chrono-origin")


@asynccontextmanager
async def lifespan(app: FastAPI):
    provider = _provider()
    key = "DEEPSEEK_API_KEY" if provider == "deepseek" else "OPENAI_API_KEY"
    if not os.getenv(key):
        logger.warning("%s is not set. /trace will fail until configured.", key)
    logger.info("Chrono-Origin LLM provider: %s", provider)
    if provider == "deepseek":
        logger.info("DeepSeek external search: %s", os.getenv("CHRONO_SEARCH_PROVIDER", "auto"))
    app.state.executor = ThreadPoolExecutor(max_workers=4)
    app.state.orchestrator = TraceOrchestrator()
    app.state.job_manager = JobManager(app.state.orchestrator, app.state.executor)
    yield
    app.state.executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="Chrono-Origin API",
    description="Trace the earliest known origin of a story, myth, or event using grounded web search.",
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


def _provider() -> str:
    return os.getenv("CHRONO_LLM_PROVIDER", "openai").strip().lower()


def _provider_configured() -> bool:
    """Is the *active* LLM provider configured? (OpenAI/GPT-5 Nano by default.)"""
    if _provider() == "deepseek":
        return bool(os.getenv("DEEPSEEK_API_KEY"))
    return bool(os.getenv("OPENAI_API_KEY"))


@app.get("/health")
def health():
    return {"status": "ok", "provider": _provider(), "configured": _provider_configured()}


def _ensure_configured() -> None:
    if not _provider_configured():
        key = "DEEPSEEK_API_KEY" if _provider() == "deepseek" else "OPENAI_API_KEY"
        raise HTTPException(status_code=503, detail=f"{key} is not configured.")


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


@app.post("/cache/invalidate")
def invalidate_cache(req: CacheKeyRequest):
    """Drop the cached trace for a title so a re-trace runs fresh. Called by the
    Node app when a user deletes an origin trace. Idempotent."""
    removed = delete_cached(normalize_title(req.title))
    return {"ok": True, "removed": removed}


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