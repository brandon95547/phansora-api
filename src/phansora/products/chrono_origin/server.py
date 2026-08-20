"""Chrono-Origin FastAPI entrypoint."""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

import asyncio  # noqa: E402

from fastapi import Depends, FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from phansora.shared.auth import enforce_user_scope
from phansora.shared.errors import fail

from .config import get_settings  # noqa: E402
from .models import CacheKeyRequest, ExpandRequest, ExpandResponse, TraceRequest, TraceResponse  # noqa: E402
from .pipeline.orchestrator import TraceOrchestrator  # noqa: E402
from .services.cache import delete_cached  # noqa: E402
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
    # Four was enough when nothing leaked. It is not a safety limit — the work is
    # network-bound, so idle threads cost almost nothing — and at four, a handful of
    # slow requests took the whole product down for every user in the process.
    app.state.executor = ThreadPoolExecutor(max_workers=16, thread_name_prefix="chrono")
    app.state.orchestrator = TraceOrchestrator()
    app.state.job_manager = JobManager(app.state.orchestrator, app.state.executor)
    yield
    app.state.executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="Chrono-Origin API",
    description="Trace the earliest known origin of a story, myth, or event using grounded web search.",
    version="0.2.0",
    lifespan=lifespan,
    dependencies=[Depends(enforce_user_scope)],
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
        # RuntimeError reaches here from the LLM client, whose message embeds the
        # provider's raw response body — not something to hand back to a browser.
        raise fail(503, "The tracing service is unavailable.", exc, logger=logger,
                   context="Chrono upstream unavailable")
    except Exception as exc:
        logger.exception("Trace failed")
        raise fail(500, "The trace failed.", exc, logger=logger, context="Trace failed")
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
    """Drop every cached variant of a title so a re-trace runs fresh. Called by the
    Node app when a user deletes an origin trace. Idempotent.

    delete_cached takes the raw title and normalises internally — it has to, now that one
    title can have several cached entries (one per context/depth/sources combination) and
    the caller knows only the title. `removed` is a count rather than a bool for the same
    reason.
    """
    removed = delete_cached(req.title)
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
        # The thread is still running and still holds a worker. Logged loudly because
        # this is the shape of a pool leak, and it was invisible until now.
        logger.warning(
            "Expand exceeded %ss; its worker is still occupied until the work finishes.",
            timeout,
        )
        raise HTTPException(status_code=504, detail=f"Expand exceeded {timeout}s timeout.")
    except RuntimeError as exc:
        # RuntimeError reaches here from the LLM client, whose message embeds the
        # provider's raw response body — not something to hand back to a browser.
        raise fail(503, "The tracing service is unavailable.", exc, logger=logger,
                   context="Chrono upstream unavailable")
    except Exception as exc:
        logger.exception("Expand failed")
        raise fail(500, "The expand failed.", exc, logger=logger, context="Expand failed")
    return result