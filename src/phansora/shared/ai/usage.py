"""Per-trace token accounting.

You cannot minimise what you do not measure. Every research client reports what
each call actually cost here, and the orchestrator attaches the total to the
response — so a change that claims to save tokens can be checked instead of
believed, and the stage that dominates the bill is visible rather than guessed.

The meter is deliberately thread-local. ``TraceOrchestrator`` is a process-wide
singleton and ``JobManager`` runs traces concurrently on a ThreadPoolExecutor, so
one shared counter would blend two users' traces together. A job owns its worker
thread for its whole run, which makes thread-local state exactly per-trace.
"""
from __future__ import annotations

import threading
from typing import Any, Dict

_local = threading.local()

# Stage label applied to calls made before the orchestrator sets one.
_UNLABELLED = "other"


class _Meter:
    __slots__ = ("input_tokens", "output_tokens", "calls", "by_stage", "stage")

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.by_stage: Dict[str, Dict[str, int]] = {}
        self.stage = _UNLABELLED


def _meter() -> _Meter:
    m = getattr(_local, "meter", None)
    if m is None:
        m = _Meter()
        _local.meter = m
    return m


def start() -> None:
    """Begin a fresh measurement on this thread.

    Called at the top of every trace: worker threads are reused across jobs, so
    without this the second trace on a thread would inherit the first one's total.
    """
    _local.meter = _Meter()


def stage(name: str) -> None:
    """Label the calls that follow, so the report shows where the tokens went."""
    _meter().stage = name or _UNLABELLED


def record(*, input_tokens: int = 0, output_tokens: int = 0) -> None:
    """Record one completed LLM call. Never raises — accounting must not break a trace."""
    try:
        m = _meter()
        it, ot = int(input_tokens or 0), int(output_tokens or 0)
        m.input_tokens += it
        m.output_tokens += ot
        m.calls += 1
        bucket = m.by_stage.setdefault(m.stage, {"input_tokens": 0, "output_tokens": 0, "calls": 0})
        bucket["input_tokens"] += it
        bucket["output_tokens"] += ot
        bucket["calls"] += 1
    except Exception:  # pragma: no cover - a broken counter must stay invisible
        pass


def record_response(resp: Any) -> None:
    """Pull usage off a provider response, tolerating either SDK's field names.

    OpenAI Responses uses input/output_tokens; DeepSeek's chat completions use
    prompt/completion_tokens. A response without usage simply counts as a call.
    """
    usage = None
    if isinstance(resp, dict):
        usage = resp.get("usage")
    else:
        usage = getattr(resp, "usage", None)

    def field(*names: str) -> int:
        for n in names:
            if isinstance(usage, dict):
                val = usage.get(n)
            else:
                val = getattr(usage, n, None)
            if isinstance(val, (int, float)):
                return int(val)
        return 0

    if usage is None:
        record()
        return
    record(
        input_tokens=field("input_tokens", "prompt_tokens"),
        output_tokens=field("output_tokens", "completion_tokens"),
    )


def absorb(child: Dict[str, Any]) -> None:
    """Fold a worker thread's snapshot into this thread's total.

    Thread-local accounting is what keeps concurrent traces from contaminating
    each other, but it also means a call made on a worker thread lands in that
    worker's meter and not the trace's. Parallel stages therefore measure
    themselves and hand the result back here. Without this, running searches
    concurrently would silently report them as free.
    """
    if not isinstance(child, dict):
        return
    try:
        m = _meter()
        m.input_tokens += int(child.get("input_tokens") or 0)
        m.output_tokens += int(child.get("output_tokens") or 0)
        m.calls += int(child.get("llm_calls") or 0)
        for name, vals in (child.get("by_stage") or {}).items():
            bucket = m.by_stage.setdefault(name, {"input_tokens": 0, "output_tokens": 0, "calls": 0})
            bucket["input_tokens"] += int((vals or {}).get("input_tokens") or 0)
            bucket["output_tokens"] += int((vals or {}).get("output_tokens") or 0)
            bucket["calls"] += int((vals or {}).get("calls") or 0)
    except Exception:  # pragma: no cover
        pass


def snapshot() -> Dict[str, Any]:
    """The running total for this thread, shaped for the API response."""
    m = _meter()
    return {
        "input_tokens": m.input_tokens,
        "output_tokens": m.output_tokens,
        "total_tokens": m.input_tokens + m.output_tokens,
        "llm_calls": m.calls,
        "by_stage": {k: dict(v) for k, v in m.by_stage.items()},
    }
