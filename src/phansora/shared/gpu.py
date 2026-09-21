"""One GPU, several tenants: who may touch CUDA while the TTS engine is loading.

The box has a single A4000 and the API runs a single worker, so the voice engine (vLLM)
and the caption machinery (faster-whisper on ``WHISPER_DEVICE=cuda``, the forced aligner)
share one CUDA context inside one process. That is fine almost all of the time — the GPU
is happy to timeshare, and neither side is big enough to starve the other.

It is not fine for the ~90 seconds at engine load when **vLLM captures CUDA graphs**. A
capture is a recording of the context, and CUDA work issued from any other thread while
one is underway invalidates it. What makes that worth a module is not the failed load: it
is that PyTorch's caching allocator never clears ``captures_underway`` afterwards, so
*every subsequent allocation in that process* dies on

    captures_underway.empty() INTERNAL ASSERT FAILED at CUDACachingAllocator.cpp:3089

The context cannot be repaired from inside. Only a restart clears it.

MEASURED on prod, 2026-09-20/21: of eight restarts, the two that took a
``/studio/lyrics/*`` request inside the capture window both died with "CUDA error:
operation failed due to a previous error during capture", and the second one left
narration returning 500 for fifteen hours before anyone hit it.

So the engine load takes the device exclusively, and everything else that touches CUDA
goes through ``guest()`` and waits the capture out. Outside a load the gate is
uncontended: guests do not exclude each other, because sharing the GPU is only illegal
while a capture is in flight. A guest already running when a load wants to start is
waited for rather than interrupted — hence the writer-priority arrangement below, which
also keeps a steady trickle of caption requests from starving the load forever.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from threading import Condition
from typing import Iterator

logger = logging.getLogger("phansora.gpu")

_COND = Condition()
_guests = 0        # GPU operations in flight that are not a capture
_exclusive = False  # a load/capture owns the device
_waiting = 0       # loads queued, so new guests hold back and let them through


def _timeout(name: str, default: float) -> float:
    try:
        raw = os.getenv(name, "").strip()
        return float(raw) if raw else default
    except ValueError:
        return default


class GpuBusy(RuntimeError):
    """The device did not come free in time. Nothing was run, so nothing is corrupted."""


@contextmanager
def guest(what: str, *, active: bool = True) -> Iterator[None]:
    """Run GPU work that must not overlap an engine load.

    ``active=False`` makes this a no-op, for a caller that has been pointed at the CPU —
    it holds no GPU and must not make a load wait for it. Waits out a load in progress;
    a load that never finishes raises ``GpuBusy`` rather than proceeding into a capture,
    because proceeding is the thing that breaks the process.
    """
    global _guests
    if not active:
        yield
        return
    wait = _timeout("PHANSORA_GPU_GUEST_WAIT_SEC", 300.0)
    with _COND:
        if not _COND.wait_for(lambda: not _exclusive and _waiting == 0, timeout=wait):
            raise GpuBusy(
                f"The GPU is still loading the voice engine, so {what} could not start. "
                "Try again in a minute."
            )
        _guests += 1
    try:
        yield
    finally:
        with _COND:
            _guests -= 1
            _COND.notify_all()


@contextmanager
def exclusive(what: str, *, active: bool = True) -> Iterator[None]:
    """Own the device outright — for a CUDA-graph capture and nothing else.

    Raises ``GpuBusy`` if the guests in flight do not finish in time. The caller's load
    then has not started, which is the whole point: a load that begins while another
    thread is on the device is worse than a load that has not begun, and the next request
    can try again against an idle device.
    """
    global _exclusive, _waiting
    if not active:
        yield
        return
    wait = _timeout("PHANSORA_GPU_EXCLUSIVE_WAIT_SEC", 180.0)
    with _COND:
        _waiting += 1
        try:
            free = _COND.wait_for(lambda: _guests == 0 and not _exclusive, timeout=wait)
        finally:
            _waiting -= 1
            _COND.notify_all()
        if not free:
            raise GpuBusy(
                f"{what} could not get the GPU to itself within {wait:.0f}s "
                f"({_guests} operation(s) still running)."
            )
        _exclusive = True
        logger.info("GPU held exclusively for %s", what)
    try:
        yield
    finally:
        with _COND:
            _exclusive = False
            _COND.notify_all()
        logger.info("GPU released after %s", what)


def cuda_available() -> bool:
    """Whether torch can see a GPU at all. False on a host with no torch installed."""
    try:
        import torch  # type: ignore
    except Exception:  # noqa: BLE001
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False
