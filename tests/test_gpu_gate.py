"""The gate that keeps caption work out of the voice engine's CUDA-graph capture.

The bug being pinned down: the box has one GPU and the API one process, so whisper and the
forced aligner share a CUDA context with vLLM. vLLM captures CUDA graphs for ~90s when it
loads, any CUDA work from another thread during that capture invalidates it, and PyTorch's
allocator never recovers afterwards — so a single caption request landing in that window
leaves the process unable to synthesize until somebody restarts it. It happened twice on
prod on 2026-09-20/21; the second one went unnoticed for fifteen hours.

So what is under test is the exclusion itself (a guest cannot run during a load, and a load
does not begin while a guest is running), that neither side can wait forever, and that a
load which DID break the context is remembered rather than retried once per request.
"""
import threading
import time

import pytest

from phansora.shared import gpu


@pytest.fixture(autouse=True)
def _fresh_gate(monkeypatch):
    """Module state is process-wide; reset it so tests cannot see each other's."""
    monkeypatch.setattr(gpu, "_guests", 0, raising=False)
    monkeypatch.setattr(gpu, "_exclusive", False, raising=False)
    monkeypatch.setattr(gpu, "_waiting", 0, raising=False)
    yield


def test_guest_waits_for_a_load_and_runs_after_it():
    order = []
    released = threading.Event()

    def load():
        with gpu.exclusive("load"):
            order.append("load-start")
            released.set()
            time.sleep(0.2)
            order.append("load-end")

    def caption():
        released.wait(2)
        with gpu.guest("caption"):
            order.append("caption")

    t1 = threading.Thread(target=load)
    t2 = threading.Thread(target=caption)
    t1.start()
    t2.start()
    t1.join(5)
    t2.join(5)

    # The capture is never overlapped: the guest lands strictly after the load finishes.
    assert order == ["load-start", "load-end", "caption"]


def test_a_load_waits_for_a_guest_already_on_the_device():
    order = []
    inside = threading.Event()

    def caption():
        with gpu.guest("caption"):
            inside.set()
            time.sleep(0.2)
            order.append("caption-end")

    def load():
        inside.wait(2)
        with gpu.exclusive("load"):
            order.append("load-start")

    t1 = threading.Thread(target=caption)
    t2 = threading.Thread(target=load)
    t1.start()
    t2.start()
    t1.join(5)
    t2.join(5)

    # Waited for, not interrupted — the in-flight decode keeps the device until it is done.
    assert order == ["caption-end", "load-start"]


def test_a_guest_that_cannot_get_the_device_raises_rather_than_barging_in():
    monkey = threading.Event()

    def hold():
        with gpu.exclusive("load"):
            monkey.wait(3)

    t = threading.Thread(target=hold, daemon=True)
    t.start()
    try:
        # Zero patience, so this is the timeout path and not a slow test.
        with pytest.MonkeyPatch.context() as m:
            m.setenv("PHANSORA_GPU_GUEST_WAIT_SEC", "0.05")
            with pytest.raises(gpu.GpuBusy):
                with gpu.guest("caption"):
                    pass
    finally:
        monkey.set()
        t.join(5)


def test_a_load_that_cannot_get_the_device_raises_before_it_starts(monkeypatch):
    """The important half: it must NOT proceed into a capture. Failing is the safe answer."""
    monkey = threading.Event()

    def hold():
        with gpu.guest("caption"):
            monkey.wait(3)

    t = threading.Thread(target=hold, daemon=True)
    t.start()
    try:
        monkeypatch.setenv("PHANSORA_GPU_EXCLUSIVE_WAIT_SEC", "0.05")
        with pytest.raises(gpu.GpuBusy):
            with gpu.exclusive("load"):
                pytest.fail("the load must not start while a guest holds the device")
    finally:
        monkey.set()
        t.join(5)


def test_two_guests_do_not_exclude_each_other():
    """Sharing the GPU is only illegal during a capture; the rest of the time it is fine."""
    both = threading.Barrier(2, timeout=3)
    ok = []

    def caption():
        with gpu.guest("caption"):
            both.wait()        # deadlocks if the gate serialized these
            ok.append(True)

    threads = [threading.Thread(target=caption) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert len(ok) == 2


def test_inactive_guest_is_a_no_op_and_never_delays_a_load():
    """A CPU-bound caller holds no GPU, so it must not make a load wait for it."""
    with gpu.guest("caption on the cpu", active=False):
        with gpu.exclusive("load"):
            pass


# ── The other half: a broken context is remembered ───────────────────────────

def _fake_cosyvoice(monkeypatch, raises: Exception):
    """Stand in for the CosyVoice checkout, with a model construction that fails how we say.

    The real one is a git checkout the dev box does not have, so ``from
    cosyvoice.cli.cosyvoice import AutoModel`` would fail first and on the wrong error.
    """
    import sys
    import types

    calls = []

    def AutoModel(**_kwargs):
        calls.append(1)
        raise raises

    pkg = types.ModuleType("cosyvoice")
    cli = types.ModuleType("cosyvoice.cli")
    leaf = types.ModuleType("cosyvoice.cli.cosyvoice")
    leaf.AutoModel = AutoModel
    cli.cosyvoice = leaf
    pkg.cli = cli
    monkeypatch.setitem(sys.modules, "cosyvoice", pkg)
    monkeypatch.setitem(sys.modules, "cosyvoice.cli", cli)
    monkeypatch.setitem(sys.modules, "cosyvoice.cli.cosyvoice", leaf)
    return calls


def _clean_client(monkeypatch):
    from phansora.products.spokenverse.txt_to_voice.adapters import cosyvoice3_client as client

    monkeypatch.setattr(client, "_COSY", None, raising=False)
    monkeypatch.setattr(client, "_DEAD", None, raising=False)
    monkeypatch.setenv("COSYVOICE3_REPO", "/nonexistent/CosyVoice")
    # No GPU in a test, so no vLLM and no capture — the failure mode is what is under test,
    # not the device.
    monkeypatch.setattr(client, "_cuda_available", lambda: False)
    return client


def test_a_poisoned_load_is_not_retried_and_names_the_cure(monkeypatch):
    client = _clean_client(monkeypatch)
    calls = _fake_cosyvoice(monkeypatch, RuntimeError(
        'captures_underway.empty() INTERNAL ASSERT FAILED at '
        '"/pytorch/c10/cuda/CUDACachingAllocator.cpp":3089, please report a bug to PyTorch.'
    ))

    with pytest.raises(client.EngineNeedsRestart) as first:
        client._load_cosy()
    assert "restart" in str(first.value).lower()
    assert client.unavailable_reason() is not None
    assert len(calls) == 1

    # The second attempt must not touch the model again. The whole cost of this bug was
    # 25-35s of a worker thread per request on a load that could never succeed.
    with pytest.raises(client.EngineNeedsRestart):
        client._load_cosy()
    assert len(calls) == 1


def test_the_other_capture_failure_is_recognised_too(monkeypatch):
    """The first process died on this wording, the next one on the assert above."""
    client = _clean_client(monkeypatch)
    _fake_cosyvoice(monkeypatch, RuntimeError(
        "CUDA error: operation failed due to a previous error during capture"
    ))
    with pytest.raises(client.EngineNeedsRestart):
        client._load_cosy()
    assert client.unavailable_reason() is not None


def test_an_ordinary_load_failure_is_still_retryable(monkeypatch):
    """A missing checkout or a bad path is fixable without a restart — do not remember it."""
    client = _clean_client(monkeypatch)
    calls = _fake_cosyvoice(monkeypatch, RuntimeError("No such file or directory: llm.pt"))

    with pytest.raises(RuntimeError) as exc:
        client._load_cosy()
    assert not isinstance(exc.value, client.EngineNeedsRestart)
    assert client.unavailable_reason() is None

    # And it IS tried again, because the next request might be after somebody fixed it.
    with pytest.raises(RuntimeError):
        client._load_cosy()
    assert len(calls) == 2


def test_a_busy_gpu_does_not_mark_the_engine_dead(monkeypatch):
    """GpuBusy means the load never began. Nothing is broken, so nothing is remembered."""
    client = _clean_client(monkeypatch)
    _fake_cosyvoice(monkeypatch, RuntimeError("unreachable"))
    monkeypatch.setattr(client, "_env_bool", lambda name, default=False: name == "COSYVOICE3_USE_VLLM")

    def busy(*_a, **_k):
        raise gpu.GpuBusy("the GPU is still loading the voice engine")

    monkeypatch.setattr(client.gpu, "exclusive", busy)
    with pytest.raises(gpu.GpuBusy):
        client._load_cosy()
    assert client.unavailable_reason() is None
