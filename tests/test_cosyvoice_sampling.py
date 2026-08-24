"""CosyVoice2's sampling config, and the fact that vLLM throws it away.

``CosyVoice2-0.5B/cosyvoice2.yaml`` declares ``ras_sampling(top_p=0.8, top_k=25,
win_size=10, tau_r=0.1)``. That is only ever reached on the plain-torch path.
``Qwen2LM.inference_wrapper`` builds vLLM's ``SamplingParams`` from scratch and carries
across exactly one field — ``top_k`` — so on the accelerated path top_p silently becomes
1.0 (vLLM's own default) and the repetition-aware resampling is gone entirely. Sampling
the full 25-candidate tail at temperature 1.0 is heard as an intermittently garbled WORD,
which is a different failure from the dropped words that MAX_CHARS=200 addresses.

``_install_sampling_defaults`` puts the yaml's values back by swapping the ``SamplingParams``
attribute on the ``vllm`` package — the attribute CosyVoice resolves at call time, since its
import sits inside ``inference_wrapper``. Upstream is never edited (``make install-tts``
re-clones the checkout, so a patched fork would not survive a redeploy).

These tests exist because the patch is invisible from our own call sites: nothing in
``cosyvoice2_client`` reads top_p, so a regression here would be silent and would only show
up as audio quality months later.
"""
from __future__ import annotations

import sys
import types

import pytest


class _FakeSamplingParams:
    """Stands in for vllm.SamplingParams — records whatever it was constructed with."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture()
def vllm_stub(monkeypatch):
    """A stub ``vllm`` package. vLLM is CUDA-only and not installed on dev machines.

    Mirrors the real layout deliberately: the class lives in ``vllm.sampling_params`` and is
    re-exported from ``vllm``. Patching only the re-export is what keeps vLLM's internal
    isinstance checks working, so the stub has to have both for that to be testable.
    """
    real = _FakeSamplingParams
    pkg = types.ModuleType("vllm")
    pkg.SamplingParams = real
    sub = types.ModuleType("vllm.sampling_params")
    sub.SamplingParams = real
    monkeypatch.setitem(sys.modules, "vllm", pkg)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sub)
    for var in ("TOP_P", "TOP_K", "TEMPERATURE", "REPETITION_PENALTY", "SEED"):
        monkeypatch.delenv(f"COSYVOICE2_{var}", raising=False)
    return pkg


@pytest.fixture()
def client():
    return pytest.importorskip(
        "phansora.products.spokenverse.txt_to_voice.adapters.cosyvoice2_client",
        reason="spokenverse adapter not importable on this host",
    )


def _upstream_call(pkg):
    """Exactly how cosyvoice/llm/llm.py builds it — kwargs only, top_k and nothing else."""
    from vllm import SamplingParams  # resolved off the (patched) package, as upstream does

    assert SamplingParams is pkg.SamplingParams
    return SamplingParams(
        top_k=25, stop_token_ids=[6561, 6562, 6563], min_tokens=4, max_tokens=400
    ).kwargs


def test_restores_the_yaml_sampling_config(vllm_stub, client):
    """top_p is the whole point: vLLM would leave it at 1.0 and sample the full tail."""
    client._install_sampling_defaults()
    got = _upstream_call(vllm_stub)

    assert got["top_p"] == client.TOP_P_DEFAULT == 0.8
    assert got["top_k"] == client.TOP_K_DEFAULT == 25
    assert got["temperature"] == client.TEMPERATURE_DEFAULT == 1.0
    assert got["repetition_penalty"] == client.REPETITION_PENALTY_DEFAULT == 1.0


def test_leaves_upstreams_own_arguments_alone(vllm_stub, client):
    """The stop tokens and length bounds are load-bearing — injection must not disturb them."""
    client._install_sampling_defaults()
    got = _upstream_call(vllm_stub)

    assert got["stop_token_ids"] == [6561, 6562, 6563]
    assert got["min_tokens"] == 4
    assert got["max_tokens"] == 400


def test_env_overrides_win(vllm_stub, client, monkeypatch):
    monkeypatch.setenv("COSYVOICE2_TOP_P", "0.7")
    monkeypatch.setenv("COSYVOICE2_TEMPERATURE", "0.85")
    monkeypatch.setenv("COSYVOICE2_REPETITION_PENALTY", "1.05")
    client._install_sampling_defaults()
    got = _upstream_call(vllm_stub)

    assert got["top_p"] == 0.7
    assert got["temperature"] == 0.85
    assert got["repetition_penalty"] == 1.05


def test_seed_is_unset_unless_asked_for(vllm_stub, client):
    """Production wants a fresh roll per render; a seed is only for A/B testing."""
    client._install_sampling_defaults()
    assert "seed" not in _upstream_call(vllm_stub)


def test_seed_is_passed_through_when_set(vllm_stub, client, monkeypatch):
    monkeypatch.setenv("COSYVOICE2_SEED", "1234")
    client._install_sampling_defaults()
    assert _upstream_call(vllm_stub)["seed"] == 1234


def test_unparseable_seed_is_ignored_not_fatal(vllm_stub, client, monkeypatch):
    """A typo in .env must not take TTS down at startup."""
    monkeypatch.setenv("COSYVOICE2_SEED", "not-a-number")
    client._install_sampling_defaults()
    assert "seed" not in _upstream_call(vllm_stub)


def test_a_caller_that_supplies_its_own_value_keeps_it(vllm_stub, client):
    """If CosyVoice is ever fixed to pass top_p itself, upstream must win and this go quiet."""
    client._install_sampling_defaults()
    from vllm import SamplingParams

    assert SamplingParams(top_k=25, top_p=0.55).kwargs["top_p"] == 0.55


def test_patching_twice_does_not_double_wrap(vllm_stub, client):
    """preload() and a racing first request both reach _load_cosy; wrapping twice would
    stack closures and make the effective values depend on call order."""
    client._install_sampling_defaults()
    once = vllm_stub.SamplingParams
    client._install_sampling_defaults()
    client._install_sampling_defaults()

    assert vllm_stub.SamplingParams is once
    assert _upstream_call(vllm_stub)["top_p"] == 0.8


def test_positional_construction_passes_through_untouched(vllm_stub, client):
    """SamplingParams' field order makes setdefault ambiguous against positional args, so
    the wrapper declines to inject rather than risk binding a value to the wrong field."""
    class _Positional:
        def __init__(self, n=1, **kwargs):
            self.n, self.kwargs = n, kwargs

    vllm_stub.SamplingParams = _Positional
    client._install_sampling_defaults()
    from vllm import SamplingParams

    built = SamplingParams(7, top_k=25)
    assert built.n == 7
    assert "top_p" not in built.kwargs


def test_vllm_internals_still_see_the_real_class(vllm_stub, client):
    """Only the top-level re-export is swapped. vLLM imports the class from
    vllm.sampling_params for its own isinstance checks, so those must be unaffected."""
    client._install_sampling_defaults()

    assert sys.modules["vllm.sampling_params"].SamplingParams is _FakeSamplingParams
    # And what the wrapper hands back is a genuine instance, not a stand-in.
    assert isinstance(vllm_stub.SamplingParams(top_k=25), _FakeSamplingParams)
