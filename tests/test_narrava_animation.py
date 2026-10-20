"""Narrava's animation call to DeepSeek: what it sends, and what a cut-off document means.

Through the shared helper's sync twin, with the network replaced at httpx.
"""
from __future__ import annotations

import json

import httpx
import pytest

from phansora.products.narrava_studio.services import animation
from phansora.shared.ai import deepseek as ds

_DOC = '<!doctype html><canvas id="stage"></canvas><script>window.renderFrame = t => {};</script>'


@pytest.fixture
def provider(monkeypatch):
    """Answers each request with the next (content, finish_reason); keeps what was sent."""
    sent: list[dict] = []
    answers: list[tuple[str, str]] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        content, finish = answers.pop(0)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"completion_tokens": 10, "prompt_tokens": 5},
        })

    monkeypatch.setenv("NARRAVA_ANIMATION_PROVIDER", "deepseek")
    monkeypatch.delenv("NARRAVA_ANIMATION_REASONING_EFFORT", raising=False)
    monkeypatch.setattr(
        ds.DeepSeekChatConfig, "from_env",
        classmethod(lambda cls, **k: ds.DeepSeekChatConfig(base_url="http://x", api_key="k", model="m")),
    )
    real_client = httpx.Client
    monkeypatch.setattr(ds.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(answer), **kw))
    return sent, answers


def _generate():
    return animation.generate_animation_html("a ball", duration_sec=3, width=640, height=360)


def test_the_thinking_is_off_both_ways_by_default(provider):
    sent, answers = provider
    answers.append((_DOC, "stop"))
    assert "renderFrame" in _generate()
    (request,) = sent
    assert request["thinking"] == {"type": "disabled"} and request["reasoning_effort"] == "none"
    assert request["temperature"] == 0.2


def test_an_effort_handed_back_to_the_provider_sends_neither(provider, monkeypatch):
    sent, answers = provider
    monkeypatch.setenv("NARRAVA_ANIMATION_REASONING_EFFORT", "default")
    answers.append((_DOC, "stop"))
    _generate()
    assert "thinking" not in sent[0] and "reasoning_effort" not in sent[0]


def test_a_cut_off_document_is_a_retryable_generation_problem(provider):
    _sent, answers = provider
    answers.append(("<!doctype html><canvas", "length"))
    with pytest.raises(ValueError, match="cut off"):
        _generate()
