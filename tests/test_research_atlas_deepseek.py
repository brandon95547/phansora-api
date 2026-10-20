"""Research Atlas's calls to DeepSeek go through the shared helper like every product's.

Each one is capped and asks for no thinking, which it used to send neither of, and a
failure is still the stage's to survive: tried once more, then None.
"""
from __future__ import annotations

import json

import httpx
import pytest

from phansora.products.research_atlas import text_cleaner
from phansora.products.research_atlas.config import Config
from phansora.shared.ai import deepseek as ds


@pytest.fixture
def provider(monkeypatch):
    """Answers every request with ``status`` (and an answer when it is 200); keeps what
    was sent, and the timeout each request was given."""
    state = {"status": 200, "sent": [], "timeouts": []}

    def answer(request: httpx.Request) -> httpx.Response:
        state["sent"].append(json.loads(request.content))
        if state["status"] != 200:
            return httpx.Response(state["status"], json={"error": "busy"})
        return httpx.Response(200, json={"choices": [{"message": {"content": " Restored. "}, "finish_reason": "stop"}]})

    real_client = httpx.Client

    def client(**kwargs):
        state["timeouts"].append(kwargs.get("timeout"))
        return real_client(transport=httpx.MockTransport(answer), **kwargs)

    for var in ("RESEARCH_ATLAS_MODEL", "RESEARCH_ATLAS_MAX_TOKENS", "DEEPSEEK_TIMEOUT_SECONDS", "DEEPSEEK_MAX_RETRIES"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-test")
    monkeypatch.setattr(ds.httpx, "Client", client)
    monkeypatch.setattr(ds.time, "sleep", lambda _s: None)
    return state


def test_every_call_is_capped_and_asks_for_no_thinking(provider):
    llm = Config().deepseek_client
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert llm.ask(messages, temperature=0.1) == "Restored."
    (sent,) = provider["sent"]
    assert sent["model"] == "deepseek-test" and sent["messages"] == messages
    assert sent["max_tokens"] == 8000 and sent["thinking"] == {"type": "disabled"}
    assert sent["temperature"] == 0.1
    assert provider["timeouts"] == [90.0]


def test_a_failed_call_is_tried_once_more_then_left_to_the_stage(provider):
    provider["status"] = 503
    assert text_cleaner._clean_chunk("x" * 40, Config().deepseek_client) is None
    assert len(provider["sent"]) == 2  # DEEPSEEK_MAX_RETRIES defaults to one retry
