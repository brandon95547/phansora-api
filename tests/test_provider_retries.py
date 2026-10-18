"""Provider calls are retried only when a second attempt could come out differently.

A timeout, a dropped connection, a 429 or a 5xx can; a 401 (bad key), 402 (no credit)
or 403 cannot, and retrying one with backoff only delays the real error. The research
clients also used to bury it in tenacity's RetryError, so a caller catching
RuntimeError for "the provider is down" got a 500 instead.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from tenacity import wait_none

from phansora.shared.ai import retries
from phansora.shared.ai.retries import ProviderError


# ── the predicate ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_busy_or_broken_is_worth_another_attempt(status):
    assert retries.worth_retrying(ProviderError("x", status=status), ())


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422])
def test_a_refusal_is_not(status):
    assert not retries.worth_retrying(ProviderError("x", status=status), ())


def test_transport_errors_are_the_callers_to_name():
    assert retries.worth_retrying(httpx.ConnectError("down"), (httpx.TransportError,))
    assert not retries.worth_retrying(KeyError("bug"), (httpx.TransportError,))


# ── the shared DeepSeek helper (aiohttp) ────────────────────────────────────

class _FakeSession:
    """Stands in for aiohttp.ClientSession, answering each post with the next status."""

    def __init__(self, statuses, calls):
        self._statuses = statuses
        self._calls = calls

    def __call__(self, *a, **kw):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, json, headers):
        status = self._statuses[min(len(self._calls), len(self._statuses) - 1)]
        self._calls.append(status)
        return _FakeResponse(status)


class _FakeResponse:
    def __init__(self, status):
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def text(self):
        return '{"error": "nope"}'

    async def json(self):
        return {"choices": [{"message": {"content": "ok"}}]}


def _deepseek(monkeypatch, statuses):
    from phansora.shared.ai import deepseek

    calls: list[int] = []
    monkeypatch.setattr(deepseek.aiohttp, "ClientSession", _FakeSession(statuses, calls))
    cfg = deepseek.DeepSeekChatConfig(
        base_url="http://x", api_key="k", model="m", max_retries=3,
        min_retry_sleep_s=0, max_retry_sleep_s=0,
    )

    def run():
        return asyncio.run(deepseek._chat_completion(
            cfg=cfg, system_prompt="s", user_text="u", max_output_tokens=10,
        ))

    return run, calls


def test_deepseek_does_not_retry_an_empty_account(monkeypatch):
    run, calls = _deepseek(monkeypatch, [402])
    with pytest.raises(ProviderError) as caught:
        run()
    assert calls == [402]
    assert caught.value.status == 402


def test_deepseek_retries_a_busy_provider_until_it_answers(monkeypatch):
    run, calls = _deepseek(monkeypatch, [503, 429, 200])
    assert run() == "ok"
    assert calls == [503, 429, 200]


def test_deepseek_gives_up_after_its_attempts_with_the_status(monkeypatch):
    run, calls = _deepseek(monkeypatch, [503])
    with pytest.raises(ProviderError) as caught:
        run()
    assert len(calls) == 4  # max_retries=3 -> four attempts
    assert caught.value.status == 503


def test_talari_reads_the_status_off_the_error():
    from phansora.products.talari import llm

    failure = llm._failure(ProviderError("DeepSeek clean failed: body", status=402))
    assert str(failure) == llm.REFUSALS[402]


# ── the Gemini research client (httpx + tenacity) ───────────────────────────

def _gemini(monkeypatch, statuses):
    from phansora.shared.ai import gemini_research as G

    calls: list[int] = []

    class Client:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, json, headers):
            status = statuses[min(len(calls), len(statuses) - 1)]
            calls.append(status)
            return httpx.Response(status, json={"candidates": []}, request=httpx.Request("POST", url))

    monkeypatch.setattr(G.httpx, "Client", Client)
    monkeypatch.setattr(G.GeminiResearchClient._generate.retry, "wait", wait_none())
    client = G.GeminiResearchClient(G.GeminiConfig(api_key="k", model="m"))
    return client, calls


def test_gemini_fails_fast_on_a_bad_key_with_its_own_error(monkeypatch):
    client, calls = _gemini(monkeypatch, [401])
    with pytest.raises(ProviderError) as caught:
        client._generate(model="m", prompt="p", temperature=0, max_tokens=10)
    assert calls == [401]
    assert caught.value.status == 401


def test_gemini_retries_a_5xx_and_surfaces_the_last_one(monkeypatch):
    client, calls = _gemini(monkeypatch, [503])
    with pytest.raises(ProviderError) as caught:
        client._generate(model="m", prompt="p", temperature=0, max_tokens=10)
    assert calls == [503, 503, 503]
    assert caught.value.status == 503


# ── the OpenAI research client ──────────────────────────────────────────────

def _openai_error(cls, status):
    request = httpx.Request("POST", "https://api.openai.test/v1/responses")
    return cls("refused", response=httpx.Response(status, request=request), body=None)


def test_openai_refusals_fail_fast_and_busy_answers_retry():
    import openai

    from phansora.shared.ai import openai_research as O

    assert not O._worth_retrying(_openai_error(openai.AuthenticationError, 401))
    assert not O._worth_retrying(_openai_error(openai.PermissionDeniedError, 403))
    assert O._worth_retrying(_openai_error(openai.RateLimitError, 429))
    assert O._worth_retrying(_openai_error(openai.InternalServerError, 500))
    assert O._worth_retrying(openai.APITimeoutError(request=httpx.Request("POST", "https://x.test")))


def test_openai_leaves_retrying_to_one_layer(monkeypatch):
    """The SDK's own two retries inside each of tenacity's three attempts was nine calls."""
    from phansora.shared.ai import openai_research as O

    client = O.OpenAIResearchClient(SimpleNamespace(
        api_key="k", base_url="", timeout_s=120,
    ))
    assert client._client.max_retries == 0
