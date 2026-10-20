"""Provider calls are retried only when a second attempt could come out differently.

A timeout, a dropped connection, a 429 or a 5xx can; a 401 (bad key), 402 (no credit)
or 403 cannot, and retrying one with backoff only delays the real error. The research
clients also used to bury it in tenacity's RetryError, so a caller catching
RuntimeError for "the provider is down" got a 500 instead.
"""
from __future__ import annotations

import asyncio
import copy
import json as jsonlib
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


# ── the shared DeepSeek helper (aiohttp, and httpx for its sync twin) ──────

_OK = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}


class _Provider:
    """DeepSeek's side of the call: each request gets the next answer in turn, the last
    one repeating. An answer is a status, or a (status, body) pair. Keeps what was sent."""

    def __init__(self):
        self.script(200)

    def script(self, *answers):
        self.answers = list(answers)
        self.sent: list[dict] = []
        self.statuses: list[int] = []

    def answer(self, payload):
        answer = self.answers[min(len(self.sent), len(self.answers) - 1)]
        status, body = answer if isinstance(answer, tuple) else (answer, _OK if answer < 400 else {"error": "nope"})
        self.sent.append(copy.deepcopy(payload))
        self.statuses.append(status)
        return status, body


class _AioSession:
    """Stands in for aiohttp.ClientSession."""

    def __init__(self, provider):
        self._provider = provider

    def __call__(self, *a, **kw):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, json, headers):
        return _AioResponse(*self._provider.answer(json))


class _AioResponse:
    def __init__(self, status, body):
        self.status = status
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def text(self):
        return body_text(self._body)

    async def json(self, content_type=None):
        return self._body


class _HttpxClient:
    """Stands in for httpx.Client."""

    def __init__(self, provider):
        self._provider = provider

    def __call__(self, *a, **kw):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json, headers):
        status, body = self._provider.answer(json)
        return httpx.Response(status, text=body_text(body), request=httpx.Request("POST", url))


def body_text(body):
    return body if isinstance(body, str) else jsonlib.dumps(body)


@pytest.fixture(params=["async", "sync"])
def deepseek_call(request, monkeypatch):
    """``run(answers, **chat_kwargs)`` through chat() or chat_sync(), with the provider
    that answered; ``json=True`` goes through chat_json() / chat_json_sync()."""
    from phansora.shared.ai import deepseek

    provider = _Provider()
    monkeypatch.setattr(deepseek.aiohttp, "ClientSession", _AioSession(provider))
    monkeypatch.setattr(deepseek.httpx, "Client", _HttpxClient(provider))
    cfg = deepseek.DeepSeekChatConfig(
        base_url="http://x", api_key="k", model="m", max_retries=3,
        min_retry_sleep_s=0, max_retry_sleep_s=0,
    )

    def run(answers, *, json=False, **kwargs):
        provider.script(*answers)
        kwargs.setdefault("max_tokens", 10)
        if "messages" not in kwargs:
            kwargs.update(system="s", user="u")
        if request.param == "sync":
            return (deepseek.chat_json_sync if json else deepseek.chat_sync)(cfg=cfg, **kwargs)
        return asyncio.run((deepseek.chat_json if json else deepseek.chat)(cfg=cfg, **kwargs))

    run.provider = provider
    return run


def test_deepseek_does_not_retry_an_empty_account(deepseek_call):
    with pytest.raises(ProviderError) as caught:
        deepseek_call([402])
    assert deepseek_call.provider.statuses == [402]
    assert caught.value.status == 402


def test_deepseek_retries_a_busy_provider_until_it_answers(deepseek_call):
    assert deepseek_call([503, 429, 200]) == ("ok", "stop")
    assert deepseek_call.provider.statuses == [503, 429, 200]


def test_deepseek_gives_up_after_its_attempts_with_the_status(deepseek_call):
    with pytest.raises(ProviderError) as caught:
        deepseek_call([503])
    assert len(deepseek_call.provider.sent) == 4  # max_retries=3 -> four attempts
    assert caught.value.status == 503


def test_a_callers_own_retry_count_is_kept(deepseek_call):
    with pytest.raises(ProviderError):
        deepseek_call([503], max_retries=0)
    assert len(deepseek_call.provider.sent) == 1


def test_a_model_that_does_not_know_thinking_is_asked_again_without_it(deepseek_call):
    # Not an attempt: even a caller that never retries gets its answer.
    refused = (400, {"error": {"message": "unknown parameter: thinking"}})
    assert deepseek_call([refused, 200], max_retries=0).text == "ok"
    first, second = deepseek_call.provider.sent
    assert first["thinking"] == {"type": "disabled"} and "thinking" not in second


def test_what_is_sent(deepseek_call):
    turns = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "so far"},
        {"role": "user", "content": "go on"},
    ]
    deepseek_call([200], messages=turns, temperature=0.7, reasoning_effort="none")
    sent = deepseek_call.provider.sent[0]
    assert sent["messages"] == turns and sent["temperature"] == 0.7 and sent["max_tokens"] == 10
    assert sent["thinking"] == {"type": "disabled"} and sent["reasoning_effort"] == "none"
    assert "response_format" not in sent

    deepseek_call([200], disable_thinking=False)
    assert "thinking" not in deepseek_call.provider.sent[0]


def test_a_cut_off_json_reply_is_asked_again_with_double_the_budget(deepseek_call):
    cut = (200, {"choices": [{"message": {"content": '{"a": [1, '}, "finish_reason": "length"}]})
    whole = (200, {"choices": [{"message": {"content": '{"a": [1, 2]}'}, "finish_reason": "stop"}]})
    assert deepseek_call([cut, whole], json=True, max_tokens=100, max_tokens_ceiling=400) == ('{"a": [1, 2]}', "stop")
    assert [s["max_tokens"] for s in deepseek_call.provider.sent] == [100, 200]
    assert all(s["response_format"] == {"type": "json_object"} for s in deepseek_call.provider.sent)


def test_the_json_budget_stops_at_its_ceiling_and_says_so(deepseek_call):
    cut = (200, {"choices": [{"message": {"content": "{"}, "finish_reason": "length"}]})
    assert deepseek_call([cut], json=True, max_tokens=100, max_tokens_ceiling=300).finish_reason == "length"
    assert [s["max_tokens"] for s in deepseek_call.provider.sent] == [100, 200, 300]


def test_no_ceiling_means_no_escalation(deepseek_call):
    cut = (200, {"choices": [{"message": {"content": "{"}, "finish_reason": "length"}]})
    deepseek_call([cut], json=True, max_tokens=100)
    assert len(deepseek_call.provider.sent) == 1


def test_a_refusal_says_what_to_check():
    assert retries.refusal_message(401, provider="DeepSeek", key_var="DEEPSEEK_API_KEY") == (
        "DeepSeek rejected the API key. Check DEEPSEEK_API_KEY."
    )
    assert "out of credit" in retries.refusal_message(402, provider="OpenAI")
    assert retries.refusal_message(500, provider="DeepSeek") is None


def test_talari_reads_the_status_off_the_error():
    from phansora.products.talari import llm

    failure = llm._failure(ProviderError("DeepSeek clean failed: body", status=402))
    assert str(failure) == llm.refusal(402)


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
