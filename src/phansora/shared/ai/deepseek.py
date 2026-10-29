# src/phansora/shared/ai/deepseek.py
#
# The one DeepSeek chat implementation every product calls through: its config
# (DeepSeekChatConfig), the request with its retries and its `thinking` fallback (chat and
# chat_json, with chat_sync and chat_json_sync for the callers that run in a thread),
# whether a key is set (key_configured), and raw OCR text -> cleaned, readable book text
# (clean_ocr_text). What an account-level refusal means lives in shared.ai.retries
# (refusal_message), because Narrava says it of OpenAI too.
#
# Env vars (via .env):
#   DEEPSEEK_BASE_URL=https://api.deepseek.com
#   DEEPSEEK_API_KEY=...
#   DEEPSEEK_MODEL=deepseek-v4-flash   (or a <PRODUCT>_MODEL; see shared/ai/models.py)
# The older DEEPSEEK_CHAT_* names are still read as a fallback.

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import aiohttp
import httpx

from . import usage
from .models import resolve_model
from .retries import ProviderError, worth_retrying

logger = logging.getLogger(__name__)

# What a DeepSeek call can fail with that a second attempt might not: each transport's
# connection and protocol errors, and its timeouts.
_ASYNC_TRANSPORT_ERRORS = (aiohttp.ClientError, asyncio.TimeoutError)
_SYNC_TRANSPORT_ERRORS = (httpx.TransportError,)

# The key's name, for messages that say what to set. The legacy DEEPSEEK_CHAT_API_KEY is
# still read after it.
KEY_VAR = "DEEPSEEK_API_KEY"

# Appended to a system prompt whose answer must be JSON. Book Alchemy and Talari both ask
# in these words.
JSON_ONLY = "\n\nRespond with valid JSON only. No prose, no markdown fences."


DEFAULT_CLEAN_PROMPT = """
You are cleaning OCR output from a book PDF.

Goal: produce clean, readable plain text while preserving the original content and structure.

KEEP:
- All meaningful book text, including headings/subheadings and references.
- The original ordering of paragraphs and sections.

REMOVE COMPLETELY:
- Only obvious OCR garbage/noise (e.g., repeated gibberish symbols, line-art labels, broken axis ticks).
- Digitization boilerplate (e.g., “Digitized by the Internet Archive”, funding lines, URLs).

CLEANUP RULES:
- Correct obvious OCR spelling mistakes when context is clear.
- Fix grammar/punctuation where OCR introduced errors, without changing meaning.
- Fix broken hyphenation across line breaks (e.g., "transfor-\nmation" -> "transformation").
- Join hard-wrapped lines into proper paragraphs.
- Preserve paragraph breaks.
- Do NOT invent new content or rewrite style.
- If uncertain, keep the text rather than deleting it.

OUTPUT:
- Return plain text only. No markdown.
""".strip()

# Model names + the per-product override chain live in shared.ai.models (see that module for
# the resolution order). Nothing is re-exported: there is no default name to re-export.


def api_key() -> str:
    """The DeepSeek key from the environment, or "" when there is none."""
    return (os.getenv(KEY_VAR) or os.getenv("DEEPSEEK_CHAT_API_KEY") or "").strip()


def key_configured() -> bool:
    """Is a DeepSeek key set? What a product's /health and its 503 for a missing key ask."""
    return bool(api_key())


@dataclass(frozen=True)
class DeepSeekChatConfig:
    base_url: str
    api_key: str
    model: str

    timeout_s: int = 180
    max_retries: int = 4
    min_retry_sleep_s: float = 0.8
    max_retry_sleep_s: float = 3.0

    @staticmethod
    def from_env(
        *,
        default_base_url: str = "https://api.deepseek.com",
        product_var: Optional[str] = None,
    ) -> "DeepSeekChatConfig":
        """``product_var`` names this product's model override (e.g. "BOOK_ALCHEMY_MODEL");
        when set it beats DEEPSEEK_MODEL. See shared.ai.models for the full order."""
        # Canonical vars are DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL. The
        # legacy DEEPSEEK_CHAT_* names are still honored as a fallback so existing
        # deployments keep working.
        base_url = (
            os.getenv("DEEPSEEK_BASE_URL")
            or os.getenv("DEEPSEEK_CHAT_BASE_URL")
            or default_base_url
        ).rstrip("/")
        model = resolve_model(product_var, provider="deepseek")

        key = api_key()
        if not key:
            raise RuntimeError(
                "Missing DEEPSEEK_API_KEY environment variable. Set it in .env."
            )

        return DeepSeekChatConfig(base_url=base_url, api_key=key, model=model)


class ChatReply(tuple):
    """``(text, finish_reason)``, unpacking like the pair it is.

    ``finish_reason`` is "length" when the reply ran out of ``max_tokens``: without it a
    cut-off answer is indistinguishable from a complete one. The response body is kept on
    ``.body`` for a caller that needs more than the answer (the usage, the reasoning).
    """

    body: dict

    def __new__(cls, text: str, finish_reason: Optional[str], body: Optional[dict] = None) -> "ChatReply":
        reply = super().__new__(cls, (text, finish_reason))
        reply.body = body or {}
        return reply

    @property
    def text(self) -> str:
        return self[0]

    @property
    def finish_reason(self) -> Optional[str]:
        return self[1]


def _payload(
    *,
    model: str,
    system: Optional[str],
    user: Optional[str],
    messages: Optional[Sequence[dict]],
    max_tokens: int,
    temperature: float,
    json_mode: bool,
    disable_thinking: bool,
    reasoning_effort: Optional[str],
) -> dict[str, Any]:
    if messages is None:
        if system is None or user is None:
            raise TypeError("chat() takes system and user, or messages")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": list(messages),
        "max_tokens": max_tokens,
        "stream": False,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if disable_thinking:
        # The v4 models reason by default and bill those tokens against max_tokens. A
        # budget sized for the answer alone then comes back empty or cut off: OCR cleaning
        # spent whole page batches thinking, Book Alchemy wrote blank lesson scripts, and
        # Narrava's animations came back with every token spent and no document.
        payload["thinking"] = {"type": "disabled"}
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    return payload


class _Call:
    """One chat request through its attempts: what is sent, what the answer means, and
    whether a failure is worth another try. chat() and chat_sync() differ only in how
    they send and how they wait."""

    def __init__(
        self,
        cfg: DeepSeekChatConfig,
        payload: dict[str, Any],
        *,
        timeout_s: Optional[float],
        max_retries: Optional[int],
    ) -> None:
        self.url = f"{cfg.base_url}/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        }
        self.payload = payload
        self.timeout_s = cfg.timeout_s if timeout_s is None else timeout_s
        self._cfg = cfg
        self._max_retries = cfg.max_retries if max_retries is None else max_retries
        self._attempt = 0

    def failed(self, status: int, body: str) -> None:
        """Raise the error for an HTTP failure, unless the request can go again as it is.

        `thinking` is a v4-era parameter. A model that predates it 400s rather than
        ignoring it, so it is dropped and the request sent again rather than failing the
        call: the model name comes from .env and can be pointed anywhere. That resend is a
        correction, not an attempt, so it is there even for a caller that never retries.
        """
        if status == 400 and "thinking" in body and "thinking" in self.payload:
            self.payload.pop("thinking")
            return
        raise ProviderError(f"DeepSeek HTTP {status}: {body[:800]}", status=status)

    def read(self, data: dict) -> ChatReply:
        usage.record_response(data)
        choices = data.get("choices") or []
        if not choices:
            return ChatReply("", None, data)
        content = (choices[0].get("message") or {}).get("content") or ""
        return ChatReply(content.strip(), choices[0].get("finish_reason"), data)

    def backoff(self, error: Exception, transport_errors: tuple) -> float:
        """Seconds to wait before the next attempt, or the call's error when there is none.

        Only what a second attempt might change is retried: a timeout, a dropped
        connection, a 429 or a 5xx. A bad key (401), an empty account (402) or a refused
        model (403) answers the same way every time.
        """
        if self._attempt >= self._max_retries or not worth_retrying(error, transport_errors):
            raise ProviderError(
                f"DeepSeek call failed after {self._attempt + 1} attempt(s): {error}",
                status=getattr(error, "status", None),
            ) from error
        sleep_s = min(
            self._cfg.max_retry_sleep_s,
            self._cfg.min_retry_sleep_s * (2 ** self._attempt) + random.random() * 0.25,
        )
        self._attempt += 1
        return sleep_s


async def chat(
    *,
    cfg: DeepSeekChatConfig,
    system: Optional[str] = None,
    user: Optional[str] = None,
    messages: Optional[Sequence[dict]] = None,
    max_tokens: int,
    # 0 suits cleaning and extraction. Writing prose wants more (DeepSeek's own guidance
    # puts creative writing well above 1), so callers that write ask for it.
    temperature: float = 0,
    json_mode: bool = False,
    disable_thinking: bool = True,
    reasoning_effort: Optional[str] = None,
    timeout_s: Optional[float] = None,
    max_retries: Optional[int] = None,
) -> ChatReply:
    """One DeepSeek reply, as ``(text, finish_reason)``. Raises ProviderError.

    Ask with ``system`` and ``user``, or with ``messages`` for a conversation that goes
    on: a reply cut off and asked to carry on, a reply handed back to be corrected.
    ``timeout_s`` and ``max_retries`` default to the config's. The timeout is the whole
    call's (aiohttp's total), since a reply that is not streamed sends nothing until it is
    finished.
    """
    call = _Call(
        cfg,
        _payload(
            model=cfg.model, system=system, user=user, messages=messages,
            max_tokens=max_tokens, temperature=temperature, json_mode=json_mode,
            disable_thinking=disable_thinking, reasoning_effort=reasoning_effort,
        ),
        timeout_s=timeout_s, max_retries=max_retries,
    )
    timeout = aiohttp.ClientTimeout(total=call.timeout_s)
    while True:
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(call.url, json=call.payload, headers=call.headers) as resp:
                    if resp.status >= 400:
                        call.failed(resp.status, await resp.text())
                        continue
                    data = await resp.json(content_type=None)
            return call.read(data)
        except Exception as error:  # noqa: BLE001 — backoff raises what is not worth another try
            await asyncio.sleep(call.backoff(error, _ASYNC_TRANSPORT_ERRORS))


def chat_sync(
    *,
    cfg: DeepSeekChatConfig,
    system: Optional[str] = None,
    user: Optional[str] = None,
    messages: Optional[Sequence[dict]] = None,
    max_tokens: int,
    temperature: float = 0,
    json_mode: bool = False,
    disable_thinking: bool = True,
    reasoning_effort: Optional[str] = None,
    timeout_s: Optional[float] = None,
    max_retries: Optional[int] = None,
) -> ChatReply:
    """chat(), blocking, for the callers that already run in a thread.

    The timeout here is httpx's, which bounds each step of the call (connecting, each
    read) rather than the whole of it.
    """
    call = _Call(
        cfg,
        _payload(
            model=cfg.model, system=system, user=user, messages=messages,
            max_tokens=max_tokens, temperature=temperature, json_mode=json_mode,
            disable_thinking=disable_thinking, reasoning_effort=reasoning_effort,
        ),
        timeout_s=timeout_s, max_retries=max_retries,
    )
    while True:
        try:
            with httpx.Client(timeout=call.timeout_s) as client:
                resp = client.post(call.url, json=call.payload, headers=call.headers)
            if resp.status_code >= 400:
                call.failed(resp.status_code, resp.text)
                continue
            return call.read(resp.json())
        except Exception as error:  # noqa: BLE001 — backoff raises what is not worth another try
            time.sleep(call.backoff(error, _SYNC_TRANSPORT_ERRORS))


def _next_budget(reply: ChatReply, budget: int, ceiling: int) -> Optional[int]:
    """The larger budget to ask again with, or None when this reply is the answer."""
    if reply.finish_reason != "length" or budget >= ceiling:
        return None
    larger = min(ceiling, budget * 2)
    logger.warning("DeepSeek reply cut off at %d tokens; asking again with %d", budget, larger)
    return larger


async def chat_json(*, max_tokens: int, max_tokens_ceiling: Optional[int] = None, **kwargs: Any) -> ChatReply:
    """chat() in JSON mode, asked again with double the budget while it is cut off.

    A cut-off JSON answer does not parse, and salvaging one can return an inner object
    that looks like a result. Up to ``max_tokens_ceiling`` (no escalation without one),
    then the last reply is returned as it stands: its ``finish_reason`` still says
    "length", and what that is worth is the caller's to decide. The prompt must ask for
    JSON (see JSON_ONLY); parsing is the caller's too. Takes chat()'s other arguments.
    """
    budget, ceiling = max_tokens, max(max_tokens, max_tokens_ceiling or 0)
    while True:
        reply = await chat(max_tokens=budget, json_mode=True, **kwargs)
        budget = _next_budget(reply, budget, ceiling)
        if budget is None:
            return reply


def chat_json_sync(*, max_tokens: int, max_tokens_ceiling: Optional[int] = None, **kwargs: Any) -> ChatReply:
    """chat_json(), blocking. Takes chat_sync()'s other arguments."""
    budget, ceiling = max_tokens, max(max_tokens, max_tokens_ceiling or 0)
    while True:
        reply = chat_sync(max_tokens=budget, json_mode=True, **kwargs)
        budget = _next_budget(reply, budget, ceiling)
        if budget is None:
            return reply


async def clean_ocr_text(
    raw_text: str,
    *,
    cfg: DeepSeekChatConfig,
    prompt: str = DEFAULT_CLEAN_PROMPT,
    max_output_tokens: int = 3500,
) -> str:
    """
    Sends OCR text to DeepSeek chat to be cleaned.
    Returns cleaned plain text.
    """
    raw_text = (raw_text or "").strip()
    if not raw_text:
        return ""

    # Cleaning OCR is mechanical transcription, not a problem to think about, so the
    # thinking stays off (chat's default). On a full page batch it consumed the entire
    # budget and the answer came back EMPTY ("DeepSeek cleaning returned empty output");
    # turning it off also cut completion tokens ~15x on a sample page.
    text, _finish = await chat(
        cfg=cfg,
        system=prompt,
        user=raw_text,
        max_tokens=max_output_tokens,
    )
    return text
