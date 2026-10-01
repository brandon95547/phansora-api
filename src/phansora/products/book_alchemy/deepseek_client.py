"""DeepSeek chat client for Book Alchemy.

Reuses the config + env vars from the existing OCR cleaner
(``services.deepseek_cleaner.DeepSeekChatConfig``) and adds:
  - ``chat()``      free-form completion
  - ``chat_json()`` structured completion that returns parsed JSON

All calls default to temperature 0 and instruct the model to stay grounded in
the supplied source text — Book Alchemy is a knowledge-transformation system,
not a generator of new content.
"""
from __future__ import annotations

import asyncio
import random
from typing import Any, Optional

import aiohttp

from phansora.shared.ai.deepseek import DeepSeekChatConfig  # reuse existing config/env
# Shared with the reasoning-model client (shared/ai/deepseek_reasoner.py) — aliased to
# their historical private names so call sites below read unchanged.
from phansora.shared.ai.json_repair import (
    parse_json_loose as _parse_json_loose,
    repair_truncated_json as _repair_truncated_json,
)

from .prompts import CONTINUE_USER
from .sentences import last_sentence_end

# DeepSeek chat caps output at 8192 tokens; we escalate JSON budgets up to here
# when a response is truncated.
MAX_JSON_TOKENS = 8000
# The same ceiling, for prose. A reply that has to FINISH (a lesson script) is
# given all of it: tokens are billed as used, so a generous cap costs nothing,
# while a tight one ends the reply mid-sentence.
MAX_OUTPUT_TOKENS = MAX_JSON_TOKENS
# How many times a reply cut off at the ceiling is asked to carry on.
MAX_CONTINUATIONS = 2
# A reply that fills the ceiling takes minutes to generate, and the request is not
# streamed — the config's 180s would time out the longest lessons, retry them into
# the same timeout, and fail the book.
LONG_REPLY_TIMEOUT_S = 600


class DeepSeekClient:
    def __init__(self, cfg: Optional[DeepSeekChatConfig] = None) -> None:
        self.cfg = cfg or DeepSeekChatConfig.from_env(product_var="BOOK_ALCHEMY_MODEL")

    @classmethod
    def from_env(cls) -> "DeepSeekClient":
        return cls(DeepSeekChatConfig.from_env(product_var="BOOK_ALCHEMY_MODEL"))

    async def chat(
        self,
        *,
        system: str,
        user: str,
        max_output_tokens: int = 4000,
        temperature: float = 0.0,
    ) -> str:
        content, _ = await self._completion(
            system=system, user=user,
            max_output_tokens=max_output_tokens, temperature=temperature,
            json_mode=False,
        )
        return content

    async def chat_to_end(
        self,
        *,
        system: str,
        user: str,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        temperature: float = 0.0,
    ) -> tuple[str, bool]:
        """Free-form completion that is allowed to finish. Returns ``(text, finished)``.

        ``chat`` throws ``finish_reason`` away, so a reply that hit ``max_tokens``
        came back looking like any other — and a lesson script cut off that way
        was recorded exactly as it stood, ending mid-sentence. Here a cut-off
        reply is trimmed to its last finished sentence and the model is asked to
        carry on from there, up to MAX_CONTINUATIONS times.

        ``finished`` is False only when the reply was still cut off after that.
        The text is returned either way; the caller decides what a ragged ending
        is worth.
        """
        timeout_s = max(self.cfg.timeout_s, LONG_REPLY_TIMEOUT_S)
        text, finish = await self._completion(
            system=system, user=user,
            max_output_tokens=max_output_tokens, temperature=temperature,
            json_mode=False, timeout_s=timeout_s,
        )
        for _ in range(MAX_CONTINUATIONS):
            if finish != "length":
                break
            cut = last_sentence_end(text)
            if cut <= 0:
                break   # not one finished sentence to resume from
            said, rest = text[:cut], text[cut:]
            more, finish = await self._completion(
                system=system, user=user,
                max_output_tokens=max_output_tokens, temperature=temperature,
                json_mode=False, continue_from=said, timeout_s=timeout_s,
            )
            if not more:
                finish = "length"
                break
            # Keep the paragraph break if the fragment that was dropped had begun one.
            gap = rest[: len(rest) - len(rest.lstrip())]
            text = said + ("\n\n" if "\n" in gap else " ") + more
        return text, finish != "length"

    async def chat_json(
        self,
        *,
        system: str,
        user: str,
        max_output_tokens: int = 4000,
        temperature: float = 0.0,
    ) -> Any:
        """Completion that must return JSON.

        If the model reports it was cut off (``finish_reason == "length"``) and
        the JSON won't parse, retry with a larger token budget (up to the model
        cap) before giving up — large books can produce long structured output
        that would otherwise truncate into invalid JSON."""
        budget = max_output_tokens
        last_err: Optional[Exception] = None
        sys_prompt = system + "\n\nRespond with valid JSON only. No prose, no markdown fences."
        for _ in range(3):
            raw, finish = await self._completion(
                system=sys_prompt, user=user,
                max_output_tokens=budget, temperature=temperature,
                json_mode=True,
            )
            try:
                return _parse_json_loose(raw)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if finish == "length" and budget < MAX_JSON_TOKENS:
                    budget = min(MAX_JSON_TOKENS, budget * 2)
                    continue
                # Budget exhausted (or a non-length failure): salvage the complete
                # portion of a (possibly truncated) response rather than failing the
                # whole chunk over a trailing, cut-off item. Repair returns None when
                # nothing complete came through, so this is safe for any failure.
                repaired = _repair_truncated_json(raw)
                if repaired is not None:
                    try:
                        # Back through the loose parser, not json.loads: a response can
                        # be both cut off AND carry a literal backslash copied out of the
                        # source, and only the loose parser fixes the second.
                        return _parse_json_loose(repaired)
                    except Exception:  # noqa: BLE001
                        pass
                raise
        raise last_err  # pragma: no cover

    async def _completion(
        self, *, system: str, user: str, max_output_tokens: int,
        temperature: float, json_mode: bool, continue_from: Optional[str] = None,
        timeout_s: Optional[float] = None,
    ) -> tuple[str, Optional[str]]:
        cfg = self.cfg
        url = f"{cfg.base_url}/v1/chat/completions"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if continue_from:
            # The reply so far, handed back as the model's own turn, then the ask
            # to carry on. See chat_to_end.
            messages += [
                {"role": "assistant", "content": continue_from},
                {"role": "user", "content": CONTINUE_USER},
            ]
        payload: dict[str, Any] = {
            "model": cfg.model,
            "temperature": temperature,
            "messages": messages,
            "max_tokens": max_output_tokens,
            "stream": False,
            # The v4 models reason by default and bill those tokens against max_tokens.
            # Every budget here is sized for the ANSWER only — a session script is
            # `words * 1.7 + 400`, about 1000 tokens — so reasoning consumed the whole
            # allowance and the content came back EMPTY. An empty script is not an error
            # anywhere downstream: the session is simply written blank and the audio phase
            # skips it, which is how a course completed with no MP3s and no failure.
            "thinking": {"type": "disabled"},
        }
        if json_mode:
            # DeepSeek supports OpenAI-style JSON mode; harmless if ignored.
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=timeout_s or cfg.timeout_s)
        last_err: Optional[Exception] = None

        for attempt in range(cfg.max_retries + 1):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=payload, headers=headers) as resp:
                        if resp.status >= 400:
                            body = await resp.text()
                            # `thinking` is a v4-era parameter; if the configured model
                            # rejects it, drop it and retry rather than failing the job.
                            if resp.status == 400 and "thinking" in body and "thinking" in payload:
                                payload.pop("thinking", None)
                                continue
                            raise RuntimeError(f"DeepSeek HTTP {resp.status}: {body[:800]}")
                        data = await resp.json()
                choices = data.get("choices") or []
                if not choices:
                    return "", None
                content = ((choices[0].get("message") or {}).get("content") or "").strip()
                finish_reason = choices[0].get("finish_reason")
                return content, finish_reason
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt >= cfg.max_retries:
                    break
                sleep_s = min(
                    cfg.max_retry_sleep_s,
                    cfg.min_retry_sleep_s * (2 ** attempt) + random.random() * 0.25,
                )
                await asyncio.sleep(sleep_s)

        raise RuntimeError(f"DeepSeek call failed after retries: {last_err}") from last_err
