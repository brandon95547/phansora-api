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
import json
import random
import re
from typing import Any, Optional

import aiohttp

from phansora.shared.ai.deepseek import DeepSeekChatConfig  # reuse existing config/env

# DeepSeek chat caps output at 8192 tokens; we escalate JSON budgets up to here
# when a response is truncated.
MAX_JSON_TOKENS = 8000


class DeepSeekClient:
    def __init__(self, cfg: Optional[DeepSeekChatConfig] = None) -> None:
        self.cfg = cfg or DeepSeekChatConfig.from_env()

    @classmethod
    def from_env(cls) -> "DeepSeekClient":
        return cls(DeepSeekChatConfig.from_env())

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
                # Budget exhausted (or a non-length failure): salvage the
                # complete portion of a truncated response rather than failing
                # the whole chunk over the trailing, cut-off item.
                if finish == "length":
                    repaired = _repair_truncated_json(raw)
                    if repaired is not None:
                        try:
                            return json.loads(repaired)
                        except json.JSONDecodeError:
                            pass
                raise
        raise last_err  # pragma: no cover

    async def _completion(
        self, *, system: str, user: str, max_output_tokens: int,
        temperature: float, json_mode: bool,
    ) -> tuple[str, Optional[str]]:
        cfg = self.cfg
        url = f"{cfg.base_url}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": cfg.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_output_tokens,
            "stream": False,
        }
        if json_mode:
            # DeepSeek supports OpenAI-style JSON mode; harmless if ignored.
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=cfg.timeout_s)
        last_err: Optional[Exception] = None

        for attempt in range(cfg.max_retries + 1):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=payload, headers=headers) as resp:
                        if resp.status >= 400:
                            body = await resp.text()
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


def _parse_json_loose(raw: str) -> Any:
    """Best-effort JSON extraction from a model response."""
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Empty DeepSeek JSON response.")
    # Strip ```json ... ``` fences if present.
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced { } or [ ] span.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = raw.find(open_ch)
        end = raw.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from DeepSeek response: {raw[:300]}")


def _repair_truncated_json(raw: str) -> Optional[str]:
    """Salvage a JSON doc that was cut off mid-generation (finish=='length').

    Walks the text tracking string/escape and bracket state, rewinds to the last
    point where a container ({…} or […]) closed cleanly, and appends closers for
    whatever is still open. For Book Alchemy's schema — an object of arrays of
    ``{title, body}`` objects — this keeps every complete item and drops only the
    trailing, cut-off one. Returns a parseable string, or None if nothing
    complete came through."""
    raw = (raw or "").strip()
    fenced = re.match(r"^```(?:json)?\s*(.*)$", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1)
    start = raw.find("{")
    alt = raw.find("[")
    if start == -1 or (alt != -1 and alt < start):
        start = alt
    if start == -1:
        return None

    stack: list[str] = []
    in_string = False
    escape = False
    cut: Optional[int] = None          # index (exclusive) of a safe truncation point
    cut_stack: tuple[str, ...] = ()     # containers still open at that point

    for i in range(start, len(raw)):
        ch = raw[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if not stack:
                break  # unbalanced close; best guess is everything before it
            stack.pop()
            cut, cut_stack = i + 1, tuple(stack)

    if cut is None:
        return None
    return raw[start:cut] + "".join(reversed(cut_stack))
