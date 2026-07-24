"""DeepSeek client that can drive a *reasoning* model (R1 / ``deepseek-reasoner``).

The reasoning models are not drop-in replacements for ``deepseek-chat``. Three differences
matter enough to break a naive port:

  1. **No JSON mode.** ``response_format={"type":"json_object"}`` is unsupported. Structured
     output has to be requested in the prompt and parsed leniently on the way back — which
     is exactly what ``shared/ai/json_repair`` exists for.
  2. **No sampling knobs.** ``temperature`` / ``top_p`` / penalties are not supported and
     sending them is at best ignored, at worst a 400.
  3. **Chain-of-thought is a separate field.** The answer is ``message.content``; the
     model's thinking is ``message.reasoning_content`` and must never be concatenated into
     the answer or fed back as conversation history.

So the request is built from the model *kind*, not from a fixed template. Non-reasoning
models keep JSON mode and temperature; reasoning models drop both and lean on the parser.

Config (base URL, key) is shared with the rest of the app via ``DeepSeekChatConfig``; the
model is chosen separately so one process can use a cheap chat model for bulk work and a
reasoning model for the calls where judgement actually matters.
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import Any, Optional

import aiohttp

from .deepseek import DeepSeekChatConfig
from .json_repair import parse_json_loose, repair_truncated_json

logger = logging.getLogger("phansora.ai.deepseek")

# Output cap. For reasoning models this bounds the ANSWER only — chain-of-thought tokens
# are billed and generated separately, so a JSON answer does not need extra headroom here.
MAX_OUTPUT_TOKENS = 8000

DEFAULT_CHAT_MODEL = "deepseek-chat"
DEFAULT_REASONING_MODEL = "deepseek-reasoner"

_JSON_INSTRUCTION = "\n\nRespond with valid JSON only. No prose, no explanation, no markdown fences."


def is_reasoning_model(model: str) -> bool:
    """Whether `model` is one of the chain-of-thought models.

    Matched by name because the API exposes no capability flag. Deliberately loose: a
    future ``deepseek-reasoner-v2`` or an ``-r1`` variant should be recognised, and the
    cost of a false positive (losing JSON mode, which we can parse around) is far lower
    than a false negative (sending an unsupported parameter and getting a 400).
    """
    name = (model or "").lower()
    return "reason" in name or "-r1" in name or name.endswith("r1")


class DeepSeekReasoner:
    """Chat/JSON completions against a chosen DeepSeek model, reasoning-aware."""

    def __init__(self, cfg: Optional[DeepSeekChatConfig] = None, model: Optional[str] = None) -> None:
        self.cfg = cfg or DeepSeekChatConfig.from_env()
        self.model = (model or self.cfg.model or DEFAULT_CHAT_MODEL).strip()

    @classmethod
    def reasoning(cls) -> "DeepSeekReasoner":
        """The judgement model — ranking chapters, writing a script."""
        return cls(model=os.getenv("DEEPSEEK_REASONING_MODEL", DEFAULT_REASONING_MODEL))

    @classmethod
    def fast(cls) -> "DeepSeekReasoner":
        """The bulk model — summarising fifty chapters, where reasoning buys nothing and
        would cost minutes per book."""
        return cls(model=os.getenv("DEEPSEEK_MODEL", DEFAULT_CHAT_MODEL))

    @property
    def reasons(self) -> bool:
        return is_reasoning_model(self.model)

    async def chat(self, *, system: str, user: str, max_output_tokens: int = 4000) -> str:
        content, _ = await self._completion(
            system=system, user=user, max_output_tokens=max_output_tokens, json_mode=False,
        )
        return content

    async def chat_json(self, *, system: str, user: str, max_output_tokens: int = 4000) -> Any:
        """Completion that must parse as JSON.

        A truncated response (``finish_reason == "length"``) is retried with a bigger
        budget, then salvaged for its complete portion rather than thrown away — a long
        chapter list should not be lost over one cut-off trailing item.
        """
        budget = max_output_tokens
        sys_prompt = system + _JSON_INSTRUCTION
        last_err: Optional[Exception] = None

        for _ in range(3):
            raw, finish = await self._completion(
                system=sys_prompt, user=user, max_output_tokens=budget, json_mode=True,
            )
            try:
                return parse_json_loose(raw)
            except Exception as exc:  # noqa: BLE001 — any parse failure gets the same salvage
                last_err = exc
                if finish == "length" and budget < MAX_OUTPUT_TOKENS:
                    budget = min(MAX_OUTPUT_TOKENS, budget * 2)
                    continue
                repaired = repair_truncated_json(raw)
                if repaired is not None:
                    try:
                        return parse_json_loose(repaired)
                    except Exception:  # noqa: BLE001
                        pass
                raise
        raise last_err  # pragma: no cover

    # ------------------------------------------------------------------ internals
    def build_payload(self, *, system: str, user: str, max_output_tokens: int, json_mode: bool) -> dict:
        """Exposed (not private) so tests can assert the reasoning-model shape without
        making a network call — the whole point of this class is the payload."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_output_tokens,
            "stream": False,
        }
        if not self.reasons:
            # Unsupported on reasoning models — see the module docstring.
            payload["temperature"] = 0.0
            if json_mode:
                payload["response_format"] = {"type": "json_object"}
        return payload

    async def _completion(
        self, *, system: str, user: str, max_output_tokens: int, json_mode: bool,
    ) -> tuple[str, Optional[str]]:
        cfg = self.cfg
        url = f"{cfg.base_url}/v1/chat/completions"
        payload = self.build_payload(
            system=system, user=user, max_output_tokens=max_output_tokens, json_mode=json_mode,
        )
        headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
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
                message = choices[0].get("message") or {}
                # reasoning_content is the model thinking aloud. It is never part of the
                # answer and must not be fed back as history.
                if message.get("reasoning_content"):
                    logger.debug("%s reasoning: %d chars", self.model, len(message["reasoning_content"]))
                return (message.get("content") or "").strip(), choices[0].get("finish_reason")
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt >= cfg.max_retries:
                    break
                sleep_s = min(
                    cfg.max_retry_sleep_s,
                    cfg.min_retry_sleep_s * (2 ** attempt) + random.random() * 0.25,
                )
                await asyncio.sleep(sleep_s)

        raise RuntimeError(f"DeepSeek call failed after retries: {last_err}") from last_err
