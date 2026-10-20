"""DeepSeek chat client for Book Alchemy, over the shared helper in ``shared.ai.deepseek``.

The helper does the HTTP, the retries and the ``thinking`` fallback; this adds:
  - ``chat()``        free-form completion
  - ``chat_to_end()`` a completion that is carried on from where it was cut off
  - ``chat_json()``   structured completion that returns parsed JSON

All calls default to temperature 0 and instruct the model to stay grounded in
the supplied source text — Book Alchemy is a knowledge-transformation system,
not a generator of new content.
"""
from __future__ import annotations

from typing import Any, Optional

from phansora.shared.ai import deepseek
from phansora.shared.ai.deepseek import JSON_ONLY, DeepSeekChatConfig
# The shared JSON salvage, aliased to its historical private names so call sites below
# read unchanged.
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
            timeout_s=timeout_s,
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
                continue_from=said, timeout_s=timeout_s,
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

        If the model reports it was cut off (``finish_reason == "length"``), it is
        asked again with a larger token budget (up to the model cap) before the
        complete part is salvaged — large books can produce long structured output
        that would otherwise truncate into invalid JSON."""
        raw, _finish = await deepseek.chat_json(
            cfg=self.cfg, system=system + JSON_ONLY, user=user,
            max_tokens=max_output_tokens, max_tokens_ceiling=MAX_JSON_TOKENS,
            temperature=temperature,
        )
        try:
            return _parse_json_loose(raw)
        except Exception:  # noqa: BLE001
            # Budget exhausted (or a non-length failure): salvage the complete portion
            # of a (possibly truncated) response rather than failing the whole chunk over
            # a trailing, cut-off item. Repair returns None when nothing complete came
            # through, so this is safe for any failure.
            repaired = _repair_truncated_json(raw)
            if repaired is not None:
                try:
                    # Back through the loose parser, not json.loads: a response can be
                    # both cut off AND carry a literal backslash copied out of the
                    # source, and only the loose parser fixes the second.
                    return _parse_json_loose(repaired)
                except Exception:  # noqa: BLE001, S110 — the original error is raised next
                    pass
            raise

    async def _completion(
        self, *, system: str, user: str, max_output_tokens: int,
        temperature: float, continue_from: Optional[str] = None,
        timeout_s: Optional[float] = None,
    ) -> tuple[str, Optional[str]]:
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
        # The thinking stays off (the helper's default). Every budget here is sized for
        # the ANSWER only — a session script is `words * 1.7 + 400`, about 1000 tokens —
        # and the v4 models bill their reasoning against max_tokens, so it consumed the
        # whole allowance and the content came back EMPTY. An empty script is not an
        # error anywhere downstream: the session is simply written blank and the audio
        # phase skips it, which is how a course completed with no MP3s and no failure.
        return await deepseek.chat(
            cfg=self.cfg, messages=messages, max_tokens=max_output_tokens,
            temperature=temperature, timeout_s=timeout_s,
        )
