"""Talari's calls to DeepSeek, through the shared helper in ``shared.ai.deepseek``.

The helper does the HTTP, the retries and the ``thinking`` fallback. What is added here
is what a request handler needs from it: a missing key or model becomes a 503 naming the
variable to set, and an upstream failure becomes a 502 in words written here rather than
the provider's error body, which stays in the log.

The model is ``TALARI_MODEL``, falling back to ``DEEPSEEK_MODEL`` (see shared.ai.models).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import re
from typing import Any

from phansora.shared.ai.deepseek import DeepSeekChatConfig, _chat_completion
from phansora.shared.ai.json_repair import parse_json_loose, repair_truncated_json
from phansora.shared.ai.models import MissingModelConfig

from .errors import NotConfigured, ServiceError

logger = logging.getLogger("talari")

MODEL_VAR = "TALARI_MODEL"
# Someone is waiting on every call here, so fewer retries than the helper's default: a
# request that has failed three times is better reported than kept open.
RETRIES = 2
# A chapter or a long plan is one long reply that is not streamed. The config's default
# timeout would cut it off, and retrying one at that length several times would keep a
# request open for most of an hour, so a long reply gets a long timeout and one retry.
LONG_REPLY_TOKENS = 3000
LONG_TIMEOUT_S = 600
LONG_RETRIES = 1
# The most any one reply may ask for. Tokens are billed as used, so a generous cap costs
# nothing; one set too low ends a chapter mid-sentence.
MAX_OUTPUT_TOKENS = 16000

JSON_INSTRUCTION = "\n\nRespond with valid JSON only. No prose, no markdown fences."

# What an account-level refusal means, in terms of what to go and do about it.
REFUSALS = {
    401: "DeepSeek rejected the API key. Check DEEPSEEK_API_KEY.",
    402: "The DeepSeek account is out of credit.",
    403: "DeepSeek refused access to the configured model. Check TALARI_MODEL / DEEPSEEK_MODEL.",
    429: "DeepSeek is over its rate limit. Try again in a minute.",
}


def config(*, long: bool = False) -> DeepSeekChatConfig:
    """DeepSeek's settings for Talari, or a 503 that names what is missing."""
    try:
        cfg = DeepSeekChatConfig.from_env(product_var=MODEL_VAR)
    except MissingModelConfig:
        raise NotConfigured(
            f"Writing is not available: no DeepSeek model is configured. Set {MODEL_VAR} or DEEPSEEK_MODEL."
        ) from None
    except RuntimeError:
        # from_env's only other failure is the missing key.
        raise NotConfigured("Writing is not available: DEEPSEEK_API_KEY is not set.") from None
    if long:
        return dataclasses.replace(cfg, timeout_s=max(cfg.timeout_s, LONG_TIMEOUT_S), max_retries=LONG_RETRIES)
    return dataclasses.replace(cfg, max_retries=min(cfg.max_retries, RETRIES))


def _failure(error: Exception) -> ServiceError:
    """The caller's message for a failed call. The full error, which can carry the
    provider's response body, goes to the log only."""
    logger.warning("Talari: DeepSeek call failed: %s", error)
    status = re.search(r"HTTP (\d{3})", str(error))
    if status and int(status.group(1)) in REFUSALS:
        return ServiceError(REFUSALS[int(status.group(1))])
    cause = error.__cause__
    # asyncio's TimeoutError is its own class before Python 3.11, which prod runs.
    if isinstance(cause, (TimeoutError, asyncio.TimeoutError)) or "Timeout" in type(cause).__name__:
        return ServiceError("DeepSeek did not answer in time. Try again, or ask for less at once.")
    if status:
        return ServiceError(f"DeepSeek could not answer (HTTP {status.group(1)}). Try again shortly.")
    return ServiceError("DeepSeek could not be reached. Try again shortly.")


async def ask(
    system: str,
    user: str,
    *,
    max_tokens: int,
    temperature: float,
    long: bool | None = None,
) -> str:
    """One reply, as text. Raises ServiceError (502) when there is none.

    ``long`` (the long timeout) follows from ``max_tokens`` unless it is given.
    """
    cfg = config(long=max_tokens > LONG_REPLY_TOKENS if long is None else long)
    try:
        reply = await _chat_completion(
            cfg=cfg,
            system_prompt=system,
            user_text=user,
            max_output_tokens=min(max_tokens, MAX_OUTPUT_TOKENS),
            # Budgets here are sized for the answer. The v4 models reason by default and
            # bill that against max_tokens, which leaves a long answer empty or cut off.
            disable_thinking=True,
            temperature=temperature,
        )
    except RuntimeError as error:
        raise _failure(error) from None
    if not (reply or "").strip():
        raise ServiceError("DeepSeek returned an empty reply.")
    return reply


async def ask_json(
    system: str,
    user: str,
    *,
    max_tokens: int,
    temperature: float,
    long: bool | None = None,
) -> dict[str, Any]:
    """One reply that must be a JSON object.

    A reply that will not parse is first salvaged for its complete part (a cut-off list
    keeps its finished items), then asked for once more; after that it is a 502.
    """
    for attempt in range(2):
        raw = await ask(system + JSON_INSTRUCTION, user, max_tokens=max_tokens, temperature=temperature, long=long)
        for candidate in (raw, repair_truncated_json(raw)):
            if not candidate:
                continue
            try:
                data = parse_json_loose(candidate)
            except ValueError:
                continue
            if isinstance(data, dict):
                return data
        logger.warning("Talari: unreadable JSON from DeepSeek (attempt %d): %.300s", attempt + 1, raw)
    raise ServiceError("DeepSeek's reply could not be read. Try again.")
