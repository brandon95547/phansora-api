"""Talari's calls to DeepSeek, through the shared helper in ``shared.ai.deepseek``.

The helper does the HTTP, the retries and the ``thinking`` fallback. What is added here
is what a request handler needs from it: a missing key or model becomes a 503 naming the
variable to set, and an upstream failure becomes a 502 in words written here rather than
the provider's error body, which stays in the log.

The model is ``TALARI_MODEL``, falling back to ``DEEPSEEK_MODEL`` (see shared.ai.models).
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from phansora.shared.ai.deepseek import JSON_ONLY, KEY_VAR, DeepSeekChatConfig, chat
from phansora.shared.ai.json_repair import parse_json_loose, repair_truncated_json
from phansora.shared.ai.models import MissingModelConfig
from phansora.shared.ai.retries import refusal_message

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

JSON_INSTRUCTION = JSON_ONLY


def config() -> DeepSeekChatConfig:
    """DeepSeek's settings for Talari, or a 503 that names what is missing."""
    try:
        return DeepSeekChatConfig.from_env(product_var=MODEL_VAR)
    except MissingModelConfig:
        raise NotConfigured(
            f"Writing is not available: no DeepSeek model is configured. Set {MODEL_VAR} or DEEPSEEK_MODEL."
        ) from None
    except RuntimeError:
        # from_env's only other failure is the missing key.
        raise NotConfigured(f"Writing is not available: {KEY_VAR} is not set.") from None


def refusal(status: int | None) -> str | None:
    """What an account-level refusal means, in terms of what to go and do about it."""
    return refusal_message(status, provider="DeepSeek", key_var=KEY_VAR, model_vars=f"{MODEL_VAR} / DEEPSEEK_MODEL")


def _failure(error: Exception) -> ServiceError:
    """The caller's message for a failed call. The full error, which can carry the
    provider's response body, goes to the log only."""
    logger.warning("Talari: DeepSeek call failed: %s", error)
    # The shared helper's error carries the status; the message is the fallback for
    # anything that raised a plain RuntimeError.
    status = getattr(error, "status", None)
    if status is None:
        found = re.search(r"HTTP (\d{3})", str(error))
        status = int(found.group(1)) if found else None
    if refusal(status):
        return ServiceError(refusal(status))
    cause = error.__cause__
    # asyncio's TimeoutError is its own class before Python 3.11, which prod runs.
    if isinstance(cause, (TimeoutError, asyncio.TimeoutError)) or "Timeout" in type(cause).__name__:
        return ServiceError("DeepSeek did not answer in time. Try again, or ask for less at once.")
    if status:
        return ServiceError(f"DeepSeek could not answer (HTTP {status}). Try again shortly.")
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
    cfg = config()
    if long is None:
        long = max_tokens > LONG_REPLY_TOKENS
    if long:
        timeout_s, retries = max(cfg.timeout_s, LONG_TIMEOUT_S), LONG_RETRIES
    else:
        timeout_s, retries = cfg.timeout_s, min(cfg.max_retries, RETRIES)
    try:
        # Budgets here are sized for the answer, so the thinking stays off (chat's
        # default): the v4 models reason by default and bill that against max_tokens,
        # which leaves a long answer empty or cut off.
        reply, _finish = await chat(
            cfg=cfg,
            system=system,
            user=user,
            max_tokens=min(max_tokens, MAX_OUTPUT_TOKENS),
            temperature=temperature,
            timeout_s=timeout_s,
            max_retries=retries,
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
