"""LLM helper for Narrava Studio.

Provider-neutral ``generate_text`` / ``generate_json`` used to write narration
scripts and to pick media search queries. Mirrors Chrono-Origin's provider switch
(NARRAVA_LLM_PROVIDER: openai | deepseek); default is GPT-5 Nano because it is the
cheapest capable option, with DeepSeek as the fallback. Both clients are imported
lazily so the product still mounts on a host that only has one provider installed.

All calls are synchronous (the OpenAI SDK path is blocking); callers run them in a
thread executor so the FastAPI event loop stays free.
"""
from __future__ import annotations

import os
from typing import Any, Dict

from phansora.shared.ai.json_repair import parse_json_loose
from phansora.shared.ai.retries import REFUSAL_STATUSES, ProviderError, refusal_message

from .. import config


class ProviderRejected(RuntimeError):
    """The provider answered, and what it said was "no".

    Auth, billing and quota — 401, 402, 403, 429. Kept apart from every other failure
    because the remedy is an ACCOUNT action by whoever runs the service, not a retry, a
    reword or a smaller budget; and because the callers that degrade gracefully must not
    degrade for this one.

    Folded in with the rest, a 402 reached the editor as "Could not write more ideas for
    this scene" while the storyboard quietly fell back to grammar-split placeholders — an
    answer nobody could tell from a real one without reading the service log.
    """

    def __init__(self, message: str, *, status: int, provider: str) -> None:
        super().__init__(message)
        self.status = status
        self.provider = provider


def _rejected(status: int, provider: str) -> ProviderRejected:
    """Each refusal in the only terms that help: what to go and do about it (see
    shared.ai.retries.refusal_message), naming the variables this product reads."""
    key_var, model_var = (
        ("DEEPSEEK_API_KEY", "DEEPSEEK_MODEL") if provider == "DeepSeek" else ("OPENAI_API_KEY", "OPENAI_MODEL")
    )
    message = refusal_message(
        status, provider=provider, key_var=key_var, model_vars=f"NARRAVA_MODEL / {model_var}"
    ) or f"The {provider} API refused the request ({status})."
    return ProviderRejected(message, status=status, provider=provider)


def _provider() -> str:
    return config.get_settings().provider


def provider_configured() -> bool:
    """Is the active provider's API key present?"""
    if _provider() == "deepseek":
        return deepseek_configured()
    return bool(os.getenv("OPENAI_API_KEY"))


def required_key_name() -> str:
    from phansora.shared.ai.deepseek import KEY_VAR

    return KEY_VAR if _provider() == "deepseek" else "OPENAI_API_KEY"


# ── OpenAI (GPT-5 Nano) ──────────────────────────────────────────────────────
def _openai_model() -> str:
    # NARRAVA_MODEL > OPENAI_MODEL > default (see phansora.shared.ai.models).
    from phansora.shared.ai.models import resolve_model

    return resolve_model("NARRAVA_MODEL", provider="openai")


def _openai_text(system: str, user: str, *, max_output_tokens: int, json_mode: bool) -> str:
    from openai import OpenAI  # lazy: keep the product importable without the SDK

    client = OpenAI(api_key=(os.getenv("OPENAI_API_KEY") or "").strip(), timeout=120)
    kwargs: Dict[str, Any] = {
        "model": _openai_model(),
        "input": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        # GPT-5 Nano is a reasoning model; keep effort low to stay cheap.
        "reasoning": {"effort": "low"},
        "max_output_tokens": max_output_tokens,
    }
    if json_mode:
        kwargs["text"] = {"format": {"type": "json_object"}}
    try:
        resp = client.responses.create(**kwargs)
    except Exception as exc:  # noqa: BLE001 — re-raised unless it is a refusal
        # Every openai error class carries the HTTP status; matching on that rather than on
        # class names keeps this working across SDK versions.
        status = getattr(exc, "status_code", None)
        if status in REFUSAL_STATUSES:
            raise _rejected(status, "OpenAI") from exc
        raise
    return (getattr(resp, "output_text", "") or "").strip()


# ── DeepSeek ─────────────────────────────────────────────────────────────────
def _deepseek_text(system: str, user: str, *, max_output_tokens: int, json_mode: bool) -> str:
    from phansora.shared.ai.deepseek import DeepSeekChatConfig, chat_sync

    cfg = DeepSeekChatConfig.from_env(product_var="NARRAVA_MODEL")
    try:
        # The thinking stays off (the helper's default). The v4 models think by default
        # and those tokens are billed against `max_tokens` — our budgets here are sized
        # for the answer alone (2500 for a script, ~1500+ for a storyboard), so leaving it
        # on returns empty or truncated JSON. The openai path keeps effort low for the
        # same reason.
        text, _finish = chat_sync(
            cfg=cfg,
            system=system,
            user=user,
            max_tokens=max_output_tokens,
            temperature=0.7 if not json_mode else 0,
            json_mode=json_mode,
            # One attempt, as this has always made: someone in the editor is waiting on
            # it, and the callers that can do without an answer already degrade.
            max_retries=0,
        )
    except ProviderError as exc:
        # A refusal is not a model problem at all, and must not reach the callers as one:
        # every caller treats any other failure as "the model misbehaved".
        if exc.status in REFUSAL_STATUSES:
            raise _rejected(exc.status, "DeepSeek") from exc
        raise
    return text


# ── Public surface ───────────────────────────────────────────────────────────
def deepseek_configured() -> bool:
    """Is DeepSeek reachable, regardless of which provider the switch points at?"""
    from phansora.shared.ai.deepseek import key_configured

    return key_configured()


def deepseek_text(system: str, user: str, *, max_output_tokens: int = 2000) -> str:
    """Always DeepSeek, whatever NARRAVA_LLM_PROVIDER says.

    Writing a script is a creative job and follows the provider switch. Polishing prose
    someone already wrote is not the same job: it is cheap, high-volume and asked for by
    name, so it is pinned rather than left to follow a setting that exists for generation.
    """
    return _deepseek_text(system, user, max_output_tokens=max_output_tokens, json_mode=False)


def deepseek_json(system: str, user: str, *, max_output_tokens: int = 2000) -> Any:
    """Always DeepSeek, in JSON mode: the narration conversation and a document's reading.

    Pinned for the same reason as deepseek_text: these are asked for by name, and the
    conversation in particular is a back-and-forth someone is waiting on, not a piece of
    writing the provider switch exists for. The answer as parsed JSON, or {} (_parse_json).
    """
    return _parse_json(_deepseek_text(system, user, max_output_tokens=max_output_tokens, json_mode=True))


def generate_text(system: str, user: str, *, max_output_tokens: int = 2000) -> str:
    if _provider() == "deepseek":
        return _deepseek_text(system, user, max_output_tokens=max_output_tokens, json_mode=False)
    return _openai_text(system, user, max_output_tokens=max_output_tokens, json_mode=False)


def generate_json(system: str, user: str, *, max_output_tokens: int = 2000) -> Dict[str, Any]:
    if _provider() == "deepseek":
        raw = _deepseek_text(system, user, max_output_tokens=max_output_tokens, json_mode=True)
    else:
        raw = _openai_text(system, user, max_output_tokens=max_output_tokens, json_mode=True)
    return _parse_json(raw)


def _parse_json(raw: str) -> Any:
    """The answer as JSON, or {} when there is none to read.

    The shared salvage (fences, prose around the object, stray backslashes). Not narrowed
    to a dict: the callers check the shape they need themselves.
    """
    try:
        return parse_json_loose(raw)
    except ValueError:
        return {}
