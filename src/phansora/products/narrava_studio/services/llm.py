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

import json
import os
from typing import Any, Dict

from .. import config


def _provider() -> str:
    return config.get_settings().provider


def provider_configured() -> bool:
    """Is the active provider's API key present?"""
    if _provider() == "deepseek":
        return bool(os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_CHAT_API_KEY"))
    return bool(os.getenv("OPENAI_API_KEY"))


def required_key_name() -> str:
    return "DEEPSEEK_API_KEY" if _provider() == "deepseek" else "OPENAI_API_KEY"


# ── OpenAI (GPT-5 Nano) ──────────────────────────────────────────────────────
def _openai_model() -> str:
    return (os.getenv("OPENAI_MODEL") or "gpt-5-nano").split("#", 1)[0].strip() or "gpt-5-nano"


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
    resp = client.responses.create(**kwargs)
    return (getattr(resp, "output_text", "") or "").strip()


# ── DeepSeek ─────────────────────────────────────────────────────────────────
def _deepseek_text(system: str, user: str, *, max_output_tokens: int, json_mode: bool) -> str:
    import httpx

    from phansora.shared.ai.deepseek import DeepSeekChatConfig

    cfg = DeepSeekChatConfig.from_env()
    payload: Dict[str, Any] = {
        "model": cfg.model,
        "temperature": 0.7 if not json_mode else 0,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_output_tokens,
        "stream": False,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    resp = httpx.post(
        f"{cfg.base_url}/v1/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"},
        timeout=cfg.timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        return ""
    return ((choices[0].get("message") or {}).get("content") or "").strip()


# ── Public surface ───────────────────────────────────────────────────────────
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


def _parse_json(raw: str) -> Dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return {}
        return {}
