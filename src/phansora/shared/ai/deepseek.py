# src/services/deepseek_cleaner.py
#
# Raw OCR text -> cleaned, readable book text using DeepSeek Chat (text-only endpoint)
#
# Env vars (via .env):
#   DEEPSEEK_CHAT_BASE_URL=https://api.deepseek.com
#   DEEPSEEK_CHAT_API_KEY=...
#   DEEPSEEK_CHAT_MODEL=deepseek-v4-flash   (or whatever your provider uses)
#
# Python deps:
#   python -m pip install aiohttp python-dotenv

from __future__ import annotations

import asyncio
import os
import random
from dataclasses import dataclass
from typing import Optional

import aiohttp

from .models import resolve_model


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


def chat_model(product_var: str | None = None) -> str:
    """The DeepSeek model this deployment should use, for an optional product override.

    Single source of truth for callers that talk to DeepSeek through an OpenAI-compatible
    client and so don't build a DeepSeekChatConfig. Hardcoding the name at call sites is what
    left Dossier Nova broken when the `deepseek-chat` alias was retired — the .env everything
    else reads couldn't reach it.
    """
    return resolve_model(product_var, provider="deepseek")


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
        # Canonical vars are DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL / DEEPSEEK_MODEL
        # (shared with Dossier Nova). The legacy DEEPSEEK_CHAT_* names are still
        # honored as a fallback so existing deployments keep working.
        base_url = (
            os.getenv("DEEPSEEK_BASE_URL")
            or os.getenv("DEEPSEEK_CHAT_BASE_URL")
            or default_base_url
        ).rstrip("/")
        api_key = (
            os.getenv("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_CHAT_API_KEY") or ""
        ).strip()
        model = resolve_model(product_var, provider="deepseek")

        if not api_key:
            raise RuntimeError(
                "Missing DEEPSEEK_API_KEY environment variable. Set it in .env."
            )

        return DeepSeekChatConfig(base_url=base_url, api_key=api_key, model=model)


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

    return await _chat_completion(
        cfg=cfg,
        system_prompt=prompt,
        user_text=raw_text,
        max_output_tokens=max_output_tokens,
        # Cleaning OCR is mechanical transcription, not a problem to think about. The v4
        # models reason by default and those tokens are billed against max_tokens — on a
        # full page batch they consumed the entire budget and the answer came back EMPTY
        # ("DeepSeek cleaning returned empty output"). Disabling it also cut completion
        # tokens ~15x on a sample page.
        disable_thinking=True,
    )


async def _chat_completion(
    *,
    cfg: DeepSeekChatConfig,
    system_prompt: str,
    user_text: str,
    max_output_tokens: int,
    disable_thinking: bool = False,
) -> str:
    url = f"{cfg.base_url}/v1/chat/completions"
    payload = {
        "model": cfg.model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "max_tokens": max_output_tokens,
        "stream": False,
    }
    if disable_thinking:
        payload["thinking"] = {"type": "disabled"}
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
                        # `thinking` is a v4-era parameter. If the configured model doesn't
                        # know it, drop it and retry rather than failing the whole job —
                        # model names change under us (the model comes from .env).
                        if resp.status == 400 and "thinking" in body and "thinking" in payload:
                            payload.pop("thinking", None)
                            continue
                        raise RuntimeError(f"DeepSeek chat HTTP {resp.status}: {body[:800]}")
                    data = await resp.json()

            choices = data.get("choices") or []
            if not choices:
                return ""
            content = (choices[0].get("message") or {}).get("content") or ""
            return (content or "").strip()

        except Exception as e:
            last_err = e
            if attempt >= cfg.max_retries:
                break
            sleep_s = min(
                cfg.max_retry_sleep_s,
                cfg.min_retry_sleep_s * (2 ** attempt) + random.random() * 0.25,
            )
            await asyncio.sleep(sleep_s)

    raise RuntimeError(f"DeepSeek clean failed after retries: {last_err}") from last_err
