# src/services/deepseek_cleaner.py
#
# Raw OCR text -> cleaned, readable book text using DeepSeek Chat (text-only endpoint)
#
# Env vars (via .env):
#   DEEPSEEK_CHAT_BASE_URL=https://api.deepseek.com
#   DEEPSEEK_CHAT_API_KEY=...
#   DEEPSEEK_CHAT_MODEL=deepseek-chat   (or whatever your provider uses)
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
        default_model: str = "deepseek-chat",
    ) -> "DeepSeekChatConfig":
        base_url = os.getenv("DEEPSEEK_CHAT_BASE_URL", default_base_url).rstrip("/")
        api_key = os.getenv("DEEPSEEK_CHAT_API_KEY", "").strip()
        model = os.getenv("DEEPSEEK_CHAT_MODEL", default_model).strip()

        if not api_key:
            raise RuntimeError(
                "Missing DEEPSEEK_CHAT_API_KEY environment variable. Set it in .env."
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
    )


async def _chat_completion(
    *,
    cfg: DeepSeekChatConfig,
    system_prompt: str,
    user_text: str,
    max_output_tokens: int,
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
