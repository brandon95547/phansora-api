"""Environment-driven configuration for Narrava Studio."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── LLM ──────────────────────────────────────────────────────────────────
    # Which provider writes scripts / picks media queries. Mirrors Chrono-Origin's
    # switch: default GPT-5 Nano (cheapest capable model), DeepSeek as a fallback.
    # Provider keys (OPENAI_API_KEY / DEEPSEEK_API_KEY) are read directly by the
    # clients in services/llm.py — none are stored here.
    narrava_llm_provider: str = "openai"  # openai | deepseek

    # ── Narration timing ────────────────────────────────────────────────────
    # Words-per-minute used to estimate how long each script beat takes to read,
    # which is what places media clips at the right spot on the timeline. 150 wpm
    # is a natural documentary-narrator pace.
    narrava_words_per_minute: int = 150

    # ── Media sourcing ───────────────────────────────────────────────────────
    # Default provider returns openly/CC-licensed media with attribution so the
    # preliminary timeline respects fair use. Openverse needs no API key.
    narrava_media_provider: str = "openverse"  # openverse | placeholder
    narrava_media_per_segment: int = 1
    narrava_media_timeout_s: int = 15

    # ── CORS ─────────────────────────────────────────────────────────────────
    cors_allow_origins: str = "*"

    @property
    def provider(self) -> str:
        return (os.getenv("NARRAVA_LLM_PROVIDER") or self.narrava_llm_provider).strip().lower()

    @property
    def cors_origins_list(self) -> List[str]:
        if not self.cors_allow_origins or self.cors_allow_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
