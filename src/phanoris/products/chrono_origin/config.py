"""Environment-driven configuration."""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Anthropic (Claude) — grounded web search + JSON reasoning
    anthropic_api_key: str = ""
    anthropic_search_model: str = "claude-sonnet-4-6"
    anthropic_reasoning_model: str = "claude-opus-4-8"
    anthropic_search_max_tokens: int = 4096
    anthropic_reason_max_tokens: int = 16000

    # Pipeline limits
    chrono_max_depth: int = 4
    chrono_max_sources_per_stage: int = 8
    chrono_max_queries_per_stage: int = 5
    # Web searches Claude may run per grounded-search call (cost cap).
    chrono_search_max_uses: int = 4
    chrono_request_timeout_s: int = 120

    # CORS
    cors_allow_origins: str = "*"

    # Cache
    chrono_cache_dir: str = "./data/chrono_origin/cache"

    @property
    def cors_origins_list(self) -> List[str]:
        if not self.cors_allow_origins or self.cors_allow_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
