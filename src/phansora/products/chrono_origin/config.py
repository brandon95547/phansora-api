"""Environment-driven configuration."""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # The LLM provider is configured via env (CHRONO_LLM_PROVIDER: openai | deepseek)
    # and read directly by the research clients in phansora.shared.ai; no
    # provider keys live here.

    # Pipeline limits. max_depth is a ceiling, not a target: the loop stops as soon
    # as a round stops finding new evidence, so most traces never reach it.
    chrono_max_depth: int = 4
    chrono_min_depth: int = 2
    chrono_max_sources_per_stage: int = 8
    chrono_max_queries_per_stage: int = 5
    chrono_request_timeout_s: int = 240

    # Reading real source pages. This is the expensive half of the budget and the
    # only thing that makes a provenance claim checkable rather than recalled, so
    # it is spent narrowly: a few top-tier pages, truncated, one per domain.
    chrono_read_sources: int = 4
    chrono_read_chars: int = 6000
    chrono_expand_read_sources: int = 2

    # Ceiling on evaluated edges. Well past what a readable timeline needs — it
    # exists to stop a model emitting an edge for every pair of events.
    chrono_max_connections: int = 24

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
