"""Environment-driven configuration."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # The LLM provider is configured via env (CHRONO_LLM_PROVIDER: openai | deepseek)
    # and read directly by the research clients in phansora.shared.ai; no
    # provider keys live here.

    # MUST exceed the LLM client's worst case, or every slow call leaks a thread.
    #
    # /expand runs its work in a ThreadPoolExecutor and bounds it with asyncio.wait_for.
    # wait_for cancels the AWAIT; it cannot cancel the thread. So when the budget is
    # shorter than the work, the handler returns 504 and the thread keeps running,
    # holding a worker until it finishes on its own. Four of those and the pool is
    # exhausted: later requests queue and never start, which looks exactly like the
    # product being dead — requests arriving, no LLM calls being made.
    #
    # Every provider client makes 3 attempts with backoff between them (1s, then 2s):
    # DeepSeek and OpenAI are 120s per attempt (OPENAI_TIMEOUT_S for the latter), Gemini
    # 90s (GEMINI_TIMEOUT_S), so the worst case is ~363s. This sits above it so the thread
    # always completes before the handler gives up. The caller sees a failure sooner
    # regardless — the Node proxy times out at 180s — but the worker is always returned.
    chrono_request_timeout_s: int = 420

    # Cache
    chrono_cache_dir: str = "./data/chrono_origin/cache"
    # How long a cached trace stays servable. The product's own description is "real
    # Google-grounded AI search"; with no expiry a trace answered once was answered
    # forever, so a user paying for a fresh look could be handed a months-old one with
    # nothing on screen to say so. 0 disables expiry.
    chrono_cache_ttl_days: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()
