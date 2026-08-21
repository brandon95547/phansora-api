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
    #
    # It went 4 -> 6 when the objective was "cover every strand this subject has",
    # which meant researching language history, institutional development and the
    # calendar alongside the texts. Under the chain rule the question is narrower —
    # what survives, in order — and a narrower question converges in fewer rounds.
    # Each round costs seven LLM calls, so a ceiling nobody reaches is still a
    # ceiling that occasionally gets reached at seven calls a time.
    chrono_max_depth: int = 3
    chrono_min_depth: int = 2
    chrono_max_queries_per_stage: int = 6
    # MUST exceed the LLM client's worst case, or every slow call leaks a thread.
    #
    # /trace and /expand run their work in a ThreadPoolExecutor and bound it with
    # asyncio.wait_for. wait_for cancels the AWAIT; it cannot cancel the thread. So
    # when the budget is shorter than the work, the handler returns 504 and the thread
    # keeps running, holding a worker until it finishes on its own. Four of those and
    # the pool is exhausted: later requests queue and never start, which looks exactly
    # like the product being dead — requests arriving, no LLM calls being made.
    #
    # The DeepSeek client is 120s per attempt with 3 attempts plus backoff: ~370s worst
    # case. This sits above it so the thread always completes before the handler gives
    # up. The caller sees a failure sooner regardless — the Node proxy times out at
    # 180s — but the worker is always returned.
    chrono_request_timeout_s: int = 420

    # Reading real source pages. This is the expensive half of the budget and the
    # only thing that makes a provenance claim checkable rather than recalled, so
    # it is spent narrowly: a few top-tier pages, truncated, one per domain.
    # Raised from 4: a claim can only be marked verified if its page was actually
    # read, so the read budget is also the ceiling on how much of a trace can come
    # back as anything better than "unverified".
    chrono_read_sources: int = 6
    chrono_read_chars: int = 6000
    chrono_expand_read_sources: int = 2

    # Chasing citations backward. Mining is two HTTP fetches and a regex — it
    # costs no tokens, so it stays on. The search half only fires for claims that
    # are still resting on a lead after the rounds finish, which on a
    # well-sourced subject is none of them.
    chrono_chase_enabled: bool = True
    chrono_chase_mine_pages: int = 2
    chrono_chase_max_targets: int = 3
    chrono_chase_max_queries: int = 3

    # Ceiling on evaluated edges. Well past what a readable timeline needs — it
    # exists to stop a model emitting an edge for every pair of events.
    chrono_max_connections: int = 8

    # CORS
    cors_allow_origins: str = "*"

    # Cache
    chrono_cache_dir: str = "./data/chrono_origin/cache"
    # How long a cached trace stays servable. The product's own description is "real
    # Google-grounded AI search"; with no expiry a trace answered once was answered
    # forever, so a user paying for a fresh look could be handed a months-old one with
    # nothing on screen to say so. 0 disables expiry.
    chrono_cache_ttl_days: int = 30

    @property
    def cors_origins_list(self) -> List[str]:
        if not self.cors_allow_origins or self.cors_allow_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
