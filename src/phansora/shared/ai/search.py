"""Pluggable web search, for providers whose models cannot search for themselves.

Two backends, chosen by ``CHRONO_SEARCH_PROVIDER`` (or auto-detected):

  - ``brave``    — Brave Search API. Needs BRAVE_API_KEY.
  - ``searxng``  — a SearXNG instance you host. Needs SEARXNG_URL.

Auto-detect order when CHRONO_SEARCH_PROVIDER is unset: brave (if key) → searxng
(if url) → **none**, and "none" is a state this module says out loud rather than
papering over.

There used to be a third, keyless backend: DuckDuckGo, via the ``ddgs`` package. It
is gone. It answered 202 when it throttled, which arrives here as an empty result
list — indistinguishable from a search that genuinely found nothing. So a throttled
trace reported "no evidence found" for subjects with abundant surviving evidence, and
the retry loop below quietly tripled the latency of every failure on the way there.
Days went into rewriting prompts that were never the problem.

Nothing keyless replaced it, on purpose. A backend that fails silently is worse than
no backend: with none configured, ``search_available()`` is False and the caller can
tell the user their search is not set up, which is a fixable answer. The default
Chrono-Origin provider is now Gemini, whose model does its own grounded searching and
never reaches this module at all.
"""
from __future__ import annotations

import logging
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import List
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)


# How many searches may be in flight at once, per backend.
#
# The callers multiply: the orchestrator runs one worker per query (six by default) and
# each grounded_search runs its derived queries concurrently, so the arithmetic reaches
# twelve simultaneous requests where it used to reach five. Brave's free tier is rate
# limited per second and a self-hosted SearXNG is only as parallel as its own upstreams,
# so a ceiling still belongs here.
#
# The gate belongs at this layer rather than in the callers. Two independent pools cannot
# see each other's depth, and a limit either of them enforces is a limit the other can
# multiply.
_DEFAULT_CONCURRENCY = {"brave": 8, "searxng": 8}

_sem_lock = threading.Lock()
_semaphores: dict = {}


def _gate(provider: str) -> threading.Semaphore:
    """One semaphore per backend, created on first use."""
    with _sem_lock:
        sem = _semaphores.get(provider)
        if sem is None:
            raw = (os.getenv("CHRONO_SEARCH_CONCURRENCY") or "").strip()
            limit = _DEFAULT_CONCURRENCY.get(provider, 4)
            if raw.isdigit() and int(raw) > 0:
                limit = int(raw)
            sem = threading.Semaphore(limit)
            _semaphores[provider] = sem
        return sem


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""


@dataclass
class SearchConfig:
    # Empty means no backend is configured. Not a default that might work.
    provider: str = ""
    brave_api_key: str = ""
    searxng_url: str = ""
    max_results: int = 10
    timeout_s: int = 20
    # Cap results from any one site. Search engines happily return five pages of
    # one domain for a niche historical query; five pages of one publisher is one
    # source told five ways, and it crowds out the corroboration worth having.
    max_per_domain: int = 2
    attempts: int = 3

    @classmethod
    def from_env(cls) -> "SearchConfig":
        brave = os.getenv("BRAVE_API_KEY", "").strip()
        searxng = os.getenv("SEARXNG_URL", "").strip().rstrip("/")
        provider = os.getenv("CHRONO_SEARCH_PROVIDER", "").strip().lower()
        if not provider:
            provider = "brave" if brave else "searxng" if searxng else ""
        return cls(
            provider=provider,
            brave_api_key=brave,
            searxng_url=searxng,
            max_results=int(os.getenv("CHRONO_SEARCH_RESULTS", "10")),
            max_per_domain=int(os.getenv("CHRONO_SEARCH_MAX_PER_DOMAIN", "2")),
            attempts=int(os.getenv("CHRONO_SEARCH_ATTEMPTS", "3")),
        )


def search_available(cfg: SearchConfig | None = None) -> bool:
    """Is there a backend that can actually run a search?

    Callers use this to tell "we searched and found nothing" apart from "we never
    searched". Those two look identical in the results and mean opposite things to
    whoever is reading the timeline.
    """
    cfg = cfg or SearchConfig.from_env()
    if cfg.provider == "brave":
        return bool(cfg.brave_api_key)
    if cfg.provider == "searxng":
        return bool(cfg.searxng_url)
    return False


def _domain(url: str) -> str:
    try:
        host = (urlsplit(url or "").hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _diversify(results: List[SearchResult], cfg: SearchConfig) -> List[SearchResult]:
    """Keep result order, but stop any one domain dominating the set."""
    if cfg.max_per_domain <= 0:
        return results
    counts: dict[str, int] = {}
    kept: List[SearchResult] = []
    for r in results:
        d = _domain(r.url)
        if d:
            if counts.get(d, 0) >= cfg.max_per_domain:
                continue
            counts[d] = counts.get(d, 0) + 1
        kept.append(r)
    return kept


def web_search(query: str, *, cfg: SearchConfig | None = None) -> List[SearchResult]:
    """Run one web search; return up to ``cfg.max_results`` results. Never raises —
    returns [] on failure so the pipeline degrades gracefully.
    """
    cfg = cfg or SearchConfig.from_env()
    query = (query or "").strip()
    if not query:
        return []

    if not search_available(cfg):
        # Not retried, and not silent. There is nothing to retry against.
        logger.warning(
            "No web search backend is configured, so %r was not searched. Set "
            "BRAVE_API_KEY or SEARXNG_URL, or use CHRONO_LLM_PROVIDER=gemini, whose "
            "model searches natively.",
            query,
        )
        return []

    attempts = max(1, cfg.attempts)
    for attempt in range(attempts):
        try:
            with _gate(cfg.provider):
                results = _brave(query, cfg) if cfg.provider == "brave" else _searxng(query, cfg)
            if results:
                return _diversify(results, cfg)
            if attempt == attempts - 1:
                return []
        except Exception as exc:  # noqa: BLE001 — search is best-effort
            if attempt == attempts - 1:
                logger.warning("web_search(%r) via %s failed: %s", query, cfg.provider, exc)
                return []
            logger.debug("web_search(%r) attempt %d failed: %s", query, attempt + 1, exc)
        time.sleep(min(4.0, (2 ** attempt)) + random.uniform(0, 0.6))
    return []


def _brave(query: str, cfg: SearchConfig) -> List[SearchResult]:
    if not cfg.brave_api_key:
        raise RuntimeError("CHRONO_SEARCH_PROVIDER=brave but BRAVE_API_KEY is unset.")
    resp = httpx.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": cfg.max_results},
        headers={"X-Subscription-Token": cfg.brave_api_key, "Accept": "application/json"},
        timeout=cfg.timeout_s,
    )
    resp.raise_for_status()
    results = (resp.json().get("web") or {}).get("results") or []
    return [
        SearchResult(title=r.get("title", ""), url=r.get("url", ""), snippet=r.get("description", ""))
        for r in results
        if r.get("url")
    ][: cfg.max_results]


def _searxng(query: str, cfg: SearchConfig) -> List[SearchResult]:
    if not cfg.searxng_url:
        raise RuntimeError("CHRONO_SEARCH_PROVIDER=searxng but SEARXNG_URL is unset.")
    resp = httpx.get(
        f"{cfg.searxng_url}/search",
        params={"q": query, "format": "json"},
        timeout=cfg.timeout_s,
    )
    resp.raise_for_status()
    results = resp.json().get("results") or []
    return [
        SearchResult(title=r.get("title", ""), url=r.get("url", ""), snippet=r.get("content", ""))
        for r in results
        if r.get("url")
    ][: cfg.max_results]
