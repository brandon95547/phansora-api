"""Pluggable web search for grounded answers.

Provides real search results (title/url/snippet) that a model can synthesize +
cite. Three backends, chosen by ``CHRONO_SEARCH_PROVIDER`` (or auto-detected):

  - ``brave``       — Brave Search API. Reliable; free tier (needs BRAVE_API_KEY).
  - ``searxng``     — a SearXNG instance you host. Free (needs SEARXNG_URL).
  - ``duckduckgo``  — no key, best-effort scraping via the ``ddgs`` package.
                      Default, but rate-limited / less reliable than the above.

Auto-detect order when CHRONO_SEARCH_PROVIDER is unset: brave (if key) →
searxng (if url) → duckduckgo.
"""
from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass
from typing import List
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""


@dataclass
class SearchConfig:
    provider: str = "duckduckgo"
    brave_api_key: str = ""
    searxng_url: str = ""
    max_results: int = 5
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
            provider = "brave" if brave else "searxng" if searxng else "duckduckgo"
        return cls(
            provider=provider,
            brave_api_key=brave,
            searxng_url=searxng,
            max_results=int(os.getenv("CHRONO_SEARCH_RESULTS", "5")),
            max_per_domain=int(os.getenv("CHRONO_SEARCH_MAX_PER_DOMAIN", "2")),
            attempts=int(os.getenv("CHRONO_SEARCH_ATTEMPTS", "3")),
        )


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

    Retries with jittered backoff because the keyless DuckDuckGo backend is rate
    limited, and a trace that loses a round of searches to throttling produces a
    materially worse timeline than one that waited a second and asked again.
    """
    cfg = cfg or SearchConfig.from_env()
    query = (query or "").strip()
    if not query:
        return []

    attempts = max(1, cfg.attempts)
    for attempt in range(attempts):
        try:
            if cfg.provider == "brave":
                results = _brave(query, cfg)
            elif cfg.provider == "searxng":
                results = _searxng(query, cfg)
            else:
                results = _duckduckgo(query, cfg)
            if results:
                return _diversify(results, cfg)
            # An empty result set from a rate-limited backend is indistinguishable
            # from a genuinely empty one, so treat it as retryable — but only
            # while attempts remain.
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


def _duckduckgo(query: str, cfg: SearchConfig) -> List[SearchResult]:
    try:
        from ddgs import DDGS  # optional dep; only needed for the keyless default
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "DuckDuckGo search needs the 'ddgs' package (pip install ddgs), or set "
            "BRAVE_API_KEY / SEARXNG_URL to use a more reliable provider."
        ) from exc
    out: List[SearchResult] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=cfg.max_results):
            url = r.get("href") or r.get("url") or ""
            if url:
                out.append(
                    SearchResult(title=r.get("title", ""), url=url, snippet=r.get("body", ""))
                )
    return out[: cfg.max_results]
