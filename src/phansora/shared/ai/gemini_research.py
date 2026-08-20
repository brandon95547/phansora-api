"""Gemini research client — the model does its own searching.

Same two methods as the OpenAI and DeepSeek clients (``grounded_search`` and
``reason_json``), so the orchestrator is unchanged. What differs is where the web
results come from.

DeepSeek has no hosted search tool — its API rejects anything but ``type: "function"``
— so that path had to fetch results itself and paste them into a prompt. The keyless
backend for that was DuckDuckGo, which answers 202 when it throttles: no results, no
error, and an expansion that reported "no evidence found" when nothing had actually
been searched. Hours went into rewriting prompts that were never at fault.

Gemini grounds against Google Search natively. There is no scraper to be throttled, no
concurrency gate to tune, and the citations come back attached to the answer rather
than being reassembled from a separate result list. Grounding is billed per grounded
REQUEST rather than per token, and the free monthly allowance covers ordinary use.
"""
from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from . import usage
from .research import GroundedAnswer

logger = logging.getLogger(__name__)

_API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"

# Output budgets. Generous on purpose: these are caps, not charges, and a budget set
# where the work does not fit means generating an answer, discarding it, and generating
# it again — which is how the DeepSeek path came to pay for three passes per synthesis.
_REASON_MAX_TOKENS = int(os.getenv("GEMINI_REASON_MAX_TOKENS", "32000"))
_SEARCH_MAX_TOKENS = int(os.getenv("GEMINI_SEARCH_MAX_TOKENS", "4000"))

_JSON_SYSTEM = "You return only valid JSON. No prose, no code fences, no commentary."

# Grounding does not hand back the pages it read. It hands back redirect proxies on
# this host, one per source, with the real domain tucked into the chunk's `title`.
#
# Left unresolved that is quietly destructive, because every downstream decision about
# a source is made from its URL. Every citation resolves to the SAME host, so the
# five-tier source policy scores quora.com and a university library identically, the
# "never read a low-authority page" rule never fires, page-read ranking becomes
# arbitrary, and per-domain diversification sees one domain and stops diversifying.
# Nothing raises; the trace just quietly rests on forum posts.
#
# They are also proxies, so a saved citation stops resolving once Google expires it —
# in a product whose whole promise is that you can go and look at the thing.
_PROXY_HOST = "vertexaisearch.cloud.google.com"
_RESOLVE_TIMEOUT_S = float(os.getenv("GEMINI_RESOLVE_TIMEOUT_S", "6"))
_RESOLVE_WORKERS = int(os.getenv("GEMINI_RESOLVE_WORKERS", "8"))

_REDIRECT_CODES = {301, 302, 303, 307, 308}


def _is_proxy(url: str) -> bool:
    try:
        return (urlsplit(url).hostname or "").lower().endswith(_PROXY_HOST)
    except ValueError:
        return False


def _looks_like_domain(text: str) -> bool:
    """Grounding puts the bare domain in `title` — "quora.com", not a page title."""
    t = (text or "").strip()
    return bool(t) and " " not in t and "." in t and "/" not in t


def _resolve_proxy(url: str) -> str:
    """Follow one grounding redirect to the page it actually points at.

    Header-only: redirects are answered without a body worth reading, and the page
    itself is fetched later by the reader stage if it earns a read.
    """
    try:
        with httpx.Client(timeout=_RESOLVE_TIMEOUT_S, follow_redirects=False) as client:
            resp = client.get(url)
        if resp.status_code in _REDIRECT_CODES:
            target = (resp.headers.get("location") or "").strip()
            if target.startswith("http"):
                return target
    except Exception as exc:  # noqa: BLE001 - an unresolved proxy is not fatal
        logger.debug("Could not resolve a grounding redirect: %s", exc)
    return url


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


@dataclass
class GeminiConfig:
    api_key: str = ""
    model: str = ""
    reasoning_model: str = ""
    # Must stay under the caller's request budget, or a slow call outlives the handler
    # that is waiting on it and strands the worker running it.
    timeout_s: int = 90

    @classmethod
    def from_env(cls) -> "GeminiConfig":
        from .models import resolve_model, resolve_reasoning_model

        return cls(
            api_key=_env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY"),
            model=resolve_model("CHRONO_MODEL", provider="gemini"),
            reasoning_model=resolve_reasoning_model("CHRONO_MODEL", provider="gemini"),
            timeout_s=int(_env("GEMINI_TIMEOUT_S", "90")),
        )


def _strip_fences(text: str) -> str:
    """Models fence JSON even when told not to."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _parse_json(raw: str) -> Dict[str, Any]:
    """Parse, falling back to the outermost object if there is text around it."""
    text = _strip_fences(raw)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


class GeminiResearchClient:
    """Grounded search and JSON reasoning against the Gemini API."""

    def __init__(self, cfg: Optional[GeminiConfig] = None) -> None:
        self._cfg = cfg or GeminiConfig.from_env()
        if not self._cfg.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Chrono-Origin cannot search without it."
            )
        if not self._cfg.model:
            raise RuntimeError(
                "No Gemini model configured. Set GEMINI_MODEL (or CHRONO_MODEL) and restart. "
                "There is no built-in default on purpose: a hardcoded name breaks silently "
                "when the provider retires it."
            )

    # ------------------------------------------------------------------ transport
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def _generate(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        grounded: bool = False,
        json_out: bool = False,
        system: str = "",
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if grounded:
            # The whole reason this client exists.
            body["tools"] = [{"google_search": {}}]
        if json_out:
            # Not combined with grounding: the API rejects a forced JSON mime type
            # alongside a search tool, and the two stages want different things anyway.
            body["generationConfig"]["responseMimeType"] = "application/json"

        url = f"{_API_ROOT}/{model}:generateContent"
        with httpx.Client(timeout=self._cfg.timeout_s) as client:
            resp = client.post(
                url,
                params={"key": self._cfg.api_key},
                json=body,
                headers={"Content-Type": "application/json"},
            )
        if resp.status_code >= 400:
            raise RuntimeError(f"Gemini {resp.status_code}: {resp.text[:400]}")
        data = resp.json()

        meta = data.get("usageMetadata") or {}
        usage.record(
            input_tokens=int(meta.get("promptTokenCount") or 0),
            output_tokens=int(meta.get("candidatesTokenCount") or 0),
        )
        return data

    @staticmethod
    def _text_of(data: Dict[str, Any]) -> str:
        for cand in data.get("candidates") or []:
            parts = ((cand.get("content") or {}).get("parts")) or []
            joined = "".join(p.get("text") or "" for p in parts).strip()
            if joined:
                return joined
        return ""

    @staticmethod
    def _citations_of(data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Sources the model actually grounded on, from groundingMetadata.

        Kept to what the response says it used. Inventing citations from the model's
        prose is how a trace ends up attributing a claim to a page that never mentioned
        it, and this product exists to make that impossible.

        Proxy URLs are resolved to the pages they point at — see _PROXY_HOST above for
        why leaving them unresolved silently disables source tiering.
        """
        raw: List[Dict[str, str]] = []
        seen = set()
        for cand in data.get("candidates") or []:
            gm = cand.get("groundingMetadata") or {}
            for chunk in gm.get("groundingChunks") or []:
                web = chunk.get("web") or {}
                url = (web.get("uri") or "").strip()
                if not url or url in seen:
                    continue
                seen.add(url)
                raw.append({"url": url, "title": (web.get("title") or "").strip()})

        proxies = [c["url"] for c in raw if _is_proxy(c["url"])]
        resolved: Dict[str, str] = {}
        if proxies:
            # Concurrent: these are sequential round trips to the same host, and a
            # trace gathers dozens of them.
            with ThreadPoolExecutor(max_workers=min(_RESOLVE_WORKERS, len(proxies))) as pool:
                resolved = dict(zip(proxies, pool.map(_resolve_proxy, proxies)))

        out: List[Dict[str, str]] = []
        final_seen = set()
        unresolved = 0
        for c in raw:
            url, title = c["url"], c["title"]
            if _is_proxy(url):
                url = resolved.get(url, url)
            if _is_proxy(url):
                unresolved += 1
                # Still not a usable URL. Fall back to the domain, which grounding
                # supplies as the chunk title: the site root is a worse link than the
                # page, but it is a REAL one that tiers correctly and outlives the
                # proxy. An untiered source is treated as though nothing is known
                # about its authority, which is how a forum post ends up weighted
                # like a university library.
                if _looks_like_domain(title):
                    url = "https://%s/" % title
            if url in final_seen:
                continue
            final_seen.add(url)
            out.append({"url": url, "title": title or url, "snippet": ""})

        if unresolved:
            logger.warning(
                "%d of %d grounding sources could not be resolved past the redirect.",
                unresolved, len(raw),
            )
        return out

    @staticmethod
    def _queries_of(data: Dict[str, Any]) -> List[str]:
        """What the model actually searched for — its own choice, not ours."""
        out: List[str] = []
        for cand in data.get("candidates") or []:
            gm = cand.get("groundingMetadata") or {}
            for q in gm.get("webSearchQueries") or []:
                q = (q or "").strip()
                if q and q not in out:
                    out.append(q)
        return out

    # -------------------------------------------------------------------- search
    def grounded_search(self, prompt: str, *, temperature: float = 0.1) -> GroundedAnswer:
        usage.stage("search")
        try:
            data = self._generate(
                model=self._cfg.model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=_SEARCH_MAX_TOKENS,
                grounded=True,
            )
        except Exception as exc:  # noqa: BLE001 — search is best-effort
            logger.warning("Gemini grounded search failed: %s", exc)
            return GroundedAnswer(text="", citations=[], queries=[])

        citations = self._citations_of(data)
        queries = self._queries_of(data)

        if not citations and not queries:
            # The model did not search. It answered from memory, and memory is not
            # evidence — a timeline step whose source is "the model recalled it" is
            # the one thing this product cannot ship. Returned as empty so it lands
            # in the caller's existing "search unavailable" path, which tells the
            # user their search did not run instead of reporting no evidence exists.
            logger.warning(
                "Gemini answered without searching; discarding an ungrounded answer "
                "rather than treating recall as research."
            )
            return GroundedAnswer(text="", citations=[], queries=[])

        if not citations:
            # It searched but cited nothing back. The text may still be usable, so it
            # is kept — but the gap is stated, because everything downstream assumes
            # a claim can be traced to a source.
            logger.warning("Gemini searched (%s) but returned no sources.", ", ".join(queries))

        return GroundedAnswer(text=self._text_of(data), citations=citations, queries=queries)

    # ------------------------------------------------------------------ reasoning
    def reason_json(
        self,
        prompt: str,
        *,
        schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.2,
        use_reasoning_model: bool = True,
    ) -> Dict[str, Any]:
        model = self._cfg.reasoning_model if use_reasoning_model else self._cfg.model
        usage.stage("extract")
        data = self._generate(
            model=model or self._cfg.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=_REASON_MAX_TOKENS,
            json_out=True,
            system=_JSON_SYSTEM,
        )

        finish = ""
        for cand in data.get("candidates") or []:
            finish = (cand.get("finishReason") or "").upper()
            break
        raw = self._text_of(data)

        if finish == "MAX_TOKENS" and not raw.rstrip().endswith("}"):
            # Truncated mid-JSON. Loud rather than silent: a cut-off answer parses into
            # a dict with none of the keys the caller wanted, and gets reported as an
            # empty result rather than as a failure.
            raise RuntimeError(
                f"The model's answer was cut off at {_REASON_MAX_TOKENS} tokens and could "
                "not be completed."
            )
        return _parse_json(raw)
