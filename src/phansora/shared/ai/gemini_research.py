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

# Said in the system channel, never in the prompt. RESEARCH_PROMPT is tuned against the
# live model and has to reach it exactly as written, so the one thing that may be added
# when the model skips the tool is added somewhere the prompt text is not.
_SEARCH_SYSTEM = (
    "You must use the Google Search tool before answering. Run searches, read what comes "
    "back, and build your answer from those results. Do not answer from memory: an answer "
    "you did not search for cannot be used here."
)

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
    # The model that does the SEARCHING, which is not the same job as the rest.
    #
    # Whether a search happens at all is the model's own call — `google_search` is
    # elected, not commanded — and the cheap tiers routinely decline it, answer from
    # recall, and return no groundingMetadata at all. Nothing in the prompt fixes that:
    # gemini-3.5-flash-lite declined a research prompt AND declined the retry that put
    # "you must search" in the system channel, then wrote a fluent from-memory list.
    #
    # So the tier is the lever, and it is worth setting on its own: grounding wants a
    # model that reliably reaches for a tool, while synthesis is formatting work that a
    # lite tier does perfectly well for a fraction of the price. Blank = use `model`,
    # the same way a blank GEMINI_REASONING_MODEL falls through.
    search_model: str = ""
    # Must stay under the caller's request budget, or a slow call outlives the handler
    # that is waiting on it and strands the worker running it.
    timeout_s: int = 90

    @classmethod
    def from_env(cls) -> "GeminiConfig":
        from .models import resolve_model, resolve_reasoning_model

        model = resolve_model("CHRONO_MODEL", provider="gemini")
        return cls(
            api_key=_env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY"),
            model=model,
            reasoning_model=resolve_reasoning_model("CHRONO_MODEL", provider="gemini"),
            search_model=_env("GEMINI_SEARCH_MODEL") or model,
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

        # The key goes in a header. As a query parameter it lands in httpx's INFO log
        # line, which is how a live key came to sit in prod's journal in plaintext,
        # readable by anything that can run journalctl and kept in the archives.
        url = f"{_API_ROOT}/{model}:generateContent"
        with httpx.Client(timeout=self._cfg.timeout_s) as client:
            resp = client.post(
                url,
                json=body,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self._cfg.api_key,
                },
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
        """The model's answer, and the sources it grounded on if it grounded on any.

        An answer that came from recall is still returned. It used to be discarded on
        the grounds that memory is not evidence — true, and beside the point when the
        product is a list of what came before what. Dropping it failed the whole trace
        and refunded the credit while the answer the user asked for sat in the response,
        complete and in order. What must never happen is a recalled answer arriving with
        citations attached, and it cannot: citations are read from groundingMetadata,
        so an ungrounded answer has none to read.
        """
        usage.stage("search")
        answer, grounded = self._grounded_attempt(prompt, temperature=temperature, system="")
        if grounded or answer is None:
            # `answer is None` means the call itself failed. It did not answer from
            # memory — it did not answer — so there is nothing to ask again about, and
            # the transport already retried three times.
            return answer or GroundedAnswer(text="", citations=[], queries=[])

        # `google_search` is model-ELECTED: unlike the retired `google_search_retrieval`
        # it carries no threshold that can force a lookup, so nothing here can require
        # one and no prompt wording guarantees it. Asking again with the obligation in
        # the system channel is the only lever inside a single call that does not touch
        # the research prompt, which is tuned by hand and must arrive as written.
        logger.warning("Gemini answered without searching; asking again with search made explicit.")
        retry, grounded = self._grounded_attempt(
            prompt, temperature=temperature, system=_SEARCH_SYSTEM
        )
        if grounded and retry is not None:
            return retry

        # Still ungrounded, so the answer is unsourced and says so by carrying no
        # citations. It is returned anyway — the caller decides what an unsourced answer
        # is worth, and for a timeline of titles and dates it is worth the whole trace.
        logger.warning(
            "Gemini answered without searching on the retry either; returning an unsourced "
            "answer. If this is every trace rather than the odd one, the model tier is the "
            "cause: set GEMINI_SEARCH_MODEL."
        )
        return retry if retry is not None and (retry.text or "").strip() else answer

    def _grounded_attempt(
        self, prompt: str, *, temperature: float, system: str
    ) -> tuple[Optional[GroundedAnswer], bool]:
        """One grounded call, and whether the model actually searched.

        Three outcomes, because the two failures are not the same failure:

            (answer, True)   grounded — the model searched and cited
            (answer, False)  ungrounded — it answered from memory, and that answer may
                             still be exactly what was asked for
            (None, False)    the call did not happen at all; nothing to retry, nothing
                             to salvage
        """
        try:
            data = self._generate(
                model=self._cfg.search_model or self._cfg.model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=_SEARCH_MAX_TOKENS,
                grounded=True,
                system=system,
            )
        except Exception as exc:  # noqa: BLE001 — search is best-effort
            # Not retried by the caller: it did not "answer from memory", it did not
            # answer at all, and the transport already retried three times.
            logger.warning("Gemini grounded search failed: %s", exc)
            return None, False

        for cand in data.get("candidates") or []:
            if (cand.get("finishReason") or "").upper() == "MAX_TOKENS":
                # Said out loud, because the list is prose and a cut-off answer looks
                # exactly like a short one: the timeline simply stops early, with
                # nothing anywhere reporting that the rest was never delivered. Raise
                # GEMINI_SEARCH_MAX_TOKENS if this appears — it is a cap, not a charge.
                logger.warning(
                    "Grounded answer was cut off at %d tokens; the list is incomplete. "
                    "Raise GEMINI_SEARCH_MAX_TOKENS.", _SEARCH_MAX_TOKENS,
                )
            break

        text = self._text_of(data)
        citations = self._citations_of(data)
        queries = self._queries_of(data)
        answer = GroundedAnswer(text=text, citations=citations, queries=queries)

        if not citations and not queries:
            # Nothing in the response says a search happened. Logged with the size of
            # what came back, because a fluent answer with no groundingMetadata is
            # exactly what recall looks like and is otherwise indistinguishable in the
            # logs from a thin one.
            logger.warning(
                "%s returned no groundingMetadata (%d chars of unsourced text).",
                self._cfg.search_model or self._cfg.model, len(text),
            )
            return answer, False

        if not citations:
            # It searched but cited nothing back. The text is still usable — the gap is
            # stated because anything downstream that wants a source will not find one.
            logger.warning("Gemini searched (%s) but returned no sources.", ", ".join(queries))

        return answer, True

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
