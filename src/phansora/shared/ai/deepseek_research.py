"""DeepSeek-backed research client — Chrono-Origin's selectable fallback provider.

Exposes the same surface the orchestrator uses (a drop-in for the OpenAI client):
  - ``grounded_search(prompt)`` -> GroundedAnswer(text, citations, queries)
  - ``reason_json(prompt, *, use_reasoning_model)`` -> dict

DeepSeek has no built-in web search, so ``grounded_search`` does it in three steps,
mirroring what a hosted web_search tool does internally:
  1. derive 1-2 search queries from the prompt,
  2. run them via ``search.web_search`` (Brave or SearXNG — both need configuring;
     the keyless DuckDuckGo backend is gone, see ``search`` for why),
  3. have DeepSeek write a concise, cited summary from the real results.

``reason_json`` is a plain DeepSeek chat call in JSON mode, parsed with the same
salvage logic the other research clients use.

Every call goes through the shared helper in ``shared.ai.deepseek``.
"""
from __future__ import annotations

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .deepseek import DeepSeekChatConfig, api_key, chat_json_sync, chat_sync
from .json_repair import parse_json_loose
from .research import GroundedAnswer
from .search import SearchConfig, SearchResult, web_search


# The ceiling the truncation retry may escalate to. Past this a prompt is not merely
# large, it is wrong — and doubling forever would turn one bad trace into a very
# expensive one.
# Headroom ABOVE the starting budget, not the starting budget itself. Raising the
# start to where the work lands without raising this left the two equal, which
# silently disabled the retry: the escalation is guarded by budget < ceiling, so a
# genuine overrun would have failed on the first attempt instead of escalating.
MAX_REASON_TOKENS = int(os.getenv("DEEPSEEK_REASON_MAX_TOKENS_CEILING", "64000"))

logger = logging.getLogger(__name__)

_JSON_SYSTEM = (
    "You are a precise research assistant. Respond with ONLY a single valid JSON "
    "object that satisfies the structure described in the user's message. Do not "
    "wrap it in markdown code fences and do not add any prose before or after the "
    "JSON."
)

_QUERY_LINE = re.compile(r"^\s*(?:search\s+)?query:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_QUOTED = re.compile(r'"([^"]{3,120})"')


def _parse_json(raw: str) -> Dict[str, Any]:
    """Parse a model's JSON answer, or raise.

    The fallback below only exists to strip prose or a markdown fence from around an
    OTHERWISE COMPLETE object. It used to be reached by truncated answers too, and on
    those it did real damage: `rfind("}")` finds the last closing brace of whatever
    survived, which in a cut-off document is the end of some INNER object. That parses
    cleanly and returns a dict — one with none of the keys the caller wanted. The caller
    then filled every missing field with a default, and a trace that had actually failed
    was stored as a success with an empty origin and an empty timeline.

    So the salvaged slice must start at the FIRST brace and end at the LAST, and it is
    accepted only if the whole slice parses. A partial answer raises, which is what the
    truncation retry in reason_json is there to catch. The shared salvage holds to that:
    every span it tries runs to the LAST closing bracket, so a cut-off document still
    fails, and an array it may find inside one is refused below.
    """
    if not (raw or "").strip():
        return {}
    parsed = parse_json_loose(raw)

    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")
    return parsed


@dataclass
class DeepSeekConfig:
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    # No literals: from_env() resolves both through shared.ai.models, which raises
    # if nothing is configured rather than falling back to a name that may be retired.
    model: str = ""
    reasoning_model: str = ""
    # Start where this workload actually lands, not where a doubling ladder would
    # eventually reach it. At 8000 a trace synthesis generated ~8k tokens, was cut off,
    # regenerated at 16k, was cut off again, and only succeeded on the third attempt at
    # 32k — three generations for one answer, two of them discarded. It is a CAP and not
    # a charge: nothing is billed unless the tokens are produced, so a ceiling set where
    # the work already is costs nothing and saves two full passes.
    reason_max_tokens: int = 32000
    search_max_tokens: int = 1024
    timeout_s: int = 120

    @classmethod
    def from_env(cls) -> "DeepSeekConfig":
        base = (os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        from .models import resolve_model, resolve_reasoning_model
        model = resolve_model("CHRONO_MODEL", provider="deepseek")
        return cls(
            api_key=api_key(),
            base_url=base,
            model=model,
            reasoning_model=resolve_reasoning_model("CHRONO_MODEL", provider="deepseek"),
            reason_max_tokens=int(os.getenv("DEEPSEEK_REASON_MAX_TOKENS", "32000")),
            search_max_tokens=int(os.getenv("DEEPSEEK_SEARCH_MAX_TOKENS", "1024")),
        )


class DeepSeekResearchClient:
    def __init__(self, config: Optional[DeepSeekConfig] = None) -> None:
        self._cfg = config or DeepSeekConfig.from_env()
        if not self._cfg.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set.")
        self._search_cfg = SearchConfig.from_env()

    # ------------------------------------------------------------- chat plumbing
    def _chat_cfg(self, model: str) -> DeepSeekChatConfig:
        # Three attempts, with 1s and then 2s of backoff between them: what every research
        # client allows itself, and what the request budget is measured against (see
        # chrono_request_timeout_s). The shared helper retries only what a second attempt
        # might change, and gives up with DeepSeek's own error and status.
        return DeepSeekChatConfig(
            base_url=self._cfg.base_url,
            api_key=self._cfg.api_key,
            model=model,
            timeout_s=self._cfg.timeout_s,
            max_retries=2,
            min_retry_sleep_s=1.0,
            max_retry_sleep_s=8.0,
        )

    def _chat(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int,
        json_mode: bool = False,
    ) -> tuple[str, Optional[str]]:
        # The thinking is left on, as this client has always run it, unlike the other
        # DeepSeek callers: reason_json's budgets are sized with the reasoning in them.
        return chat_sync(
            cfg=self._chat_cfg(model),
            system=system,
            user=user,
            max_tokens=max_tokens,
            json_mode=json_mode,
            disable_thinking=False,
        )

    # --------------------------------------------------------------- reasoning
    def reason_json(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        use_reasoning_model: bool = True,
    ) -> Dict[str, Any]:
        model = self._cfg.reasoning_model if use_reasoning_model else self._cfg.model

        # Retry on TRUNCATION, doubling the budget — the shared chat_json does it, as it
        # does for Book Alchemy.
        #
        # This is a REASONING model, and its reasoning tokens are billed against
        # max_tokens (see shared/ai/deepseek.py). A synthesize prompt carrying a trace's
        # whole evidence set can spend the entire budget thinking and get cut off
        # mid-JSON. Nothing here used to notice: the cut-off text went to a parser that
        # salvaged some inner object from it, and the caller got a dict with none of the
        # keys it wanted — an empty origin and an empty timeline, reported as success.
        reply = chat_json_sync(
            cfg=self._chat_cfg(model),
            system=_JSON_SYSTEM,
            user=prompt,
            max_tokens=self._cfg.reason_max_tokens,
            max_tokens_ceiling=MAX_REASON_TOKENS,
            disable_thinking=False,  # see _chat
        )
        if reply.finish_reason == "length":
            # Out of headroom and still cut off. Failing is the honest outcome: a partial
            # answer here becomes a trace with no origin and no timeline, and the user has
            # already paid for it.
            raise RuntimeError(
                f"The model's answer was cut off at {max(self._cfg.reason_max_tokens, MAX_REASON_TOKENS)} "
                "tokens and could not be completed."
            )
        return _parse_json(reply.text or "{}")

    # ------------------------------------------------------------------ search
    def grounded_search(self, prompt: str, *, temperature: float = 0.1) -> GroundedAnswer:
        queries = self._derive_queries(prompt)

        # Concurrently, not one after the other. Each web_search is up to three
        # attempts against a 20s timeout with backoff sleeps between them, so two
        # queries in sequence is a worst case north of two minutes — inside a single
        # worker that the orchestrator is already running five of in parallel. That
        # made this the largest latency amplifier on the DeepSeek path.
        #
        # Results are merged in QUERY ORDER rather than completion order: the first
        # query is the one derived from the prompt's own "Search query:" line, and
        # letting a slower second query jump ahead of it would quietly reorder what
        # the summariser sees as the most relevant sources.
        if len(queries) == 1:
            per_query = [web_search(queries[0], cfg=self._search_cfg)]
        else:
            with ThreadPoolExecutor(max_workers=len(queries)) as pool:
                per_query = list(pool.map(lambda q: web_search(q, cfg=self._search_cfg), queries))

        results: List[SearchResult] = []
        seen: set[str] = set()
        for batch in per_query:
            for r in batch:
                if r.url and r.url not in seen:
                    seen.add(r.url)
                    results.append(r)

        if not results:
            return GroundedAnswer(text="", citations=[], queries=queries)

        results = results[: max(self._search_cfg.max_results * 2, 6)]
        sources_block = "\n".join(
            f"[{i}] {r.title or r.url}\nURL: {r.url}\n{(r.snippet or '')[:400]}"
            for i, r in enumerate(results, 1)
        )
        synth_user = (
            f"{prompt}\n\n"
            "SEARCH RESULTS (use ONLY these; cite the exact URLs):\n"
            f"{sources_block}\n\n"
            "Write the requested factual summary grounded strictly in the results above. "
            "Mention specific dates, eras, manuscript names, authors, and cultures when the "
            "sources do. Do not invent facts or URLs. If the results are irrelevant, say so briefly."
        )
        try:
            text, _finish = self._chat(
                system="You are a precise research assistant. Ground every statement in the "
                "provided search results and never fabricate sources.",
                user=synth_user,
                model=self._cfg.model,
                max_tokens=self._cfg.search_max_tokens,
            )
            text = text.strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("DeepSeek synthesis failed: %s", exc)
            text = ""

        citations = [
            {"url": r.url, "title": r.title or r.url, "snippet": (r.snippet or "")[:400]}
            for r in results
        ]
        return GroundedAnswer(text=text, citations=citations, queries=queries)

    def _derive_queries(self, prompt: str) -> List[str]:
        """Get 1-2 web queries from the orchestrator's search prompt, cheaply."""
        queries: List[str] = []

        m = _QUERY_LINE.search(prompt)
        if m:
            queries.append(m.group(1).strip())

        # Add the quoted story title as a second angle (helps the expand path,
        # which has no explicit "Search query:" line).
        q = _QUOTED.search(prompt)
        if q:
            title = q.group(1).strip()
            if title and title not in queries:
                queries.append(title)

        if queries:
            return queries[:2]

        # Fallback: let DeepSeek propose queries from the instruction.
        try:
            raw, _finish = self._chat(
                system=_JSON_SYSTEM,
                user=(
                    "From the research instruction below, return JSON "
                    '{"queries": [up to 2 concise, self-contained web search queries]}.\n\n'
                    f"{prompt}"
                ),
                model=self._cfg.model,
                max_tokens=256,
                json_mode=True,
            )
            data = _parse_json(raw or "{}")
            queries = [str(x).strip() for x in (data.get("queries") or []) if str(x).strip()]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Query derivation failed: %s", exc)
        return queries[:2] or [prompt[:200]]
