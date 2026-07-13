"""DeepSeek-backed research client — Chrono-Origin's selectable fallback provider.

Exposes the same surface the orchestrator uses (a drop-in for the OpenAI client):
  - ``grounded_search(prompt)`` -> GroundedAnswer(text, citations, queries)
  - ``reason_json(prompt, *, use_reasoning_model)`` -> dict

DeepSeek has no built-in web search, so ``grounded_search`` does it in three steps,
mirroring what a hosted web_search tool does internally:
  1. derive 1-2 search queries from the prompt,
  2. run them via ``search.web_search`` (Brave / SearXNG / DuckDuckGo),
  3. have DeepSeek write a concise, cited summary from the real results.

``reason_json`` is a plain DeepSeek chat call in JSON mode, parsed with the same
salvage logic the other research clients use.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .research import GroundedAnswer
from .search import SearchConfig, SearchResult, web_search

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
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


@dataclass
class DeepSeekConfig:
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    reasoning_model: str = "deepseek-chat"
    reason_max_tokens: int = 8000
    search_max_tokens: int = 1024
    timeout_s: int = 120

    @classmethod
    def from_env(cls) -> "DeepSeekConfig":
        base = (os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
        return cls(
            api_key=(os.getenv("DEEPSEEK_API_KEY") or "").strip(),
            base_url=base,
            model=model,
            reasoning_model=os.getenv("DEEPSEEK_REASONING_MODEL", model).strip(),
            reason_max_tokens=int(os.getenv("DEEPSEEK_REASON_MAX_TOKENS", "8000")),
            search_max_tokens=int(os.getenv("DEEPSEEK_SEARCH_MAX_TOKENS", "1024")),
        )


class DeepSeekResearchClient:
    def __init__(self, config: Optional[DeepSeekConfig] = None) -> None:
        self._cfg = config or DeepSeekConfig.from_env()
        if not self._cfg.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set.")
        self._search_cfg = SearchConfig.from_env()

    # ------------------------------------------------------------- chat plumbing
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def _chat(
        self,
        *,
        system: str,
        user: str,
        model: str,
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        resp = httpx.post(
            f"{self._cfg.base_url}/v1/chat/completions",
            json=payload,
            headers={
                "Authorization": f"Bearer {self._cfg.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self._cfg.timeout_s,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"DeepSeek HTTP {resp.status_code}: {resp.text[:500]}")
        choices = resp.json().get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message") or {}).get("content") or ""

    # --------------------------------------------------------------- reasoning
    def reason_json(
        self,
        prompt: str,
        *,
        schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.2,
        use_reasoning_model: bool = True,
    ) -> Dict[str, Any]:
        model = self._cfg.reasoning_model if use_reasoning_model else self._cfg.model
        raw = self._chat(
            system=_JSON_SYSTEM,
            user=prompt,
            model=model,
            max_tokens=self._cfg.reason_max_tokens,
            json_mode=True,
        )
        return _parse_json(raw or "{}")

    # ------------------------------------------------------------------ search
    def grounded_search(self, prompt: str, *, temperature: float = 0.1) -> GroundedAnswer:
        queries = self._derive_queries(prompt)

        results: List[SearchResult] = []
        seen: set[str] = set()
        for q in queries:
            for r in web_search(q, cfg=self._search_cfg):
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
            text = self._chat(
                system="You are a precise research assistant. Ground every statement in the "
                "provided search results and never fabricate sources.",
                user=synth_user,
                model=self._cfg.model,
                max_tokens=self._cfg.search_max_tokens,
            ).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("DeepSeek synthesis failed: %s", exc)
            text = ""

        citations = [{"url": r.url, "title": r.title or r.url} for r in results]
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
            raw = self._chat(
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
