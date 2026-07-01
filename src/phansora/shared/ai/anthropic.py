"""Anthropic-backed client for grounded web search + JSON reasoning.

Drop-in replacement for the old GeminiClient: it exposes the same
``grounded_search`` / ``reason_json`` surface so the orchestrator is unchanged.

Grounded search uses Claude's built-in ``web_search`` server tool — Claude
decides when to search, runs the queries, and returns an answer with citations.
Unlike the Gemini path, the returned URLs are already canonical article links,
so no redirect-resolution step is needed.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass
class AnthropicConfig:
    """Standalone config for the shared Anthropic client.

    Reads the same environment variables the products already use, so it stays
    decoupled from any single product's settings module.
    """

    api_key: str = ""
    search_model: str = "claude-sonnet-4-6"
    reasoning_model: str = "claude-opus-4-8"
    search_max_tokens: int = 4096
    reason_max_tokens: int = 16000
    search_max_uses: int = 4

    @classmethod
    def from_env(cls) -> "AnthropicConfig":
        return cls(
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            search_model=os.getenv("ANTHROPIC_SEARCH_MODEL", "claude-sonnet-4-6"),
            reasoning_model=os.getenv("ANTHROPIC_REASONING_MODEL", "claude-opus-4-8"),
            search_max_tokens=int(os.getenv("ANTHROPIC_SEARCH_MAX_TOKENS", "4096")),
            reason_max_tokens=int(os.getenv("ANTHROPIC_REASON_MAX_TOKENS", "16000")),
            search_max_uses=int(os.getenv("CHRONO_SEARCH_MAX_USES", "4")),
        )


# Claude Opus 4.8 / Sonnet 4.6 reject sampling params (temperature/top_p/top_k)
# and `budget_tokens`, so we never pass them. Output is steered via prompting.
_JSON_SYSTEM = (
    "You are a precise research assistant. Respond with ONLY a single valid JSON "
    "object that satisfies the structure described in the user's message. Do not "
    "wrap it in markdown code fences and do not add any prose before or after the "
    "JSON."
)


def _parse_json(raw: str) -> Dict[str, Any]:
    """Parse a JSON object, salvaging by trimming to the outermost braces."""
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


@dataclass
class GroundedAnswer:
    text: str
    citations: List[Dict[str, str]] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)


class AnthropicClient:
    def __init__(self, config: Optional[AnthropicConfig] = None) -> None:
        cfg = config or AnthropicConfig.from_env()
        if not cfg.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        self._client = anthropic.Anthropic(api_key=cfg.api_key)
        self._search_model = cfg.search_model
        self._reasoning_model = cfg.reasoning_model
        self._search_max_tokens = cfg.search_max_tokens
        self._reason_max_tokens = cfg.reason_max_tokens
        self._search_max_uses = cfg.search_max_uses

    # ------------------------------------------------------------------ search
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def grounded_search(self, prompt: str, *, temperature: float = 0.1) -> GroundedAnswer:
        """Run a Claude call with the web search tool enabled (grounded + cited).

        ``temperature`` is accepted for interface compatibility but ignored — the
        current Claude models reject the parameter.
        """
        resp = self._client.messages.create(
            model=self._search_model,
            max_tokens=self._search_max_tokens,
            tools=[
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": self._search_max_uses,
                }
            ],
            messages=[{"role": "user", "content": prompt}],
        )

        text_parts: List[str] = []
        citations: List[Dict[str, str]] = []
        queries: List[str] = []
        seen_urls: set[str] = set()

        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", "") or "")
            elif btype == "server_tool_use" and getattr(block, "name", "") == "web_search":
                query = (getattr(block, "input", None) or {}).get("query")
                if query:
                    queries.append(query)
            elif btype == "web_search_tool_result":
                content = getattr(block, "content", None)
                # Success -> list of web_search_result; error -> a single object.
                if isinstance(content, list):
                    for result in content:
                        url = getattr(result, "url", "") or ""
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            citations.append(
                                {
                                    "url": url,
                                    "title": getattr(result, "title", "") or url,
                                }
                            )

        text = "\n".join(p for p in text_parts if p).strip()
        return GroundedAnswer(text=text, citations=citations, queries=queries)

    # --------------------------------------------------------------- reasoning
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def reason_json(
        self,
        prompt: str,
        *,
        schema: Optional[Dict[str, Any]] = None,
        temperature: float = 0.2,
        use_reasoning_model: bool = True,
    ) -> Dict[str, Any]:
        """Call Claude (no tools) and parse a JSON object from the reply.

        ``schema`` and ``temperature`` are accepted for interface compatibility;
        JSON shape is steered via the system prompt and the per-call instructions.
        """
        model = self._reasoning_model if use_reasoning_model else self._search_model
        resp = self._client.messages.create(
            model=model,
            max_tokens=self._reason_max_tokens,
            system=_JSON_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(
            getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text"
        )
        return _parse_json(raw or "{}")
