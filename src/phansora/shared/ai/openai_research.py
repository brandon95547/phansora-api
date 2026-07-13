"""OpenAI GPT-5 Nano research client — Chrono-Origin's default provider.

Exposes the same surface the orchestrator uses (a drop-in for the DeepSeek
research client):
  - ``grounded_search(prompt)`` -> GroundedAnswer(text, citations, queries)
  - ``reason_json(prompt, *, use_reasoning_model)`` -> dict

Unlike DeepSeek, GPT-5 Nano searches the web itself: ``grounded_search`` runs a
Responses API call with the native ``web_search`` tool, so the model picks the
queries, runs them on OpenAI's infrastructure, and returns an answer with URL
citations — no external Brave/DuckDuckGo step. ``reason_json`` is a plain
Responses call in JSON-object mode.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .research import GroundedAnswer

logger = logging.getLogger(__name__)

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
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


@dataclass
class OpenAIResearchConfig:
    api_key: str = ""
    base_url: str = ""  # blank = api.openai.com
    # gpt-5-nano is a reasoning model with access to the native web_search tool.
    search_model: str = "gpt-5-nano"
    reasoning_model: str = "gpt-5-nano"
    # Reasoning effort: minimal | low | medium | high.
    reason_effort: str = "medium"
    light_effort: str = "low"
    search_effort: str = "low"
    reason_max_output_tokens: int = 8000
    search_max_output_tokens: int = 4000
    # The hosted web-search tool type; override to "web_search_preview" on older tiers.
    web_search_tool: str = "web_search"
    timeout_s: int = 120

    @classmethod
    def from_env(cls) -> "OpenAIResearchConfig":
        model = (os.getenv("OPENAI_MODEL", "gpt-5-nano") or "gpt-5-nano").strip()
        return cls(
            api_key=(os.getenv("OPENAI_API_KEY") or "").strip(),
            base_url=(os.getenv("OPENAI_BASE_URL") or "").strip(),
            search_model=(os.getenv("OPENAI_SEARCH_MODEL") or model).strip(),
            reasoning_model=(os.getenv("OPENAI_REASONING_MODEL") or model).strip(),
            reason_effort=(os.getenv("OPENAI_REASON_EFFORT", "medium") or "medium").strip(),
            light_effort=(os.getenv("OPENAI_LIGHT_EFFORT", "low") or "low").strip(),
            search_effort=(os.getenv("OPENAI_SEARCH_EFFORT", "low") or "low").strip(),
            reason_max_output_tokens=int(os.getenv("OPENAI_REASON_MAX_TOKENS", "8000")),
            search_max_output_tokens=int(os.getenv("OPENAI_SEARCH_MAX_TOKENS", "4000")),
            web_search_tool=(os.getenv("OPENAI_WEB_SEARCH_TOOL", "web_search") or "web_search").strip(),
            timeout_s=int(os.getenv("CHRONO_REQUEST_TIMEOUT_S", "120")),
        )


class OpenAIResearchClient:
    def __init__(self, config: Optional[OpenAIResearchConfig] = None) -> None:
        self._cfg = config or OpenAIResearchConfig.from_env()
        if not self._cfg.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        kwargs: Dict[str, Any] = {"api_key": self._cfg.api_key, "timeout": self._cfg.timeout_s}
        if self._cfg.base_url:
            kwargs["base_url"] = self._cfg.base_url
        self._client = OpenAI(**kwargs)

    # ------------------------------------------------------------------ search
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
    def grounded_search(self, prompt: str, *, temperature: float = 0.1) -> GroundedAnswer:
        """Responses API call with the native web_search tool (grounded + cited).

        ``temperature`` is accepted for interface parity but not sent — GPT-5
        reasoning models reject sampling params.
        """
        resp = self._client.responses.create(
            model=self._cfg.search_model,
            input=prompt,
            tools=[{"type": self._cfg.web_search_tool}],
            reasoning={"effort": self._cfg.search_effort},
            max_output_tokens=self._cfg.search_max_output_tokens,
        )

        citations: List[Dict[str, str]] = []
        queries: List[str] = []
        seen: set[str] = set()
        for item in getattr(resp, "output", None) or []:
            itype = getattr(item, "type", None)
            if itype == "web_search_call":
                action = getattr(item, "action", None)
                query = action.get("query") if isinstance(action, dict) else getattr(action, "query", None)
                if query:
                    queries.append(query)
            elif itype == "message":
                for part in getattr(item, "content", None) or []:
                    for ann in getattr(part, "annotations", None) or []:
                        if getattr(ann, "type", None) == "url_citation":
                            url = getattr(ann, "url", "") or ""
                            if url and url not in seen:
                                seen.add(url)
                                citations.append({"url": url, "title": getattr(ann, "title", "") or url})

        text = (getattr(resp, "output_text", "") or "").strip()
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
        """Responses call in JSON-object mode.

        ``schema`` and ``temperature`` are accepted for interface parity; the JSON
        shape is steered via the system prompt. ``use_reasoning_model`` toggles
        reasoning effort (heavier for the main reasoning passes, lighter for the
        cheap auxiliary ones).
        """
        effort = self._cfg.reason_effort if use_reasoning_model else self._cfg.light_effort
        resp = self._client.responses.create(
            model=self._cfg.reasoning_model,
            input=[
                {"role": "system", "content": _JSON_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            reasoning={"effort": effort},
            text={"format": {"type": "json_object"}},
            max_output_tokens=self._cfg.reason_max_output_tokens,
        )
        raw = getattr(resp, "output_text", "") or ""
        return _parse_json(raw or "{}")
