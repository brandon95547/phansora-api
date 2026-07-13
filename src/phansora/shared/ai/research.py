"""Provider-neutral research client surface for Chrono-Origin.

The orchestrator depends only on this module: a ``GroundedAnswer`` type and a
``build_research_client()`` factory. The concrete client (OpenAI or DeepSeek) is
chosen by the ``CHRONO_LLM_PROVIDER`` env var. Both clients expose the same
``grounded_search`` / ``reason_json`` methods, so the orchestrator is unchanged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class GroundedAnswer:
    text: str
    citations: List[Dict[str, str]] = field(default_factory=list)
    queries: List[str] = field(default_factory=list)


def build_research_client(provider: str | None = None):
    """Return the research client for the configured provider.

    CHRONO_LLM_PROVIDER=openai (default) -> GPT-5 Nano + native web_search tool.
    CHRONO_LLM_PROVIDER=deepseek          -> DeepSeek + external web search.
    """
    provider = (provider or os.getenv("CHRONO_LLM_PROVIDER", "openai")).strip().lower()
    if provider == "deepseek":
        # Lazy import so an OpenAI-only deployment doesn't build the search stack.
        from .deepseek_research import DeepSeekResearchClient

        return DeepSeekResearchClient()
    from .openai_research import OpenAIResearchClient

    return OpenAIResearchClient()
