"""Provider-neutral research client surface for Chrono-Origin.

The orchestrator depends only on this module: a ``GroundedAnswer`` type and a
``build_research_client()`` factory. The concrete client (DeepSeek or Anthropic)
is chosen by the ``CHRONO_LLM_PROVIDER`` env var. Both clients expose the same
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

    CHRONO_LLM_PROVIDER=deepseek (default) -> DeepSeek + external web search.
    CHRONO_LLM_PROVIDER=anthropic          -> Claude's built-in web search.
    """
    provider = (provider or os.getenv("CHRONO_LLM_PROVIDER", "deepseek")).strip().lower()
    if provider in ("anthropic", "claude"):
        # Lazy import so a DeepSeek-only deployment doesn't need the anthropic SDK.
        from .anthropic import AnthropicClient

        return AnthropicClient()
    from .deepseek_research import DeepSeekResearchClient

    return DeepSeekResearchClient()
