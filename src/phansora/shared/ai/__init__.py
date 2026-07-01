"""Shared AI clients.

    anthropic — Claude grounded web search + JSON reasoning (used by chrono_origin)
    deepseek  — DeepSeek chat / OCR-cleanup helper (used by spokenverse)
"""
from .anthropic import AnthropicClient, AnthropicConfig, GroundedAnswer

__all__ = ["AnthropicClient", "AnthropicConfig", "GroundedAnswer"]
