"""Shared AI clients.

    research  — provider-neutral Chrono-Origin surface (GroundedAnswer + factory);
                backed by openai_research (GPT-5 Nano) or deepseek_research.
    deepseek  — the one DeepSeek chat implementation (chat / chat_json and their sync
                twins) every product calls, and the OCR-cleanup helper.
"""
from .research import GroundedAnswer, build_research_client

__all__ = ["GroundedAnswer", "build_research_client"]
