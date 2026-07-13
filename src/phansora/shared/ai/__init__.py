"""Shared AI clients.

    research  — provider-neutral Chrono-Origin surface (GroundedAnswer + factory);
                backed by openai_research (GPT-5 Nano) or deepseek_research.
    deepseek  — DeepSeek chat / OCR-cleanup helper (spokenverse, book_alchemy, dossier).
"""
from .research import GroundedAnswer, build_research_client

__all__ = ["GroundedAnswer", "build_research_client"]
