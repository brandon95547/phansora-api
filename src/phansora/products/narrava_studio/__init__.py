"""Narrava Studio — AI-assisted video production.

Turn a prompt (or a pasted/uploaded narration script) into a preliminary video
timeline: the system writes/accepts a narrator-formatted script, segments it into
timed beats, then auto-sources fair-use media for each beat and lays it on a
timeline the user can edit, reorder and preview.

Reuses existing Phansora infrastructure:
  - CosyVoice2 (via SpokenVerse) for narration voices — see products/spokenverse.
  - The shared LLM providers (GPT-5 Nano default, DeepSeek fallback) — see
    services/llm.py, mirroring Chrono-Origin's ``*_LLM_PROVIDER`` switch.
"""
