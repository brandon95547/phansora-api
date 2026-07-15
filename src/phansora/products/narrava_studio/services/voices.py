"""Preset narration voices for Narrava Studio.

These are the curated, ready-to-use voices exposed in the Studio UI. Synthesis
runs through the existing CosyVoice2 engine (see products/spokenverse); each preset
maps to a CosyVoice reference/speaker. The list is intentionally static for now —
Brandon is adding the actual preset reference clips to the engine; until then the
ids here are the contract the frontend selects against.
"""
from __future__ import annotations

from typing import List

from ..models import VoicePreset

PRESET_VOICES: List[VoicePreset] = [
    VoicePreset(
        id="narrator_evan",
        name="Evan — Documentary",
        description="Warm, measured male narrator. Great for explainers and docs.",
        language="en",
        gender="male",
        tags=["documentary", "warm", "calm"],
    ),
    VoicePreset(
        id="narrator_mara",
        name="Mara — Storyteller",
        description="Bright, expressive female voice with a storytelling cadence.",
        language="en",
        gender="female",
        tags=["storytelling", "expressive", "bright"],
    ),
    VoicePreset(
        id="narrator_cole",
        name="Cole — Trailer",
        description="Deep, dramatic delivery suited to trailers and hype reels.",
        language="en",
        gender="male",
        tags=["dramatic", "deep", "trailer"],
    ),
    VoicePreset(
        id="narrator_isla",
        name="Isla — Conversational",
        description="Friendly, casual female voice for social and tutorial content.",
        language="en",
        gender="female",
        tags=["conversational", "friendly", "social"],
    ),
]


def list_voices() -> List[VoicePreset]:
    return PRESET_VOICES
