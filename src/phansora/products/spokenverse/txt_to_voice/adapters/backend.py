"""TTS engine selection.

StyleTTS2 is the sole engine. The selector is kept as a thin indirection so the
rest of the pipeline stays engine-agnostic (and a second engine could be added
later), but there is only one implementation today.

The engine module exposes: ``synthesize_to_file``, ``_discover_voices_sync`` and
``list_voices``.
"""

from __future__ import annotations

import os
from typing import Awaitable, Callable

_STYLETTS2_ALIASES = {"styletts2", "styletts-2", "style", "stts2", "st2", "default", ""}


def resolve_engine(engine: str | None = None) -> str:
    name = (engine if engine is not None else os.getenv("TTS_ENGINE", "")).strip().lower()
    if name in _STYLETTS2_ALIASES:
        return "styletts2"
    if name == "kokoro":
        raise RuntimeError(
            "The Kokoro engine has been removed; StyleTTS2 is the only TTS engine. "
            "Unset TTS_ENGINE / drop --engine kokoro."
        )
    # Unknown value: fall back to the only engine rather than crash mid-run.
    return "styletts2"


def _module(engine: str | None):
    resolve_engine(engine)  # validate (raises on kokoro)
    from . import styletts2_client as mod  # type: ignore
    return mod


def get_synthesizer(engine: str | None = None) -> Callable[..., Awaitable[None]]:
    """Return the active engine's async ``synthesize_to_file``."""
    return _module(engine).synthesize_to_file


def discover_voices(engine: str | None = None) -> list[str]:
    return _module(engine)._discover_voices_sync()


async def list_voices(engine: str | None = None) -> None:
    await _module(engine).list_voices()
