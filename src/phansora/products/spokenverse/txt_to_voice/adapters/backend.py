"""TTS engine selection.

CosyVoice2 is the sole engine. The selector is kept as a thin indirection so the
rest of the pipeline stays engine-agnostic (and a second engine could be added
later), but there is only one implementation today.

The engine module exposes: ``synthesize_to_file``, ``_discover_voices_sync``,
``list_voices`` and ``preload``.
"""

from __future__ import annotations

import logging
import os
from typing import Awaitable, Callable

LOG = logging.getLogger("txt_to_voice")

_COSYVOICE2_ALIASES = {
    "cosyvoice2", "cosyvoice", "cosy-voice", "cosy_voice", "cosy", "cosyvoice-2",
    "default", "",
}
# Engines that used to exist here; a leftover TTS_ENGINE / --engine pointing at one
# degrades to CosyVoice2 with a warning rather than crashing every request.
_RETIRED_ALIASES = {
    "indextts2", "indextts", "index-tts2", "index_tts2", "index-tts", "indextts-2",
    "gptsovits", "gpt-sovits", "gpt_sovits", "sovits", "gsv",
    "styletts2", "styletts-2", "style", "stts2", "st2",
    "kokoro", "openvoice", "chatterbox", "xtts",
}

_warned_retired: set[str] = set()


def resolve_engine(engine: str | None = None) -> str:
    name = (engine if engine is not None else os.getenv("TTS_ENGINE", "")).strip().lower()
    if name in _RETIRED_ALIASES:
        if name not in _warned_retired:
            LOG.warning(
                "TTS engine '%s' has been removed; using CosyVoice2. "
                "Update TTS_ENGINE / --engine to 'cosyvoice2' to silence this.",
                name,
            )
            _warned_retired.add(name)
        return "cosyvoice2"
    # Everything else (cosyvoice2 aliases, unknown values) resolves to the only engine.
    return "cosyvoice2"


def _module(engine: str | None):
    resolve_engine(engine)  # validate (warns on retired engines)
    from . import cosyvoice2_client as mod  # type: ignore
    return mod


def get_synthesizer(engine: str | None = None) -> Callable[..., Awaitable[None]]:
    """Return the active engine's async ``synthesize_to_file``."""
    return _module(engine).synthesize_to_file


def discover_voices(engine: str | None = None) -> list[str]:
    return _module(engine)._discover_voices_sync()


async def list_voices(engine: str | None = None) -> None:
    await _module(engine).list_voices()


def preload(engine: str | None = None) -> None:
    """Load the active engine's model once (called at FastAPI startup)."""
    _module(engine).preload()
