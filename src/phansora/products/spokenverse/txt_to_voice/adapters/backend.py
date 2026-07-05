"""TTS engine selection.

CosyVoice 2 is the sole engine. The selector is kept as a thin indirection so the
rest of the pipeline stays engine-agnostic (and a second engine could be added
later), but there is only one implementation today.

The engine module exposes: ``synthesize_to_file``, ``_discover_voices_sync`` and
``list_voices``.
"""

from __future__ import annotations

import logging
import os
from typing import Awaitable, Callable

LOG = logging.getLogger("txt_to_voice")

_COSYVOICE_ALIASES = {"cosyvoice", "cosyvoice2", "cosy-voice", "cosy_voice", "cosy", "default", ""}
# Engines that used to exist here; a leftover TTS_ENGINE / --engine pointing at one
# degrades to CosyVoice with a warning rather than crashing every request.
_RETIRED_ALIASES = {
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
                "TTS engine '%s' has been removed; using CosyVoice. "
                "Update TTS_ENGINE / --engine to 'cosyvoice' to silence this.",
                name,
            )
            _warned_retired.add(name)
        return "cosyvoice"
    # Everything else (cosyvoice aliases, unknown values) resolves to the only engine.
    return "cosyvoice"


def _module(engine: str | None):
    resolve_engine(engine)  # validate (warns on retired engines)
    from . import cosyvoice_client as mod  # type: ignore
    return mod


def get_synthesizer(engine: str | None = None) -> Callable[..., Awaitable[None]]:
    """Return the active engine's async ``synthesize_to_file``."""
    return _module(engine).synthesize_to_file


def discover_voices(engine: str | None = None) -> list[str]:
    return _module(engine)._discover_voices_sync()


async def list_voices(engine: str | None = None) -> None:
    await _module(engine).list_voices()
