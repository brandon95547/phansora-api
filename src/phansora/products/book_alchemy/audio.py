"""Render a session script to a single audio file via the shared TTS service.

The CosyVoice2 model is a per-process singleton with no unload path, so a second
resident copy does not fit alongside the API's copy on a 16 GB GPU. Instead of
loading its own model, the worker POSTs the script to the API's existing
``POST /spokenverse/txt-to-audio`` endpoint (the same one the frontend uses) and
saves the returned audio. That keeps a single, warmed model in the API process
and funnels all synthesis through it (the GPU serializes inference anyway).

The endpoint resolves a cloned-voice *id* -> its reference clip AND its stored
transcript (``ref_text`` -> prompt_text, which CosyVoice conditions on), so we
pass the raw voice id (not a file path).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

import aiohttp

# Base URL of the FastAPI app that owns the resident TTS model. Co-located with
# the worker in prod, so localhost by default.
_API_BASE = os.getenv("PHANSORA_API_BASE", "http://127.0.0.1:8000").rstrip("/")
# A single session can synthesize for minutes; the endpoint runs the whole job
# server-side before streaming the file back, so this must cover full synthesis.
_HTTP_TIMEOUT_S = float(os.getenv("BOOK_ALCHEMY_TTS_HTTP_TIMEOUT_S", "3600"))


async def render_script_to_audio(
    *,
    script: str,
    out_path: Path,
    user_id,
    voice: str = "default",
    output_format: str = "mp3",
    speed: Optional[float] = None,
) -> int:
    """Synthesize ``script`` to ``out_path`` via the shared TTS endpoint.

    Returns duration in whole seconds."""
    script = (script or "").strip()
    if not script:
        raise ValueError("Empty script; nothing to synthesize.")

    url = f"{_API_BASE}/spokenverse/txt-to-audio"

    form = aiohttp.FormData()
    # The endpoint keys the output filename off the uploaded stem and requires .txt.
    form.add_field("file", script.encode("utf-8"),
                   filename="session.txt", content_type="text/plain")
    form.add_field("user_id", str(user_id))
    form.add_field("voice", str(voice or "default"))
    form.add_field("output_format", output_format)
    if speed is not None:
        form.add_field("speed", str(speed))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=_HTTP_TIMEOUT_S)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, data=form) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"txt-to-audio HTTP {resp.status}: {body[:800]}")
            with open(out_path, "wb") as fh:
                async for chunk in resp.content.iter_chunked(1 << 16):
                    fh.write(chunk)

    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise RuntimeError("TTS endpoint returned no audio.")

    return _probe_duration_seconds(out_path)


def _probe_duration_seconds(path: Path) -> int:
    """Best-effort media duration via ffprobe; 0 if unavailable."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(path),
            ],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return int(float((out.stdout or "0").strip() or 0))
    except Exception:  # noqa: BLE001
        return 0
