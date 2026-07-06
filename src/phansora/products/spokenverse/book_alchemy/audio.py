"""Render a session script to a single audio file via the IndexTTS2 TTS engine.

We reuse ``BatchConverter`` (which already handles chunking, concatenation and
transcoding) by writing the script to a temporary input folder and collecting
the single produced file — the same pattern server.py uses for /txt-to-audio.
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from phansora.products.spokenverse.txt_to_voice.pipeline import BatchConverter, TTSConfig


async def render_script_to_audio(
    *,
    script: str,
    out_path: Path,
    voice: str = "default",
    use_gpu: Optional[bool] = None,
    rate: str = "+0%",
    output_format: str = "mp3",
) -> int:
    """Synthesize ``script`` to ``out_path``. Returns duration in whole seconds."""
    script = (script or "").strip()
    if not script:
        raise ValueError("Empty script; nothing to synthesize.")

    import os

    # IndexTTS2 uses CUDA automatically when available; there's nothing to resolve
    # per call here.
    if use_gpu is None:
        use_gpu = False

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ba_tts_") as tmp:
        tmp_in = Path(tmp) / "in"
        tmp_out = Path(tmp) / "out"
        tmp_in.mkdir(parents=True, exist_ok=True)
        tmp_out.mkdir(parents=True, exist_ok=True)
        (tmp_in / "session.txt").write_text(script, encoding="utf-8")

        cfg = TTSConfig(
            voice=voice,
            use_gpu=bool(use_gpu),
            rate=rate,
            volume="+0%",
            output_format=output_format,
            speaker=voice,
            max_concurrency=int(os.getenv("BOOK_ALCHEMY_TTS_CONCURRENCY", "4")),
            file_concurrency=1,
        )
        rc = await BatchConverter(cfg).convert_folder(tmp_in, tmp_out)
        if rc != 0:
            raise RuntimeError(f"TTS conversion failed (rc={rc}).")

        produced = next((p for p in tmp_out.rglob(f"*.{output_format}") if p.is_file()), None)
        if produced is None:
            raise RuntimeError("TTS produced no audio file.")
        shutil.move(str(produced), str(out_path))

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


# Convenience for callers that are already inside a running loop vs. not.
def render_sync(**kwargs) -> int:
    return asyncio.run(render_script_to_audio(**kwargs))
