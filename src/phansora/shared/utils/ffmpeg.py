# src/txt_to_voice/utils/ffmpeg_concat.py

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence

LOG = logging.getLogger(__name__)


def require_ffmpeg_if_needed(required: bool) -> None:
    if not required:
        return
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found. Install it system-wide "
            "(macOS: brew install ffmpeg | Ubuntu/Pop!_OS: sudo apt install ffmpeg)."
        )


def concat_audio_files_ffmpeg(files: Sequence[Path], final_path: Path) -> None:
    """
    Concatenate audio using ffmpeg concat demuxer.
    Works best when all chunk files share the same codec/settings.
    """
    final_path.parent.mkdir(parents=True, exist_ok=True)
    list_file = final_path.parent / f".concat_{final_path.stem}.txt"

    with list_file.open("w", encoding="utf-8") as f:
        for audio in files:
            f.write(f"file '{audio.resolve()}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        str(final_path),
    ]

    subprocess.run(cmd, check=True)
    list_file.unlink(missing_ok=True)


def transcode_audio_ffmpeg(src_path: Path, dst_path: Path) -> None:
    """Transcode a single audio file to target format inferred by extension."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    ext = dst_path.suffix.lower()
    if ext == ".mp3":
        codec_args = ["-codec:a", "libmp3lame", "-q:a", "2"]
    elif ext == ".wav":
        codec_args = ["-codec:a", "pcm_s16le"]
    else:
        raise ValueError(f"Unsupported target extension for transcode: {ext}")

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src_path),
        *codec_args,
        str(dst_path),
    ]
    subprocess.run(cmd, check=True)


# EBU R128 loudness target applied to ALL synthesized TTS/voice output, so every clip
# sits at the same professional loudness regardless of the source voice. -16 LUFS is
# the podcast/audiobook spoken-word standard (clear and present on phones/laptops).
LOUDNESS_I = -16.0    # integrated loudness, LUFS
LOUDNESS_TP = -1.0    # true-peak ceiling, dBTP
LOUDNESS_LRA = 11.0   # target loudness range, LU


def _loudnorm_measure(
    src_path: Path, i: float, tp: float, lra: float
) -> Optional[dict]:
    """Pass 1: measure ``src`` with ``loudnorm`` in analysis mode and return the measured
    values as a dict. Returns None if ffmpeg fails or prints no parsable JSON, so the
    caller can fall back to a single pass rather than failing the render."""
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats",
        "-i", str(src_path),
        "-af", f"loudnorm=I={i}:TP={tp}:LRA={lra}:print_format=json",
        "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except Exception:
        return None
    # loudnorm prints its JSON block to stderr, after the log. Take the LAST {...} so a
    # brace in an earlier log line can't derail the parse.
    err = proc.stderr or ""
    start, end = err.rfind("{"), err.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(err[start:end + 1])
    except json.JSONDecodeError:
        return None
    required = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
    if not all(k in data for k in required):
        return None
    # A fully-silent input measures as -inf, which loudnorm won't accept back as a
    # measured_* value; fall back to single pass for those.
    if any(str(data[k]).lstrip("-").startswith("inf") for k in required):
        return None
    return data


def loudnorm_audio(
    src_path: Path,
    dst_path: Path,
    *,
    i: float = LOUDNESS_I,
    tp: float = LOUDNESS_TP,
    lra: float = LOUDNESS_LRA,
    sample_rate: int = 24000,
) -> None:
    """Loudness-normalize ``src`` to an EBU R128 target and encode to ``dst`` (format
    inferred by extension). Forcing ``-ar`` keeps loudnorm from resampling to 192 kHz.
    Used as the final pass on every rendered TTS file.

    Runs loudnorm in TWO passes. Single-pass loudnorm is an adaptive real-time filter that
    only estimates the program loudness as it goes: measured output routinely lands ~1 LU
    off target and can breach the true-peak ceiling (a demo render measured -17.2 LUFS /
    -0.9 dBTP against an I=-16 / TP=-1 target). Feeding pass 1's measurements back in as
    ``measured_*`` turns pass 2 into a straight linear gain, landing within ~0.1 LU and
    actually honoring TP. If the measure pass fails for any reason we fall back to the old
    single-pass behavior — slightly off target beats no audio.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    ext = dst_path.suffix.lower()
    if ext == ".mp3":
        codec_args = ["-codec:a", "libmp3lame", "-q:a", "2"]
    elif ext == ".wav":
        codec_args = ["-codec:a", "pcm_s16le"]
    else:
        raise ValueError(f"Unsupported target extension for loudnorm: {ext}")

    af = f"loudnorm=I={i}:TP={tp}:LRA={lra}"
    measured = _loudnorm_measure(src_path, i, tp, lra)
    if measured is not None:
        af += (
            f":measured_I={measured['input_i']}"
            f":measured_TP={measured['input_tp']}"
            f":measured_LRA={measured['input_lra']}"
            f":measured_thresh={measured['input_thresh']}"
            f":offset={measured['target_offset']}"
            ":linear=true"
        )
    else:
        LOG.warning("loudnorm measure pass failed for %s; using single-pass", src_path.name)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src_path),
        "-af",
        af,
        "-ar",
        str(sample_rate),
        *codec_args,
        str(dst_path),
    ]
    subprocess.run(cmd, check=True)
