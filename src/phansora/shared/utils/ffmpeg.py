# src/txt_to_voice/utils/ffmpeg_concat.py

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence


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
