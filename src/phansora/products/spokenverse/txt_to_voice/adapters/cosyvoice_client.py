"""CosyVoice 2 TTS adapter — the project's TTS engine (Apache-2.0, commercial-OK).

CosyVoice 2 is a zero-shot voice-cloning TTS. It clones from a short reference clip
and, for best quality, that clip's transcript (``prompt_text``). It also supports a
natural-language *style* instruction (e.g. "speak cheerfully") via its instruct mode.
It is run **in-process** from a CosyVoice checkout (it is not a pip package).

Modes are chosen automatically per request:
    * ``style`` given            -> ``inference_instruct2`` (clone + style control)
    * ``prompt_text`` given      -> ``inference_zero_shot``  (clone, same-language)
    * neither                    -> ``inference_cross_lingual`` (clone, ref-free)

Exposes the backend surface used by ``adapters.backend``:
    * ``synthesize_to_file(...)`` — async, writes a WAV to ``out_path``
    * ``_discover_voices_sync()`` — list selectable presets ("default")
    * ``list_voices()`` — print them

Install (prod): clone CosyVoice, install its requirements (+ ``pynini`` via conda) and
download the CosyVoice2-0.5B checkpoints, then point the app at the checkout:

    COSYVOICE_REPO=/path/to/CosyVoice

Model dir defaults to ``<repo>/pretrained_models/CosyVoice2-0.5B`` (override with
COSYVOICE_MODEL_DIR). The built-in "default" voice needs a reference clip too — set
COSYVOICE_DEFAULT_REF (+ optional COSYVOICE_DEFAULT_REF_TEXT), else only cloned voices
work.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import uuid
import wave
from pathlib import Path
from threading import Lock
from typing import Optional

_MODEL_LOCK = Lock()
_TTS = None  # cached CosyVoice2 instance (one per process)
_INFER_LOCK = Lock()

# Audio suffixes we treat as "this argument is a reference clip to clone".
_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
_DEFAULT_VOICES = ["default"]

# Languages we surface (used for reference transcription + metadata; CosyVoice infers
# language from the text itself, so this is not passed to the model directly).
LANGUAGES = ["en", "zh", "ja", "ko", "yue", "auto"]
LANGUAGE_DEFAULT = "en"

# Generation-knob ranges (kept in sync with voices.clamp_settings / the UI).
SPEED_MIN, SPEED_MAX, SPEED_DEFAULT = 0.5, 2.0, 1.0
# Free-text style/instruct prompt (CosyVoice2 instruct mode). Empty => plain clone.
STYLE_MAX_LEN = 200


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, "").strip() or default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def _resolve_language(language: Optional[str]) -> str:
    lang = (language or "").strip().lower()
    if not lang:
        lang = _env("COSYVOICE_LANGUAGE", LANGUAGE_DEFAULT).lower()
    return lang if lang in LANGUAGES else LANGUAGE_DEFAULT


def _repo() -> Path:
    repo = _env("COSYVOICE_REPO")
    if not repo:
        raise RuntimeError(
            "CosyVoice is not configured. Set COSYVOICE_REPO to your CosyVoice "
            "checkout (with requirements installed + CosyVoice2-0.5B checkpoints "
            "downloaded)."
        )
    return Path(repo)


def _model_dir(repo: Path) -> str:
    return _env("COSYVOICE_MODEL_DIR", str(repo / "pretrained_models" / "CosyVoice2-0.5B"))


def _load_tts(use_gpu: bool):
    global _TTS
    with _MODEL_LOCK:
        if _TTS is not None:
            return _TTS
        repo = _repo()
        # CosyVoice is a repo, not a package, and its code imports the vendored
        # Matcha-TTS from third_party/. Put both on sys.path before importing.
        for p in (str(repo), str(repo / "third_party" / "Matcha-TTS")):
            if p not in sys.path:
                sys.path.insert(0, p)
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2  # type: ignore

            model_dir = _model_dir(repo)
            fp16 = _env_bool("COSYVOICE_FP16", False) and (use_gpu and _cuda_available())
            _TTS = CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=fp16)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Could not import/load CosyVoice2 from "
                f"{repo} — check COSYVOICE_REPO and that its requirements are "
                f"installed.\nOriginal error: {type(e).__name__}: {e}"
            ) from e
        return _TTS


# CosyVoice works best with a short prompt clip; trim over-long references defensively
# (uploads are already capped in voices.py, but the default ref / direct paths may not be).
_REF_MAX_SECONDS = 20.0
_REF_TRIM_SECONDS = 15.0


def _probe_seconds(path: str) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return float((out.stdout or "0").strip() or 0)
    except Exception:
        return 0.0


def _ensure_ref_length(ref_clip: str) -> tuple[str, Optional[str]]:
    """If ``ref_clip`` is longer than ``_REF_MAX_SECONDS``, trim the first
    ``_REF_TRIM_SECONDS`` into a temp WAV. Returns (path_to_use, temp_to_delete_or_None).
    Covers every reference source (cloned voices, the default ref, direct paths)."""
    if _probe_seconds(ref_clip) <= _REF_MAX_SECONDS:
        return ref_clip, None
    tmp = str(Path(tempfile.gettempdir()) / f"cosyref_{uuid.uuid4().hex}.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(ref_clip), "-t", str(_REF_TRIM_SECONDS),
             "-ac", "1", "-ar", "16000", tmp],
            check=True, capture_output=True, timeout=60,
        )
        return tmp, tmp
    except Exception:
        return ref_clip, None  # best-effort; let the engine surface any error


def _resolve_reference(voice: str, speaker: Optional[str], ref_audio: Optional[str]) -> Optional[str]:
    for candidate in (ref_audio, speaker, voice, os.getenv("COSYVOICE_REF_AUDIO")):
        if not candidate:
            continue
        c = str(candidate).strip()
        if not c:
            continue
        p = Path(c)
        if p.suffix.lower() in _AUDIO_SUFFIXES and p.is_file():
            return str(p)
    return None


def _write_wav(out_path: Path, samples, sample_rate: int) -> None:
    """Write CosyVoice output (float array/tensor) as a mono PCM WAV."""
    import numpy as np  # type: ignore

    arr = np.asarray(samples).flatten()
    if arr.dtype != np.int16:
        arr = np.clip(np.asarray(arr, dtype="float32"), -1.0, 1.0)
        arr = (arr * 32767.0).astype("<i2")
    pcm = arr.astype("<i2").tobytes()
    if not pcm:
        raise RuntimeError("CosyVoice synthesis returned empty audio.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm)


def _synthesize_sync(
    text: str,
    out_path: Path,
    voice: str,
    use_gpu: bool,
    rate: str,
    volume: str,
    speaker: Optional[str],
    language: Optional[str],
    ref_audio: Optional[str],
    prompt_text: Optional[str] = None,
    speed: Optional[float] = None,
    style: Optional[str] = None,
) -> None:
    _ = (rate, volume, language)  # CosyVoice uses speed + text content, not rate/volume/lang.
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty text; nothing to synthesize.")

    import torch  # type: ignore

    ref_clip = _resolve_reference(voice, speaker, ref_audio)
    if not ref_clip:
        # No cloned voice → fall back to a configured default reference.
        ref_clip = _env("COSYVOICE_DEFAULT_REF")
        if not prompt_text:
            prompt_text = _env("COSYVOICE_DEFAULT_REF_TEXT")
    if not ref_clip:
        raise RuntimeError(
            "CosyVoice needs a reference clip. Select a cloned voice, or set "
            "COSYVOICE_DEFAULT_REF (+ COSYVOICE_DEFAULT_REF_TEXT) for the default voice."
        )
    ref_clip, _ref_tmp = _ensure_ref_length(ref_clip)

    # Clamp/normalize knobs; None => env/default.
    speed = max(SPEED_MIN, min(SPEED_MAX, float(speed if speed is not None else _env_float("COSYVOICE_SPEED", SPEED_DEFAULT))))
    prompt_text = (prompt_text or "").strip()
    style = (style or "").strip()[:STYLE_MAX_LEN]
    text_frontend = _env_bool("COSYVOICE_TEXT_FRONTEND", True)

    tts = _load_tts(use_gpu)

    # Load the prompt clip at 16 kHz (CosyVoice's expected prompt rate).
    from cosyvoice.utils.file_utils import load_wav  # type: ignore

    sr, chunks = None, []
    try:
        with _INFER_LOCK:
            prompt_speech_16k = load_wav(ref_clip, 16000)
            sr = int(getattr(tts, "sample_rate", 24000))
            if style:
                gen = tts.inference_instruct2(
                    text, style, prompt_speech_16k, stream=False, speed=speed, text_frontend=text_frontend,
                )
            elif prompt_text:
                gen = tts.inference_zero_shot(
                    text, prompt_text, prompt_speech_16k, stream=False, speed=speed, text_frontend=text_frontend,
                )
            else:
                gen = tts.inference_cross_lingual(
                    text, prompt_speech_16k, stream=False, speed=speed, text_frontend=text_frontend,
                )
            for out in gen:
                chunks.append(out["tts_speech"])
    finally:
        if _ref_tmp:
            Path(_ref_tmp).unlink(missing_ok=True)

    if not chunks:
        raise RuntimeError("CosyVoice synthesis produced no audio.")
    audio = torch.concat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
    _write_wav(out_path, audio.detach().cpu().numpy(), sr or 24000)


async def synthesize_to_file(
    text: str,
    out_path: Path,
    voice: str,  # reference-clip path for cloning, or a preset name ("default")
    use_gpu: bool,
    rate: str,  # accepted for interface parity; ignored by CosyVoice
    volume: str,  # accepted for interface parity; ignored by CosyVoice
    speaker: Optional[str] = None,  # optional alias; treated like voice
    language: Optional[str] = None,  # en/zh/ja/ko/yue/auto; used for ref transcription/metadata
    ref_audio: Optional[str] = None,  # explicit reference clip for cloning
    prompt_text: Optional[str] = None,  # reference transcript (better quality; None => ref-free)
    speed: Optional[float] = None,  # 0.5-2.0
    style: Optional[str] = None,  # natural-language style, e.g. "speak cheerfully"
) -> None:
    await asyncio.to_thread(
        _synthesize_sync,
        text, out_path, voice, use_gpu, rate, volume, speaker, language, ref_audio,
        prompt_text, speed, style,
    )


def _discover_voices_sync() -> list[str]:
    return list(_DEFAULT_VOICES)


async def list_voices() -> None:
    for voice in await asyncio.to_thread(_discover_voices_sync):
        print(voice)
    print(
        "\n(CosyVoice clones a voice from a reference clip — pass --ref-audio "
        "/path/to/sample.wav. Quality is best with the clip's transcript; add "
        "--style \"speak cheerfully\" for instruct-mode style control.)"
    )
    print("Languages: " + ", ".join(LANGUAGES))
