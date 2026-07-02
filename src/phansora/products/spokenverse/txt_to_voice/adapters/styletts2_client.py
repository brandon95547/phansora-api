"""StyleTTS2 TTS adapter — the project's TTS engine.

Exposes the backend surface used by ``adapters.backend``:

    * ``synthesize_to_file(...)`` — async, writes a mono 24 kHz WAV to ``out_path``
    * ``_discover_voices_sync()`` — list selectable voices
    * ``list_voices()`` — print them

StyleTTS2 supports *voice cloning* from a short reference clip. The reference
clip is chosen, in priority order:

    1. an explicit ``ref_audio`` path passed by the caller / ``--ref-audio``
    2. ``speaker`` or ``voice`` when it points at an existing audio file
    3. the ``STYLETTS2_REF_AUDIO`` environment variable
    4. otherwise the model's built-in default LibriTTS voice

Install (optional, heavy — see requirements.txt):

    pip install styletts2
    # plus the espeak-ng system package, required by the phonemizer:
    #   Debian/Ubuntu:  apt-get install espeak-ng
    #   macOS (brew):   brew install espeak-ng
"""

from __future__ import annotations

import asyncio
import os
import wave
from pathlib import Path
from threading import Lock
from typing import Optional

_MODEL_LOCK = Lock()
_MODEL_CACHE: dict[tuple[str, str], object] = {}
_INFER_LOCK = Lock()
_NLTK_READY = False


def _ensure_nltk_data() -> None:
    """StyleTTS2's tokenizer needs NLTK's punkt data; fetch it once if missing.

    Downloads to the default NLTK data dir on first run (needs network once).
    Best-effort: a failure here surfaces later as a clear tokenizer error rather
    than crashing the import.
    """
    global _NLTK_READY
    if _NLTK_READY:
        return
    try:
        import nltk  # type: ignore

        # NLTK >=3.9 renamed punkt -> punkt_tab; try both for compatibility.
        for resource in ("punkt_tab", "punkt"):
            try:
                nltk.data.find(f"tokenizers/{resource}")
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except Exception:  # noqa: BLE001 — network/permission issues are non-fatal here
                    pass
    except Exception:  # noqa: BLE001
        pass
    _NLTK_READY = True

# StyleTTS2 (LibriTTS checkpoint) synthesizes at 24 kHz; the pipeline writes every
# chunk at this rate so downstream ffmpeg concatenation stays sample-rate uniform.
_SAMPLE_RATE = 24000
# Fewest diffusion steps StyleTTS2 can run without its sampler producing NaN.
_MIN_DIFFUSION_STEPS = 3

# Audio suffixes we treat as "this voice argument is a reference clip to clone".
_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}

# The stock StyleTTS2 package ships a single default (LibriTTS) voice. Cloned
# voices are supplied at call time via a reference clip, so there is no fixed
# preset catalogue — the reference clip *is* the voice.
_DEFAULT_VOICES = ["default"]


def _load_styletts2_class():
    import importlib.util

    # Distinguish "package not installed" from "installed but its import chain broke",
    # otherwise a missing transitive/system dep gets misreported as "install styletts2".
    if importlib.util.find_spec("styletts2") is None:
        raise RuntimeError(
            "StyleTTS2 is not installed. Install the optional engine:\n"
            "  pip install \"styletts2>=0.1.6\"\n"
            "and the espeak-ng system package it phonemizes with, e.g.\n"
            "  brew install espeak-ng      # macOS\n"
            "  apt-get install espeak-ng   # Debian/Ubuntu"
        )

    try:
        from styletts2 import tts  # type: ignore
    except Exception as e:  # noqa: BLE001 — surface the real cause, don't mask it
        raise RuntimeError(
            "StyleTTS2 is installed but failed to import — usually a missing "
            "transitive/system dependency (e.g. espeak-ng not found on the library "
            "path, or a broken phonemizer/nltk install).\n"
            f"Original error: {type(e).__name__}: {e}"
        ) from e

    return tts.StyleTTS2


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _get_model(use_gpu: bool):
    StyleTTS2 = _load_styletts2_class()

    cuda = _cuda_available()
    # Single global switch: STYLETTS2_USE_GPU=1 forces GPU for *every* TTS path
    # (regular TTS, create-voice samples, Book Alchemy, CLI) without per-call edits.
    # It only turns GPU on; a per-call use_gpu=True still works on its own. When the
    # switch is set but no CUDA device is present, degrade to CPU instead of crashing.
    if _env_bool("STYLETTS2_USE_GPU") and cuda:
        use_gpu = True

    if use_gpu and not cuda:
        raise RuntimeError(
            "GPU was requested for StyleTTS2, but torch.cuda.is_available() is False. "
            "Install CUDA-enabled PyTorch on an NVIDIA machine, or run on CPU."
        )

    device = "cuda" if use_gpu else "cpu"
    # Allow pointing at a custom fine-tuned checkpoint/config; empty => package default.
    checkpoint = os.getenv("STYLETTS2_CHECKPOINT", "").strip()
    config = os.getenv("STYLETTS2_CONFIG", "").strip()
    cache_key = (device, checkpoint)

    with _MODEL_LOCK:
        model = _MODEL_CACHE.get(cache_key)
        if model is not None:
            return model

        kwargs = {}
        if checkpoint:
            kwargs["model_checkpoint_path"] = checkpoint
        if config:
            kwargs["config_path"] = config

        # Some builds pin the device internally via torch defaults; when we want
        # CPU on a CUDA box, hide the GPU for this process before construction.
        if not use_gpu and cuda:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

        model = StyleTTS2(**kwargs)
        _MODEL_CACHE[cache_key] = model
        return model


def _resolve_reference(
    voice: str,
    speaker: Optional[str],
    ref_audio: Optional[str],
) -> Optional[str]:
    """Return a path to a reference clip for cloning, or None for default voice."""
    for candidate in (ref_audio, speaker, voice, os.getenv("STYLETTS2_REF_AUDIO")):
        if not candidate:
            continue
        c = str(candidate).strip()
        if not c:
            continue
        p = Path(c)
        if p.suffix.lower() in _AUDIO_SUFFIXES and p.is_file():
            return str(p)
    return None


def _write_wav(out_path: Path, samples) -> None:
    """Write a float array in [-1, 1] as mono 16-bit PCM WAV."""
    try:
        import numpy as np  # type: ignore

        arr = np.asarray(samples, dtype="float32").flatten()
        arr = np.clip(arr, -1.0, 1.0)
        pcm = (arr * 32767.0).astype("<i2").tobytes()
    except Exception:
        # Fallback without numpy (slower, but keeps the adapter working).
        pcm = bytearray()
        flat = getattr(samples, "flatten", lambda: samples)()
        seq = flat.tolist() if hasattr(flat, "tolist") else list(flat)
        for s in seq:
            clipped = max(-1.0, min(1.0, float(s)))
            pcm.extend(int(clipped * 32767.0).to_bytes(2, byteorder="little", signed=True))
        pcm = bytes(pcm)

    if not pcm:
        raise RuntimeError("StyleTTS2 synthesis returned empty audio.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(_SAMPLE_RATE)
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
    diffusion_steps: Optional[int] = None,
    embedding_scale: Optional[float] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
) -> None:
    _ = (rate, volume, language)  # StyleTTS2 wrapper has no rate/volume/lang knob.
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty text; nothing to synthesize.")

    _ensure_nltk_data()
    target_voice = _resolve_reference(voice, speaker, ref_audio)
    model = _get_model(use_gpu)

    # Expression / quality knobs. Per-call values (from the UI / caller) take
    # precedence over the env defaults; all are clamped to StyleTTS2's ranges.
    alpha_val = alpha if alpha is not None else _env_float("STYLETTS2_ALPHA", 0.3)
    beta_val = beta if beta is not None else _env_float("STYLETTS2_BETA", 0.7)
    steps = diffusion_steps if diffusion_steps is not None else _env_int("STYLETTS2_DIFFUSION_STEPS", 10)
    scale = embedding_scale if embedding_scale is not None else _env_float("STYLETTS2_EMBEDDING_SCALE", 1.0)
    alpha = max(0.0, min(1.0, float(alpha_val)))
    beta = max(0.0, min(1.0, float(beta_val)))
    # StyleTTS2's diffusion sampler produces NaN at very low step counts (a 1-step
    # schedule divides by steps-1 == 0), which later surfaces as "cannot convert
    # float NaN to integer". Floor well above that even though the nominal min is 1.
    diffusion_steps = max(_MIN_DIFFUSION_STEPS, min(20, int(steps)))
    embedding_scale = max(0.5, min(3.0, float(scale)))

    kwargs = {
        "text": text,
        "output_sample_rate": _SAMPLE_RATE,
        "alpha": alpha,
        "beta": beta,
        "diffusion_steps": diffusion_steps,
        "embedding_scale": embedding_scale,
    }
    if target_voice:
        kwargs["target_voice_path"] = target_voice

    with _INFER_LOCK:
        audio = model.inference(**kwargs)

    if audio is None:
        raise RuntimeError("StyleTTS2 synthesis returned no audio.")

    _write_wav(out_path, audio)


async def synthesize_to_file(
    text: str,
    out_path: Path,
    voice: str,  # reference-clip path for cloning, or a preset name ("default")
    use_gpu: bool,
    rate: str,  # accepted for interface parity; ignored by StyleTTS2
    volume: str,  # accepted for interface parity; ignored by StyleTTS2
    speaker: Optional[str] = None,  # optional alias; treated like voice
    language: Optional[str] = None,  # accepted for parity; ignored
    ref_audio: Optional[str] = None,  # explicit reference clip for cloning
    diffusion_steps: Optional[int] = None,  # 1-20; fewer = faster. None => env default
    embedding_scale: Optional[float] = None,  # 0.5-3.0; higher = more expressive
    alpha: Optional[float] = None,  # 0-1; None => env default
    beta: Optional[float] = None,  # 0-1; None => env default
) -> None:
    await asyncio.to_thread(
        _synthesize_sync,
        text,
        out_path,
        voice,
        use_gpu,
        rate,
        volume,
        speaker,
        language,
        ref_audio,
        diffusion_steps,
        embedding_scale,
        alpha,
        beta,
    )


def _discover_voices_sync() -> list[str]:
    # StyleTTS2 clones from a reference clip rather than exposing a preset list.
    return list(_DEFAULT_VOICES)


async def list_voices() -> None:
    for voice in await asyncio.to_thread(_discover_voices_sync):
        print(voice)
    print(
        "\n(StyleTTS2 clones a voice from a reference clip — pass --ref-audio "
        "/path/to/sample.wav, or set STYLETTS2_REF_AUDIO. Omit it to use the "
        "built-in default LibriTTS voice.)"
    )
