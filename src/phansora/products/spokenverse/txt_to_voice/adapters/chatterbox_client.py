"""Chatterbox (Resemble AI) TTS adapter — the project's TTS engine.

Chatterbox is an MIT-licensed, zero-shot voice-cloning TTS. A voice is cloned
directly from a short reference clip passed as ``audio_prompt_path``; there is no
per-voice training step. Delivery is shaped by a handful of generation knobs
(exaggeration, cfg_weight, temperature, and the usual sampling controls).

Exposes the backend surface used by ``adapters.backend``:

    * ``synthesize_to_file(...)`` — async, writes a mono 24 kHz WAV to ``out_path``
    * ``_discover_voices_sync()`` — list selectable presets ("default")
    * ``list_voices()`` — print them

The reference clip for cloning is chosen, in priority order:

    1. an explicit ``ref_audio`` path passed by the caller / ``--ref-audio``
    2. ``speaker`` or ``voice`` when it points at an existing audio file
    3. the ``CHATTERBOX_REF_AUDIO`` environment variable
    4. otherwise no cloning — Chatterbox's built-in default voice

Install (prod only — Chatterbox pins torch 2.6, so it does NOT run on machines
capped at older torch, e.g. Intel macOS):

    pip install chatterbox-tts

Model weights download from HuggingFace on first use.
"""

from __future__ import annotations

import asyncio
import os
import wave
from pathlib import Path
from threading import Lock
from typing import Optional

_MODEL_LOCK = Lock()
_MODEL_CACHE: dict[str, object] = {}
_INFER_LOCK = Lock()

# Everything downstream (ffmpeg concat) expects one uniform sample rate, so the
# final clip is resampled to this regardless of the model's native rate.
_SAMPLE_RATE = 24000

# Audio suffixes we treat as "this argument is a reference clip to clone".
_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}

# Chatterbox clones from a reference clip rather than exposing a preset catalogue.
_DEFAULT_VOICES = ["default"]

# Generation-knob ranges (kept in sync with voices.clamp_settings / the UI).
EXAGGERATION_MIN, EXAGGERATION_MAX, EXAGGERATION_DEFAULT = 0.25, 2.0, 0.5
CFG_WEIGHT_MIN, CFG_WEIGHT_MAX, CFG_WEIGHT_DEFAULT = 0.0, 1.0, 0.5
TEMPERATURE_MIN, TEMPERATURE_MAX, TEMPERATURE_DEFAULT = 0.05, 5.0, 0.8
MIN_P_MIN, MIN_P_MAX, MIN_P_DEFAULT = 0.0, 1.0, 0.05
TOP_P_MIN, TOP_P_MAX, TOP_P_DEFAULT = 0.0, 1.0, 1.0
REPETITION_PENALTY_MIN, REPETITION_PENALTY_MAX, REPETITION_PENALTY_DEFAULT = 1.0, 2.0, 1.2


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def _resolve_device(use_gpu: bool) -> str:
    explicit = os.getenv("CHATTERBOX_DEVICE", "").strip()
    if explicit:
        return explicit
    if _env_bool("CHATTERBOX_USE_GPU") and _cuda_available():
        use_gpu = True
    if use_gpu and _cuda_available():
        return "cuda"
    # Apple Silicon MPS, if available and requested; otherwise CPU.
    try:
        import torch  # type: ignore

        if use_gpu and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _load_model(device: str):
    try:
        from chatterbox.tts import ChatterboxTTS  # type: ignore
    except Exception as e:  # noqa: BLE001 — surface the real cause
        raise RuntimeError(
            "Chatterbox is not installed. Install the TTS engine (prod only — it "
            "pins torch 2.6):\n  pip install chatterbox-tts\n"
            f"Original error: {type(e).__name__}: {e}"
        ) from e

    with _MODEL_LOCK:
        model = _MODEL_CACHE.get(device)
        if model is None:
            model = ChatterboxTTS.from_pretrained(device=device)
            _MODEL_CACHE[device] = model
        return model


def _resolve_reference(
    voice: str,
    speaker: Optional[str],
    ref_audio: Optional[str],
) -> Optional[str]:
    """Return a path to a reference clip for cloning, or None for the default voice."""
    for candidate in (ref_audio, speaker, voice, os.getenv("CHATTERBOX_REF_AUDIO")):
        if not candidate:
            continue
        c = str(candidate).strip()
        if not c:
            continue
        p = Path(c)
        if p.suffix.lower() in _AUDIO_SUFFIXES and p.is_file():
            return str(p)
    return None


def _write_wav(out_path: Path, samples, source_rate: int) -> None:
    """Resample a float waveform to mono 24 kHz and write it as 16-bit PCM WAV."""
    import numpy as np  # type: ignore

    arr = np.asarray(samples, dtype="float32").flatten()
    if source_rate and source_rate != _SAMPLE_RATE and arr.size:
        # Linear resample — dependency-free and fine for speech playback.
        duration = arr.size / float(source_rate)
        new_len = max(1, int(round(duration * _SAMPLE_RATE)))
        arr = np.interp(
            np.linspace(0.0, arr.size - 1, new_len, dtype="float64"),
            np.arange(arr.size),
            arr,
        ).astype("float32")
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype("<i2").tobytes()
    if not pcm:
        raise RuntimeError("Chatterbox synthesis returned empty audio.")

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
    exaggeration: Optional[float] = None,
    cfg_weight: Optional[float] = None,
    temperature: Optional[float] = None,
    min_p: Optional[float] = None,
    top_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
) -> None:
    _ = (rate, volume, language)  # Chatterbox has no rate/volume/lang knob.
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty text; nothing to synthesize.")

    device = _resolve_device(use_gpu)
    ref_clip = _resolve_reference(voice, speaker, ref_audio)

    # Per-call values (from the UI / caller) take precedence over env defaults;
    # all are clamped to Chatterbox's supported ranges.
    def pick(val, env, default, lo, hi):
        v = val if val is not None else _env_float(env, default)
        return max(lo, min(hi, float(v)))

    exaggeration = pick(exaggeration, "CHATTERBOX_EXAGGERATION", EXAGGERATION_DEFAULT, EXAGGERATION_MIN, EXAGGERATION_MAX)
    cfg_weight = pick(cfg_weight, "CHATTERBOX_CFG_WEIGHT", CFG_WEIGHT_DEFAULT, CFG_WEIGHT_MIN, CFG_WEIGHT_MAX)
    temperature = pick(temperature, "CHATTERBOX_TEMPERATURE", TEMPERATURE_DEFAULT, TEMPERATURE_MIN, TEMPERATURE_MAX)
    min_p = pick(min_p, "CHATTERBOX_MIN_P", MIN_P_DEFAULT, MIN_P_MIN, MIN_P_MAX)
    top_p = pick(top_p, "CHATTERBOX_TOP_P", TOP_P_DEFAULT, TOP_P_MIN, TOP_P_MAX)
    repetition_penalty = pick(
        repetition_penalty, "CHATTERBOX_REPETITION_PENALTY",
        REPETITION_PENALTY_DEFAULT, REPETITION_PENALTY_MIN, REPETITION_PENALTY_MAX,
    )

    model = _load_model(device)

    gen_kwargs = {
        "exaggeration": exaggeration,
        "cfg_weight": cfg_weight,
        "temperature": temperature,
        "min_p": min_p,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
    }
    if ref_clip:
        gen_kwargs["audio_prompt_path"] = ref_clip

    with _INFER_LOCK:
        wav = model.generate(text, **gen_kwargs)

    # ``generate`` returns a torch tensor shaped (1, N) at ``model.sr``.
    try:
        import torch  # type: ignore

        if isinstance(wav, torch.Tensor):
            wav = wav.detach().to("cpu").numpy()
    except Exception:
        pass

    _write_wav(out_path, wav, int(getattr(model, "sr", _SAMPLE_RATE)))


async def synthesize_to_file(
    text: str,
    out_path: Path,
    voice: str,  # reference-clip path for cloning, or a preset name ("default")
    use_gpu: bool,
    rate: str,  # accepted for interface parity; ignored by Chatterbox
    volume: str,  # accepted for interface parity; ignored by Chatterbox
    speaker: Optional[str] = None,  # optional alias; treated like voice
    language: Optional[str] = None,  # accepted for parity; ignored (English model)
    ref_audio: Optional[str] = None,  # explicit reference clip for cloning
    exaggeration: Optional[float] = None,  # 0.25-2.0; emotion/intensity
    cfg_weight: Optional[float] = None,  # 0-1; lower = slower/steadier pacing
    temperature: Optional[float] = None,  # 0.05-5.0; sampling randomness
    min_p: Optional[float] = None,  # 0-1; min-p sampling floor
    top_p: Optional[float] = None,  # 0-1; nucleus sampling
    repetition_penalty: Optional[float] = None,  # 1-2; discourage repetition
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
        exaggeration,
        cfg_weight,
        temperature,
        min_p,
        top_p,
        repetition_penalty,
    )


def _discover_voices_sync() -> list[str]:
    # Chatterbox clones from a reference clip rather than exposing a preset list.
    return list(_DEFAULT_VOICES)


async def list_voices() -> None:
    for voice in await asyncio.to_thread(_discover_voices_sync):
        print(voice)
    print(
        "\n(Chatterbox clones a voice from a reference clip — pass --ref-audio "
        "/path/to/sample.wav, or set CHATTERBOX_REF_AUDIO. Omit it to use the "
        "built-in default voice.)"
    )
