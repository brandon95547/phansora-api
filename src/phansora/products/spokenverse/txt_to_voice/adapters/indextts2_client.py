"""IndexTTS2 TTS adapter — the project's TTS engine.

IndexTTS2 (index-tts/index-tts) is an autoregressive zero-shot voice-cloning TTS with
emotion control. It clones from a short reference clip (the *speaker prompt*) and can
shape emotion via an *expressiveness* weight (``emo_alpha``) and/or an 8-way *emotion
vector*. It is run **in-process** from an index-tts checkout (it is not a pip package).

License note: IndexTTS2 is non-commercial; commercial use needs a separate license from
Bilibili. This module only integrates it — licensing is the deployer's responsibility.

Emotion is applied per request:
    * ``emo_vector`` given (8 floats, not all zero) -> that emotion mix, scaled by ``emo_alpha``
    * otherwise                                     -> the speaker clip's inherent emotion, scaled by ``emo_alpha``

Speed has no native knob in IndexTTS2, so it is approximated with a pitch-preserving
ffmpeg ``atempo`` post-process (0.5-2.0).

Exposes the backend surface used by ``adapters.backend``:
    * ``synthesize_to_file(...)`` — async, writes a WAV to ``out_path``
    * ``_discover_voices_sync()`` — list selectable presets ("default")
    * ``list_voices()`` — print them

Install (prod): run ``make install`` (torch 2.8 + transformers 4.52.1), clone index-tts,
``pip install -e`` it, install ``pynini`` via conda, download the IndexTeam/IndexTTS-2
checkpoints, then point the app at the checkout:

    INDEXTTS2_REPO=/path/to/index-tts

Model dir defaults to ``<repo>/checkpoints`` (override with INDEXTTS2_MODEL_DIR); it must
hold ``config.yaml`` + the ``*.pth`` weights. The built-in "default" voice needs a
reference clip too — set INDEXTTS2_DEFAULT_REF, else only cloned voices work.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from threading import Lock
from typing import Optional, Sequence

_MODEL_LOCK = Lock()
_TTS = None  # cached IndexTTS2 instance (one per process)
_INFER_LOCK = Lock()

# Audio suffixes we treat as "this argument is a reference clip to clone".
_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
_DEFAULT_VOICES = ["default"]

# Languages we surface (used for reference transcription + metadata; IndexTTS2 infers
# language from the text itself, so this is not passed to the model directly).
LANGUAGES = ["en", "zh", "ja", "ko", "yue", "auto"]
LANGUAGE_DEFAULT = "en"

# Generation-knob ranges (kept in sync with voices.clamp_settings / the UI).
# Speed is applied as an ffmpeg atempo post-process (IndexTTS2 has no speed param).
SPEED_MIN, SPEED_MAX, SPEED_DEFAULT = 0.5, 2.0, 1.0

# Emotion controls. ``emo_alpha`` is the expressiveness weight; ``emo_vector`` is the
# 8-way mix (each 0-1) in this fixed label order.
EMO_LABELS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
EMO_VECTOR_LEN = len(EMO_LABELS)
EMO_ALPHA_MIN, EMO_ALPHA_MAX, EMO_ALPHA_DEFAULT = 0.0, 1.0, 1.0


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
        lang = _env("INDEXTTS2_LANGUAGE", LANGUAGE_DEFAULT).lower()
    return lang if lang in LANGUAGES else LANGUAGE_DEFAULT


def _repo() -> Path:
    repo = _env("INDEXTTS2_REPO")
    if not repo:
        raise RuntimeError(
            "IndexTTS2 is not configured. Set INDEXTTS2_REPO to your index-tts "
            "checkout (with `pip install -e` done + IndexTTS-2 checkpoints downloaded)."
        )
    return Path(repo)


def _model_dir(repo: Path) -> str:
    return _env("INDEXTTS2_MODEL_DIR", str(repo / "checkpoints"))


def _load_tts(use_gpu: bool):
    global _TTS
    with _MODEL_LOCK:
        if _TTS is not None:
            return _TTS
        repo = _repo()
        # index-tts is a checkout, not a package; put its root on sys.path before import.
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        try:
            from indextts.infer_v2 import IndexTTS2  # type: ignore

            model_dir = _model_dir(repo)
            cfg_path = _env("INDEXTTS2_CONFIG", str(Path(model_dir) / "config.yaml"))
            # IndexTTS2 auto-selects CUDA whenever it's available, independent of the
            # per-request `use_gpu` flag — so gate fp16 on CUDA availability + the env
            # toggle, NOT on use_gpu. fp16 roughly halves model VRAM (critical on small GPUs).
            fp16 = _env_bool("INDEXTTS2_FP16", False) and _cuda_available()
            _TTS = IndexTTS2(
                cfg_path=cfg_path,
                model_dir=model_dir,
                use_fp16=fp16,
                use_cuda_kernel=_env_bool("INDEXTTS2_USE_CUDA_KERNEL", False),
                use_deepspeed=_env_bool("INDEXTTS2_USE_DEEPSPEED", False),
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Could not import/load IndexTTS2 from "
                f"{repo} — check INDEXTTS2_REPO / INDEXTTS2_MODEL_DIR and that its "
                f"deps are installed.\nOriginal error: {type(e).__name__}: {e}"
            ) from e
        return _TTS


# IndexTTS2 works best with a short prompt clip; trim over-long references defensively
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
    ``_REF_TRIM_SECONDS`` into a temp WAV. Returns (path_to_use, temp_to_delete_or_None)."""
    if _probe_seconds(ref_clip) <= _REF_MAX_SECONDS:
        return ref_clip, None
    tmp = str(Path(tempfile.gettempdir()) / f"idx2ref_{uuid.uuid4().hex}.wav")
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
    for candidate in (ref_audio, speaker, voice, os.getenv("INDEXTTS2_REF_AUDIO")):
        if not candidate:
            continue
        c = str(candidate).strip()
        if not c:
            continue
        p = Path(c)
        if p.suffix.lower() in _AUDIO_SUFFIXES and p.is_file():
            return str(p)
    return None


def _normalize_emo_vector(emo_vector: Optional[Sequence[float]]) -> Optional[list[float]]:
    """Clamp to 8 floats in [0,1]. Returns None if absent or all-zero (=> inherent emotion)."""
    if not emo_vector:
        return None
    try:
        vals = [max(0.0, min(1.0, float(x))) for x in emo_vector]
    except (TypeError, ValueError):
        return None
    vals = (vals + [0.0] * EMO_VECTOR_LEN)[:EMO_VECTOR_LEN]
    if sum(vals) <= 0.0:
        return None
    return vals


def _atempo(src: str, out_path: Path, factor: float) -> None:
    """Pitch-preserving time-stretch to approximate a speed control. factor in [0.5, 2.0]."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-filter:a", f"atempo={factor:.4f}",
         "-ac", "1", str(out_path)],
        check=True, capture_output=True, timeout=300,
    )


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
    emo_alpha: Optional[float] = None,
    emo_vector: Optional[Sequence[float]] = None,
) -> None:
    # IndexTTS2 clones from the speaker clip + emotion; rate/volume/language/prompt_text/
    # style are accepted for interface parity but not used by the model.
    _ = (rate, volume, language, prompt_text, style)
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty text; nothing to synthesize.")

    ref_clip = _resolve_reference(voice, speaker, ref_audio)
    if not ref_clip:
        ref_clip = _env("INDEXTTS2_DEFAULT_REF")
    if not ref_clip:
        raise RuntimeError(
            "IndexTTS2 needs a reference clip. Select a cloned voice, or set "
            "INDEXTTS2_DEFAULT_REF for the default voice."
        )
    ref_clip, _ref_tmp = _ensure_ref_length(ref_clip)

    # Clamp/normalize knobs; None => env/default.
    speed = max(SPEED_MIN, min(SPEED_MAX, float(
        speed if speed is not None else _env_float("INDEXTTS2_SPEED", SPEED_DEFAULT))))
    alpha = max(EMO_ALPHA_MIN, min(EMO_ALPHA_MAX, float(
        emo_alpha if emo_alpha is not None else _env_float("INDEXTTS2_EMO_ALPHA", EMO_ALPHA_DEFAULT))))
    vec = _normalize_emo_vector(emo_vector)

    tts = _load_tts(use_gpu)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # IndexTTS2 writes the WAV itself; synth to a temp, then (optionally) atempo to out_path.
    tmp_wav = str(Path(tempfile.gettempdir()) / f"idx2_{uuid.uuid4().hex}.wav")
    try:
        kwargs = dict(
            spk_audio_prompt=ref_clip,
            text=text,
            output_path=tmp_wav,
            emo_alpha=alpha,
            verbose=False,
        )
        if vec is not None:
            kwargs["emo_vector"] = vec
        with _INFER_LOCK:
            tts.infer(**kwargs)
        if not Path(tmp_wav).is_file() or Path(tmp_wav).stat().st_size == 0:
            raise RuntimeError("IndexTTS2 synthesis produced no audio.")
        if abs(speed - 1.0) > 1e-3:
            _atempo(tmp_wav, out_path, speed)
        else:
            Path(tmp_wav).replace(out_path)
    finally:
        if _ref_tmp:
            Path(_ref_tmp).unlink(missing_ok=True)
        Path(tmp_wav).unlink(missing_ok=True)


async def synthesize_to_file(
    text: str,
    out_path: Path,
    voice: str,  # reference-clip path for cloning, or a preset name ("default")
    use_gpu: bool,
    rate: str = "+0%",  # accepted for interface parity; ignored by IndexTTS2
    volume: str = "+0%",  # accepted for interface parity; ignored by IndexTTS2
    speaker: Optional[str] = None,  # optional alias; treated like voice
    language: Optional[str] = None,  # en/zh/ja/ko/yue/auto; used for ref transcription/metadata
    ref_audio: Optional[str] = None,  # explicit reference clip for cloning
    prompt_text: Optional[str] = None,  # accepted for parity; IndexTTS2 doesn't use a transcript
    speed: Optional[float] = None,  # 0.5-2.0 (applied via ffmpeg atempo)
    style: Optional[str] = None,  # accepted for parity; not used
    emo_alpha: Optional[float] = None,  # expressiveness weight 0-1
    emo_vector: Optional[Sequence[float]] = None,  # 8-way emotion mix (each 0-1)
    **_ignored,
) -> None:
    await asyncio.to_thread(
        _synthesize_sync,
        text, out_path, voice, use_gpu, rate, volume, speaker, language, ref_audio,
        prompt_text, speed, style, emo_alpha, emo_vector,
    )


def _discover_voices_sync() -> list[str]:
    return list(_DEFAULT_VOICES)


async def list_voices() -> None:
    for voice in await asyncio.to_thread(_discover_voices_sync):
        print(voice)
    print(
        "\n(IndexTTS2 clones a voice from a reference clip — pass --ref-audio "
        "/path/to/sample.wav. Shape emotion with --emo-alpha 0-1 and/or --emo-vector "
        "(8 comma-separated 0-1 weights: " + ",".join(EMO_LABELS) + ").)"
    )
    print("Languages: " + ", ".join(LANGUAGES))
