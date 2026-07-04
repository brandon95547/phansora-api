"""GPT-SoVITS TTS adapter — the project's TTS engine.

GPT-SoVITS is a zero-shot (and few-shot) voice-cloning TTS. It clones from a short
reference clip and, for best quality, the reference's transcript (``prompt_text``);
without a transcript it falls back to reference-free mode. It is run **in-process**
via ``TTS_infer_pack`` from a GPT-SoVITS checkout (it is not a pip package).

Exposes the backend surface used by ``adapters.backend``:

    * ``synthesize_to_file(...)`` — async, writes a WAV to ``out_path``
    * ``_discover_voices_sync()`` — list selectable presets ("default")
    * ``list_voices()`` — print them

Install (prod only): clone GPT-SoVITS, install its requirements + the v2 model
checkpoints, and point the app at the checkout:

    GPTSOVITS_REPO=/path/to/GPT-SoVITS

Model paths default to that repo's ``GPT_SoVITS/pretrained_models`` (override with
GPTSOVITS_T2S / GPTSOVITS_VITS / GPTSOVITS_BERT / GPTSOVITS_HUBERT). The built-in
"default" voice needs a reference clip too — set GPTSOVITS_DEFAULT_REF (+ optional
GPTSOVITS_DEFAULT_REF_TEXT), otherwise only cloned voices work.
"""

from __future__ import annotations

import asyncio
import os
import sys
import wave
from pathlib import Path
from threading import Lock
from typing import Optional

_MODEL_LOCK = Lock()
_TTS = None  # cached TTS instance (one per process)
_INFER_LOCK = Lock()

# Audio suffixes we treat as "this argument is a reference clip to clone".
_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
_DEFAULT_VOICES = ["default"]

# Languages GPT-SoVITS v2 handles.
LANGUAGES = ["en", "zh", "ja", "ko", "yue", "auto"]
LANGUAGE_DEFAULT = "en"

# Generation-knob ranges (kept in sync with voices.clamp_settings / the UI).
SPEED_MIN, SPEED_MAX, SPEED_DEFAULT = 0.6, 1.65, 1.0
TOP_K_MIN, TOP_K_MAX, TOP_K_DEFAULT = 1, 100, 5
TOP_P_MIN, TOP_P_MAX, TOP_P_DEFAULT = 0.0, 1.0, 1.0
TEMPERATURE_MIN, TEMPERATURE_MAX, TEMPERATURE_DEFAULT = 0.01, 1.0, 1.0
REPETITION_PENALTY_MIN, REPETITION_PENALTY_MAX, REPETITION_PENALTY_DEFAULT = 0.0, 2.0, 1.35


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


def _resolve_device(use_gpu: bool) -> str:
    explicit = _env("GPTSOVITS_DEVICE")
    if explicit:
        return explicit
    if _env_bool("GPTSOVITS_USE_GPU") and _cuda_available():
        use_gpu = True
    return "cuda" if (use_gpu and _cuda_available()) else "cpu"


def _resolve_language(language: Optional[str]) -> str:
    lang = (language or "").strip().lower()
    if not lang:
        lang = _env("GPTSOVITS_LANGUAGE", LANGUAGE_DEFAULT).lower()
    return lang if lang in LANGUAGES else LANGUAGE_DEFAULT


def _repo() -> Path:
    repo = _env("GPTSOVITS_REPO")
    if not repo:
        raise RuntimeError(
            "GPT-SoVITS is not configured. Set GPTSOVITS_REPO to your GPT-SoVITS "
            "checkout (with requirements installed + v2 checkpoints downloaded)."
        )
    return Path(repo)


def _model_paths(repo: Path) -> dict:
    pm = repo / "GPT_SoVITS" / "pretrained_models"
    return {
        "t2s": _env("GPTSOVITS_T2S", str(pm / "gsv-v2final-pretrained" / "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt")),
        "vits": _env("GPTSOVITS_VITS", str(pm / "gsv-v2final-pretrained" / "s2G2333k.pth")),
        "bert": _env("GPTSOVITS_BERT", str(pm / "chinese-roberta-wwm-ext-large")),
        "hubert": _env("GPTSOVITS_HUBERT", str(pm / "chinese-hubert-base")),
    }


def _load_tts(use_gpu: bool):
    global _TTS
    with _MODEL_LOCK:
        if _TTS is not None:
            return _TTS
        repo = _repo()
        # GPT-SoVITS is a repo, not a package — make its modules importable.
        for p in (str(repo), str(repo / "GPT_SoVITS")):
            if p not in sys.path:
                sys.path.insert(0, p)
        try:
            from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Could not import GPT-SoVITS TTS_infer_pack from "
                f"{repo} — check GPTSOVITS_REPO and that its requirements are "
                f"installed.\nOriginal error: {type(e).__name__}: {e}"
            ) from e

        mp = _model_paths(repo)
        device = _resolve_device(use_gpu)
        # is_half=False by default: fp16 can produce silent/NaN output on some GPUs.
        is_half = _env_bool("GPTSOVITS_IS_HALF", False) and device == "cuda"
        _TTS = TTS(TTS_Config({"custom": {
            "device": device,
            "is_half": is_half,
            "version": _env("GPTSOVITS_VERSION", "v2"),
            "t2s_weights_path": mp["t2s"],
            "vits_weights_path": mp["vits"],
            "bert_base_path": mp["bert"],
            "cnhuhbert_base_path": mp["hubert"],
        }}))
        return _TTS


def _resolve_reference(voice: str, speaker: Optional[str], ref_audio: Optional[str]) -> Optional[str]:
    for candidate in (ref_audio, speaker, voice, os.getenv("GPTSOVITS_REF_AUDIO")):
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
    """Write GPT-SoVITS output (int16 or float array) as a mono PCM WAV."""
    import numpy as np  # type: ignore

    arr = np.asarray(samples).flatten()
    if arr.dtype != np.int16:
        arr = np.clip(np.asarray(arr, dtype="float32"), -1.0, 1.0)
        arr = (arr * 32767.0).astype("<i2")
    pcm = arr.astype("<i2").tobytes()
    if not pcm:
        raise RuntimeError("GPT-SoVITS synthesis returned empty audio.")
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
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
) -> None:
    _ = (rate, volume)  # GPT-SoVITS uses speed_factor, not rate/volume.
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty text; nothing to synthesize.")

    import numpy as np  # type: ignore

    lang = _resolve_language(language)
    ref_clip = _resolve_reference(voice, speaker, ref_audio)
    if not ref_clip:
        # No cloned voice → fall back to a configured default reference.
        ref_clip = _env("GPTSOVITS_DEFAULT_REF")
        if not prompt_text:
            prompt_text = _env("GPTSOVITS_DEFAULT_REF_TEXT")
    if not ref_clip:
        raise RuntimeError(
            "GPT-SoVITS needs a reference clip. Select a cloned voice, or set "
            "GPTSOVITS_DEFAULT_REF (+ GPTSOVITS_DEFAULT_REF_TEXT) for the default voice."
        )

    # Clamp knobs to supported ranges; None => env/default.
    speed = max(SPEED_MIN, min(SPEED_MAX, float(speed if speed is not None else _env_float("GPTSOVITS_SPEED", SPEED_DEFAULT))))
    top_k = int(max(TOP_K_MIN, min(TOP_K_MAX, int(top_k) if top_k is not None else TOP_K_DEFAULT)))
    top_p = max(TOP_P_MIN, min(TOP_P_MAX, float(top_p) if top_p is not None else TOP_P_DEFAULT))
    temperature = max(TEMPERATURE_MIN, min(TEMPERATURE_MAX, float(temperature) if temperature is not None else TEMPERATURE_DEFAULT))
    repetition_penalty = max(REPETITION_PENALTY_MIN, min(REPETITION_PENALTY_MAX, float(repetition_penalty) if repetition_penalty is not None else REPETITION_PENALTY_DEFAULT))

    tts = _load_tts(use_gpu)

    prompt_text = (prompt_text or "").strip()
    inputs = {
        "text": text,
        "text_lang": lang,
        "ref_audio_path": ref_clip,
        "prompt_text": prompt_text,
        "prompt_lang": lang,
        "ref_free": not bool(prompt_text),  # reference-free when no transcript
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "speed_factor": speed,
        "text_split_method": _env("GPTSOVITS_SPLIT", "cut5"),
        "batch_size": 1,
    }

    sr, chunks = None, []
    with _INFER_LOCK:
        for sr, audio in tts.run(inputs):
            chunks.append(audio)
    if not chunks:
        raise RuntimeError("GPT-SoVITS synthesis produced no audio.")
    audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    _write_wav(out_path, audio, sr or 32000)


async def synthesize_to_file(
    text: str,
    out_path: Path,
    voice: str,  # reference-clip path for cloning, or a preset name ("default")
    use_gpu: bool,
    rate: str,  # accepted for interface parity; ignored by GPT-SoVITS
    volume: str,  # accepted for interface parity; ignored by GPT-SoVITS
    speaker: Optional[str] = None,  # optional alias; treated like voice
    language: Optional[str] = None,  # en/zh/ja/ko/yue/auto; None => default (en)
    ref_audio: Optional[str] = None,  # explicit reference clip for cloning
    prompt_text: Optional[str] = None,  # reference transcript (better quality; None => ref-free)
    speed: Optional[float] = None,  # 0.6-1.65; speed_factor
    top_k: Optional[int] = None,  # 1-100; GPT sampling
    top_p: Optional[float] = None,  # 0-1; nucleus sampling
    temperature: Optional[float] = None,  # 0.01-1.0
    repetition_penalty: Optional[float] = None,  # 0-2
) -> None:
    await asyncio.to_thread(
        _synthesize_sync,
        text, out_path, voice, use_gpu, rate, volume, speaker, language, ref_audio,
        prompt_text, speed, top_k, top_p, temperature, repetition_penalty,
    )


def _discover_voices_sync() -> list[str]:
    return list(_DEFAULT_VOICES)


async def list_voices() -> None:
    for voice in await asyncio.to_thread(_discover_voices_sync):
        print(voice)
    print(
        "\n(GPT-SoVITS clones a voice from a reference clip — pass --ref-audio "
        "/path/to/sample.wav. Quality is best with the clip's transcript; without "
        "one it runs reference-free.)"
    )
    print("Languages: " + ", ".join(LANGUAGES))
