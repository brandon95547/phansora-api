"""CosyVoice 2 TTS adapter — the project's TTS engine.

CosyVoice 2 (FunAudioLLM/CosyVoice, ``CosyVoice2-0.5B``) is a zero-shot voice-cloning TTS.
It clones from a short reference clip (the *speaker prompt*) plus that clip's **transcript**
(``prompt_text``) — the transcript is required; CosyVoice conditions on it, unlike IndexTTS2.
It is run **in-process** from a CosyVoice checkout (it is not a pip package).

Pipeline stages: a Qwen2-0.5B LLM autoregressively emits speech tokens, a flow-matching
model turns them into a mel-spectrogram, and a HiFT vocoder renders 24 kHz audio.

Acceleration (all quality-preserving, all default-on; see the load flags below):
    * vLLM backend for the LLM  — CUDA graphs + paged attention remove the per-token CPU
      sync that otherwise starves the GPU (the dominant cost). Registers CosyVoice2's
      custom ``CosyVoice2ForCausalLM`` with vLLM before the engine loads.
    * fp16                      — half the LLM/flow memory bandwidth, ~half the VRAM.
    * TensorRT flow estimator   — fp16 engine for the flow ODE (built once, cached to disk).

Speed is a NATIVE CosyVoice2 knob (mel time-scaled at synthesis, 0.5-2.0) — no ffmpeg
post-process. There is no emotion control (removed with IndexTTS2); ``emo_*`` args are
accepted for interface parity and ignored.

Exposes the backend surface used by ``adapters.backend``:
    * ``synthesize_to_file(...)`` — async, writes a WAV to ``out_path``
    * ``_discover_voices_sync()`` — list selectable presets ("default")
    * ``list_voices()`` — print them
    * ``preload()`` — load the model ONCE (called at FastAPI startup)

Install (prod): CosyVoice is a git checkout + a model download, not a pip package. Clone
FunAudioLLM/CosyVoice (+ its Matcha-TTS submodule), install its requirements into this
venv (see requirements.txt / Makefile — the API is pinned to torch 2.7 + vllm 0.9.0 to
match), download the CosyVoice2-0.5B checkpoints, then point the app at the checkout:

    COSYVOICE2_REPO=/path/to/CosyVoice

Model dir defaults to ``<repo>/pretrained_models/CosyVoice2-0.5B`` (override with
COSYVOICE2_MODEL_DIR). The built-in "default" voice needs a reference clip + its transcript
— set COSYVOICE2_DEFAULT_REF and COSYVOICE2_DEFAULT_REF_TEXT, else only cloned voices work.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from threading import Lock
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

_MODEL_LOCK = Lock()      # guards the one-per-process model construction
_INFER_LOCK = Lock()      # serializes synthesis (the vLLM engine + flow are shared state)
_COSY = None              # cached CosyVoice2 instance
_SPK_CACHE: dict[str, str] = {}  # ref-clip signature -> cached zero-shot speaker id

# Audio suffixes we treat as "this argument is a reference clip to clone".
_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
_DEFAULT_VOICES = ["default"]

# Languages we surface (for reference transcription + metadata; CosyVoice infers language
# from the text/prompt itself, so this is not passed to the model directly).
LANGUAGES = ["en", "zh", "ja", "ko", "yue", "auto"]
LANGUAGE_DEFAULT = "en"

# Generation-knob ranges (kept in sync with voices.clamp_settings / the UI). Speed is a
# native CosyVoice2 parameter (mel time-scaling), applied at synthesis time.
SPEED_MIN, SPEED_MAX, SPEED_DEFAULT = 0.5, 2.0, 1.0

# CosyVoice2 intermittently drops/truncates words when a single inference chunk is long (the
# drop clusters at the chunk tail), and it is far worse with cloned voices. Measured on prod
# with a cloned voice + whisper transcription: 550/400 dropped whole sentences, 300 dropped
# the tail, 250 dropped words in 1/2 trials, while 200 was clean in 8/8 trials. So we cap the
# per-inference chunk at 200 chars. Input is split into <= MAX_CHARS chunks (on line and
# sentence boundaries) and the rendered audio is concatenated. Only a run longer than
# MAX_CHARS is broken mid-boundary (at a word). Override with COSYVOICE2_MAX_CHARS.
MAX_CHARS_DEFAULT = 200


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


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    try:
        return int(raw) if raw else default
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
        lang = _env("COSYVOICE2_LANGUAGE", LANGUAGE_DEFAULT).lower()
    return lang if lang in LANGUAGES else LANGUAGE_DEFAULT


def _repo() -> Path:
    repo = _env("COSYVOICE2_REPO")
    if not repo:
        raise RuntimeError(
            "CosyVoice2 is not configured. Set COSYVOICE2_REPO to your CosyVoice "
            "checkout (with its requirements installed + CosyVoice2-0.5B checkpoints "
            "downloaded)."
        )
    return Path(repo)


def _model_dir(repo: Path) -> str:
    return _env("COSYVOICE2_MODEL_DIR", str(repo / "pretrained_models" / "CosyVoice2-0.5B"))


def _load_cosy():
    """Construct the CosyVoice2 engine once per process (lock-guarded, cached).

    This is the expensive call: it loads weights, and (with vLLM) captures CUDA graphs +
    (with TRT, first run only) compiles the flow TensorRT engine. Called from ``preload``
    at startup so requests never pay it; a request that races the warmup just waits here.
    """
    global _COSY
    with _MODEL_LOCK:
        if _COSY is not None:
            return _COSY
        repo = _repo()
        # CosyVoice is a checkout, not a package; put its root + the Matcha-TTS submodule on
        # sys.path before importing (mirrors the upstream examples).
        matcha = repo / "third_party" / "Matcha-TTS"
        for p in (str(matcha), str(repo)):
            if p not in sys.path:
                sys.path.insert(0, p)
        try:
            use_fp16 = _env_bool("COSYVOICE2_FP16", True) and _cuda_available()
            use_vllm = _env_bool("COSYVOICE2_USE_VLLM", True) and _cuda_available()
            use_trt = _env_bool("COSYVOICE2_USE_TRT", True) and _cuda_available()

            # CosyVoice2's LLM is a CUSTOM vLLM architecture — it must be registered with
            # vLLM's ModelRegistry BEFORE the engine loads, or vLLM raises "Cannot find
            # model module 'CosyVoice2ForCausalLM'".
            if use_vllm:
                from vllm import ModelRegistry  # type: ignore
                from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM  # type: ignore
                ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

            from cosyvoice.cli.cosyvoice import CosyVoice2  # type: ignore

            model_dir = _model_dir(repo)
            logger.info(
                "Loading CosyVoice2 from %s (fp16=%s, vllm=%s, trt=%s) — first run also "
                "captures CUDA graphs / builds the TRT engine.",
                model_dir, use_fp16, use_vllm, use_trt,
            )
            _COSY = CosyVoice2(
                model_dir,
                load_jit=False,
                load_trt=use_trt,
                load_vllm=use_vllm,
                fp16=use_fp16,
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Could not import/load CosyVoice2 from "
                f"{repo} — check COSYVOICE2_REPO / COSYVOICE2_MODEL_DIR and that its deps "
                f"(torch 2.7 + vllm 0.9.0) are installed.\nOriginal error: "
                f"{type(e).__name__}: {e}"
            ) from e
        return _COSY


def preload() -> None:
    """Warm the engine so the first real request is fast. Two stages:
      1. Construct the model (weights + vLLM graph capture + first-time TRT build) — the
         critical win; this is the ~80s+ cold start we don't want on a user request.
      2. Run a throwaway synthesis with the default voice to warm the remaining kernels.
    Lock-guarded and cached; safe from a background thread (a racing request waits on the
    same load). Every failure is logged, never raised — stage 2 needs COSYVOICE2_DEFAULT_REF
    (+ _REF_TEXT); if unset, weights are still loaded (the big win) and warmup is skipped."""
    try:
        _load_cosy()
    except Exception as e:  # noqa: BLE001
        logger.warning("CosyVoice2 preload skipped (model load failed): %s: %s", type(e).__name__, e)
        return
    try:
        warm_out = Path(tempfile.gettempdir()) / f"cosy_warmup_{uuid.uuid4().hex}.wav"
        _synthesize_sync(
            text="Ready.",
            out_path=warm_out,
            voice="default",
            use_gpu=True,
            rate="+0%",
            volume="+0%",
            speaker=None,
            language=None,
            ref_audio=None,
        )
        warm_out.unlink(missing_ok=True)
        logger.info("CosyVoice2 preloaded + kernel-warmed (first request will be fast)")
    except Exception as e:  # noqa: BLE001
        logger.warning("CosyVoice2 weights loaded, kernel warmup skipped: %s: %s", type(e).__name__, e)


# CosyVoice works best with a short prompt clip (<= 30s hard limit; 3-10s ideal). Trim
# over-long references defensively (uploads are capped in voices.py, but the default ref /
# direct paths may not be).
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


# CosyVoice2 has no end-of-prompt token: it interleaves the prompt transcript with the
# prompt audio, so if the transcript doesn't clearly "end", the model can treat the prompt
# audio as unfinished and leak ~1s of prompt-like (often non-English) audio at the START of
# the generated clip. Guaranteeing terminal punctuation demarcates the prompt boundary and
# suppresses that leak. Our reference clips are cut at a fixed length (voices.MAX_SECONDS),
# which routinely lands mid-sentence, so the auto-transcript often lacks closing punctuation.
# See github.com/FunAudioLLM/CosyVoice issues #967 and #1704.
_TERMINAL_PUNCT = ".!?。！？…"


def _ensure_prompt_terminal(p_text: str) -> str:
    """Append a period if the reference transcript lacks sentence-final punctuation, so
    CosyVoice2 sees a clean prompt boundary (prevents leaked prompt audio at the start)."""
    p_text = (p_text or "").strip()
    if p_text and p_text[-1] not in _TERMINAL_PUNCT:
        # Use a full-width period when the text looks CJK, else an ASCII period.
        p_text += "。" if any("　" <= ch <= "鿿" for ch in p_text) else "."
    return p_text


def _resolve_reference(voice: str, speaker: Optional[str], ref_audio: Optional[str]) -> Optional[str]:
    for candidate in (ref_audio, speaker, voice, os.getenv("COSYVOICE2_REF_AUDIO")):
        if not candidate:
            continue
        c = str(candidate).strip()
        if not c:
            continue
        p = Path(c)
        if p.suffix.lower() in _AUDIO_SUFFIXES and p.is_file():
            return str(p)
    return None


def _spk_id_for(cosy, ref_clip: str, prompt_text: str) -> str:
    """Extract + cache the reference speaker ONCE, keyed by (clip, transcript).

    CosyVoice's frontend otherwise re-runs prompt extraction (speech tokenizer, campplus
    speaker embedding, speech feat) for every sentence; caching via add_zero_shot_spk turns
    that into a one-time cost per voice. Features are identical, so voice similarity is
    unchanged."""
    sig = hashlib.sha1(f"{ref_clip}\x00{prompt_text}".encode("utf-8")).hexdigest()[:16]
    cached = _SPK_CACHE.get(sig)
    if cached is not None:
        return cached
    spk_id = f"spk_{sig}"
    cosy.add_zero_shot_spk(prompt_text, ref_clip, spk_id)
    _SPK_CACHE[sig] = spk_id
    return spk_id


def _hard_split(segment: str, max_chars: int) -> list[str]:
    """Split a segment longer than ``max_chars`` at clause, then word boundaries.

    Used only when a single line/sentence exceeds the limit (e.g. punctuation-free
    verse). Never cuts mid-word; every returned piece is ``<= max_chars`` unless a
    single word is itself longer.
    """
    import re
    pieces: list[str] = []
    for clause in re.split(r"(?<=[,;:])\s+", segment):
        clause = clause.strip()
        if not clause:
            continue
        if len(clause) <= max_chars:
            pieces.append(clause)
            continue
        buf = ""
        for word in clause.split():
            if not buf:
                buf = word
            elif len(buf) + 1 + len(word) <= max_chars:
                buf = f"{buf} {word}"
            else:
                pieces.append(buf)
                buf = word
        if buf:
            pieces.append(buf)
    return pieces


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Pack lines/sentences into chunks no longer than ``max_chars``.

    Splits on blank lines, single newlines (verse lines), and sentence
    terminators, so verse (line breaks, no periods) chunks correctly. A single
    segment longer than ``max_chars`` is hard-split at clause/word boundaries.
    """
    import re
    chunks: list[str] = []
    buf = ""
    for segment in re.split(r"\n\s*\n+|\n|(?<=[.!?])\s+", text.strip()):
        segment = segment.strip()
        if not segment:
            continue
        pieces = [segment] if len(segment) <= max_chars else _hard_split(segment, max_chars)
        for piece in pieces:
            if not buf:
                buf = piece
            elif len(buf) + 1 + len(piece) <= max_chars:
                buf = f"{buf} {piece}"
            else:
                chunks.append(buf)
                buf = piece
    if buf:
        chunks.append(buf)
    return chunks or ([text.strip()] if text.strip() else [])


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
    # CosyVoice2 clones from the speaker clip + its transcript. rate/volume/language/style
    # and emo_* are accepted for interface parity but not used by the model.
    _ = (rate, volume, language, style, emo_alpha, emo_vector)
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty text; nothing to synthesize.")

    ref_clip = _resolve_reference(voice, speaker, ref_audio)
    if not ref_clip:
        ref_clip = _env("COSYVOICE2_DEFAULT_REF")
    if not ref_clip:
        raise RuntimeError(
            "CosyVoice2 needs a reference clip. Select a cloned voice, or set "
            "COSYVOICE2_DEFAULT_REF for the default voice."
        )

    # The transcript of the reference clip. CosyVoice conditions on it; for the default
    # voice fall back to COSYVOICE2_DEFAULT_REF_TEXT.
    p_text = (prompt_text or "").strip() or _env("COSYVOICE2_DEFAULT_REF_TEXT")
    if not p_text:
        raise RuntimeError(
            "CosyVoice2 needs the reference clip's transcript (prompt_text). Cloned voices "
            "store it as ref_text; for the default voice set COSYVOICE2_DEFAULT_REF_TEXT."
        )
    # Demarcate the prompt boundary so CosyVoice2 doesn't leak prompt audio at the start.
    p_text = _ensure_prompt_terminal(p_text)

    ref_clip, _ref_tmp = _ensure_ref_length(ref_clip)

    speed = max(SPEED_MIN, min(SPEED_MAX, float(
        speed if speed is not None else _env_float("COSYVOICE2_SPEED", SPEED_DEFAULT))))
    max_chars = _env_int("COSYVOICE2_MAX_CHARS", MAX_CHARS_DEFAULT)

    cosy = _load_cosy()

    import torch  # type: ignore
    import torchaudio  # type: ignore

    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunks = _chunk_text(text, max_chars)
    try:
        with _INFER_LOCK:
            spk_id = _spk_id_for(cosy, ref_clip, p_text)
            parts: list["torch.Tensor"] = []
            for chunk in chunks:
                # zero-shot via the cached speaker id (prompt_text/prompt_wav unused then).
                for out in cosy.inference_zero_shot(
                    chunk, "", "", zero_shot_spk_id=spk_id, stream=False, speed=speed
                ):
                    parts.append(out["tts_speech"])
            if not parts:
                raise RuntimeError("CosyVoice2 synthesis produced no audio.")
            wav = torch.cat(parts, dim=1)  # each tts_speech is [1, samples]
        torchaudio.save(str(out_path), wav, cosy.sample_rate)
        if not out_path.is_file() or out_path.stat().st_size == 0:
            raise RuntimeError("CosyVoice2 synthesis produced no audio.")
    finally:
        if _ref_tmp:
            Path(_ref_tmp).unlink(missing_ok=True)


async def synthesize_to_file(
    text: str,
    out_path: Path,
    voice: str,  # reference-clip path for cloning, or a preset name ("default")
    use_gpu: bool,
    rate: str = "+0%",  # accepted for interface parity; ignored by CosyVoice2
    volume: str = "+0%",  # accepted for interface parity; ignored by CosyVoice2
    speaker: Optional[str] = None,  # optional alias; treated like voice
    language: Optional[str] = None,  # en/zh/ja/ko/yue/auto; used for ref transcription/metadata
    ref_audio: Optional[str] = None,  # explicit reference clip for cloning
    prompt_text: Optional[str] = None,  # transcript of the reference clip (REQUIRED by CosyVoice)
    speed: Optional[float] = None,  # 0.5-2.0 (native CosyVoice2 mel time-scaling)
    style: Optional[str] = None,  # accepted for parity; not used
    emo_alpha: Optional[float] = None,  # accepted for parity; CosyVoice2 has no emotion control
    emo_vector: Optional[Sequence[float]] = None,  # accepted for parity; ignored
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
        "\n(CosyVoice2 clones a voice from a reference clip + its transcript — pass "
        "--ref-audio /path/to/sample.wav and the clip's text. Speed 0.5-2.0 is native.)"
    )
    print("Languages: " + ", ".join(LANGUAGES))
