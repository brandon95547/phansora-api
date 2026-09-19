"""Fun-CosyVoice 3 TTS adapter — the project's TTS engine.

Fun-CosyVoice 3 (FunAudioLLM/CosyVoice, ``Fun-CosyVoice3-0.5B-2512``) is a zero-shot
voice-cloning TTS. It clones from a short reference clip (the *speaker prompt*) plus that
clip's **transcript** (``prompt_text``) — the transcript is required; CosyVoice conditions
on it. It is run **in-process** from a CosyVoice checkout (it is not a pip package).

Pipeline stages: a Qwen2-0.5B LLM autoregressively emits speech tokens, a flow-matching
model turns them into a mel-spectrogram, and a HiFT vocoder renders 24 kHz audio.

Upgraded from CosyVoice2-0.5B. Same 0.5B parameter count and the same class surface —
``CosyVoice3`` subclasses ``CosyVoice2`` and overrides only ``__init__`` — but trained on
1M hours instead of 10k, with a new speech tokenizer (``speech_tokenizer_v3.onnx``). The
motivating defect was a word ("transformative") that v2 mispronounced identically on every
render and every voice, which is the signature of a rare token with a badly-learned
pronunciation rather than anything in our pipeline.

We run the RL post-trained LLM — CER 0.81/1.68/5.44 (zh/en/hard) against the base model's
1.21/2.24/6.71. There is no separate RL model id: the one repo ships both ``llm.pt`` and
``llm.rl.pt``, and since ``CosyVoice3.__init__`` hardcodes ``<model_dir>/llm.pt``, the
Makefile activates RL by moving the file into that name. So a model dir here holds the RL
weights under the base weights' filename — surprising if you go looking, hence this note.

Acceleration (all quality-preserving, all default-on; see the load flags below):
    * vLLM backend for the LLM  — CUDA graphs + paged attention remove the per-token CPU
      sync that otherwise starves the GPU (the dominant cost). Registers the custom
      ``CosyVoice2ForCausalLM`` architecture with vLLM before the engine loads (the class
      name is unchanged for v3 — upstream registers it and then loads a v3 checkpoint).
    * fp16                      — half the LLM/flow memory bandwidth, ~half the VRAM.
    * TensorRT flow estimator   — fp16 engine for the flow ODE (built once, cached to disk).

Speed is a NATIVE CosyVoice knob (mel time-scaled at synthesis, 0.5-2.0) — no ffmpeg
post-process.

Delivery is steered with ``instruct_text`` — a short natural-language direction ("speak
in a calm, reassuring tone") routed through ``inference_instruct2``. It shapes prosody
while the reference clip still supplies the timbre, so the clone stays on-voice. There is
no emotion *vector* (that went with IndexTTS2); ``emo_*`` args are accepted for interface
parity and ignored.

``<|endofprompt|>`` IS REQUIRED — this is the one hard break from v2. CosyVoice3's LLM
asserts the token (id 151646) is present in the conditioning text and refuses to run
without it, so every prompt we build is prefixed by ``_ensure_endofprompt``. Upstream's
conventions, which we follow: a plain clone gets ``"You are a helpful assistant.<|endofprompt|>"``
in front of the reference transcript, and an instruction gets the marker appended to
itself. Prefixing is idempotent, mirroring upstream's own triton runtime.

Exposes the backend surface used by ``adapters.backend``:
    * ``synthesize_to_file(...)`` — async, writes a WAV to ``out_path``
    * ``_discover_voices_sync()`` — list selectable presets ("default")
    * ``list_voices()`` — print them
    * ``preload()`` — load the model ONCE (called at FastAPI startup)

Install (prod): CosyVoice is a git checkout + a model download, not a pip package. Clone
FunAudioLLM/CosyVoice (+ its Matcha-TTS submodule), install its requirements into this
venv (see requirements.txt / Makefile — the API is pinned to torch 2.7 + vllm 0.9.0, which
upstream still supports for v3 alongside the newer 0.11.x V1 engine), download the
Fun-CosyVoice3 checkpoints, then point the app at the checkout:

    COSYVOICE3_REPO=/path/to/CosyVoice

Model dir defaults to ``<repo>/pretrained_models/Fun-CosyVoice3-0.5B-RL`` (override with
COSYVOICE3_MODEL_DIR). The built-in "default" voice needs a reference clip + its transcript
— set COSYVOICE3_DEFAULT_REF and COSYVOICE3_DEFAULT_REF_TEXT, else only cloned voices work.

Every COSYVOICE3_* variable falls back to its COSYVOICE2_* predecessor when unset, so an
.env written for v2 keeps working across the deploy and can be renamed afterwards rather
than in lockstep with the code.
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

from phansora.shared.utils.tts_text import normalize_for_tts

logger = logging.getLogger(__name__)

_MODEL_LOCK = Lock()      # guards the one-per-process model construction
_INFER_LOCK = Lock()      # serializes synthesis (the vLLM engine + flow are shared state)
_COSY = None              # cached CosyVoice engine instance
_SPK_CACHE: dict[str, str] = {}  # ref-clip signature -> cached zero-shot speaker id

# Audio suffixes we treat as "this argument is a reference clip to clone".
_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
_DEFAULT_VOICES = ["default"]

# Languages we surface (for reference transcription + metadata; CosyVoice infers language
# from the text/prompt itself, so this is not passed to the model directly).
LANGUAGES = ["en", "zh", "ja", "ko", "yue", "auto"]
LANGUAGE_DEFAULT = "en"

# Generation-knob ranges (kept in sync with voices.clamp_settings / the UI). Speed is a
# native CosyVoice parameter (mel time-scaling), applied at synthesis time.
SPEED_MIN, SPEED_MAX, SPEED_DEFAULT = 0.5, 2.0, 1.0

# CosyVoice2 intermittently dropped/truncated words when a single inference chunk was long (the
# drop clusters at the chunk tail), and it is far worse with cloned voices. Measured on prod
# with a cloned voice + whisper transcription: 550/400 dropped whole sentences, 300 dropped
# the tail, 250 dropped words in 1/2 trials, while 200 was clean in 8/8 trials. So we cap the
# per-inference chunk at 200 chars. Input is split into <= MAX_CHARS chunks (on line and
# sentence boundaries) and the rendered audio is concatenated. Only a run longer than
# MAX_CHARS is broken mid-boundary (at a word). Override with COSYVOICE3_MAX_CHARS.
#
# ⚠ NOT yet re-measured on v3. The numbers above are v2's, and a new speech tokenizer can
# move the threshold either way — v3 may not need a 200-char cap at all. Re-run the
# whisper-diff sweep (200/250/300/400) and raise this if it holds, since a bigger chunk is
# both faster and more natural across sentence boundaries.
MAX_CHARS_DEFAULT = 200


def _raw(name: str) -> str:
    """Read COSYVOICE3_X, falling back to the COSYVOICE2_X it replaced.

    The engine upgrade renamed every variable. Reading the old name too means a prod .env
    written for v2 survives the deploy — otherwise the rename would silently unset
    COSYVOICE2_REPO and take TTS down until someone edited .env on the box, in the middle
    of a restart. The fallback can be dropped once the deployed .env is renamed.
    """
    value = os.getenv(name, "").strip()
    if value or not name.startswith("COSYVOICE3_"):
        return value
    return os.getenv(name.replace("COSYVOICE3_", "COSYVOICE2_", 1), "").strip()


def _env(name: str, default: str = "") -> str:
    return _raw(name) or default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _raw(name).lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = _raw(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = _raw(name)
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


def _repo() -> Path:
    repo = _env("COSYVOICE3_REPO")
    if not repo:
        raise RuntimeError(
            "CosyVoice is not configured. Set COSYVOICE3_REPO to your CosyVoice "
            "checkout (with its requirements installed + Fun-CosyVoice3 checkpoints "
            "downloaded)."
        )
    return Path(repo)


MODEL_DIR_NAME = "Fun-CosyVoice3-0.5B-RL"  # what the Makefile's snapshot_download writes


def _model_dir(repo: Path) -> str:
    return _env("COSYVOICE3_MODEL_DIR", str(repo / "pretrained_models" / MODEL_DIR_NAME))


def _load_cosy():
    """Construct the CosyVoice engine once per process (lock-guarded, cached).

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
            use_fp16 = _env_bool("COSYVOICE3_FP16", True) and _cuda_available()
            use_vllm = _env_bool("COSYVOICE3_USE_VLLM", True) and _cuda_available()
            # TRT defaults OFF on v3: upstream warns "DiT tensorRT fp16 engine have some
            # performance issue, use at caution!" when loading it for CosyVoice3. It is not
            # installed either: make install-tts skips the tensorrt packages and the ONNX the
            # engine is built from, so COSYVOICE3_USE_TRT=1 fails to load until both are back.
            use_trt = _env_bool("COSYVOICE3_USE_TRT", False) and _cuda_available()

            # The LLM is a CUSTOM vLLM architecture — it must be registered with vLLM's
            # ModelRegistry BEFORE the engine loads, or vLLM raises "Cannot find model
            # module 'CosyVoice2ForCausalLM'". The class name is unchanged for v3: upstream's
            # own vllm_example.py registers CosyVoice2ForCausalLM and then loads a v3 model
            # through AutoModel, so this is correct despite reading like a leftover.
            if use_vllm:
                from vllm import ModelRegistry  # type: ignore
                from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM  # type: ignore
                ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)

            # AutoModel dispatches on which cosyvoice*.yaml the model dir contains, so a
            # future checkpoint bump needs no code change here — and pointing
            # COSYVOICE3_MODEL_DIR back at a v2 directory still works, which is the rollback.
            from cosyvoice.cli.cosyvoice import AutoModel  # type: ignore

            model_dir = _model_dir(repo)
            logger.info(
                "Loading CosyVoice from %s (fp16=%s, vllm=%s, trt=%s) — first run also "
                "captures CUDA graphs / builds the TRT engine.",
                model_dir, use_fp16, use_vllm, use_trt,
            )
            # NOTE: no load_jit here. CosyVoice3.__init__ does not accept it (CosyVoice2's
            # did); passing it raises TypeError.
            _COSY = AutoModel(
                model_dir=model_dir,
                load_trt=use_trt,
                load_vllm=use_vllm,
                fp16=use_fp16,
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "Could not import/load CosyVoice from "
                f"{repo} — check COSYVOICE3_REPO / COSYVOICE3_MODEL_DIR and that its deps "
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
    same load). Every failure is logged, never raised — stage 2 needs COSYVOICE3_DEFAULT_REF
    (+ _REF_TEXT); if unset, weights are still loaded (the big win) and warmup is skipped."""
    try:
        _load_cosy()
    except Exception as e:  # noqa: BLE001
        logger.warning("CosyVoice preload skipped (model load failed): %s: %s", type(e).__name__, e)
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
        logger.info("CosyVoice preloaded + kernel-warmed (first request will be fast)")
    except Exception as e:  # noqa: BLE001
        logger.warning("CosyVoice weights loaded, kernel warmup skipped: %s: %s", type(e).__name__, e)


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


# CosyVoice2 had no end-of-prompt token: it interleaves the prompt transcript with the
# prompt audio, so if the transcript doesn't clearly "end", the model can treat the prompt
# audio as unfinished and leak ~1s of prompt-like (often non-English) audio at the START of
# the generated clip. Guaranteeing terminal punctuation demarcates the prompt boundary and
# suppresses that leak. Our reference clips are cut at a fixed length (voices.MAX_SECONDS),
# which routinely lands mid-sentence, so the auto-transcript often lacks closing punctuation.
# See github.com/FunAudioLLM/CosyVoice issues #967 and #1704.
#
# v3 marks the boundary explicitly with <|endofprompt|>, so this is belt-and-braces now.
# Kept because it costs nothing and the leak it prevents was real; worth re-testing whether
# v3 still needs it before removing.
_TERMINAL_PUNCT = ".!?。！？…"


def _ensure_prompt_terminal(p_text: str) -> str:
    """Append a period if the reference transcript lacks sentence-final punctuation, so
    the model sees a clean prompt boundary (prevents leaked prompt audio at the start)."""
    p_text = (p_text or "").strip()
    if p_text and p_text[-1] not in _TERMINAL_PUNCT:
        # Use a full-width period when the text looks CJK, else an ASCII period.
        p_text += "。" if any("　" <= ch <= "鿿" for ch in p_text) else "."
    return p_text


# CosyVoice3 REQUIRES this marker in the conditioning text — cosyvoice/llm/llm.py asserts
# `151646 in text` and raises outright without it, so a missing marker is a hard failure on
# every request, not a quality regression. Upstream's own triton runtime prefixes only when
# absent (runtime/triton_trtllm/model_repo_cosyvoice3/cosyvoice3/1/model.py), which is the
# shape copied here.
_ENDOFPROMPT = "<|endofprompt|>"
# What upstream's example.py puts in front of a plain clone's reference transcript. It reads
# like a chat system prompt because the LLM is a Qwen2 derivative and that is the slot.
_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def _ensure_endofprompt(text: str, system_prompt: str = _DEFAULT_SYSTEM_PROMPT) -> str:
    """Guarantee the conditioning text carries ``<|endofprompt|>``.

    Idempotent: text that already contains the marker anywhere is returned untouched, so
    this is safe to apply to a caller-supplied instruction that already has one, and safe
    to apply twice.
    """
    text = (text or "").strip()
    if _ENDOFPROMPT in text:
        return text
    return f"{system_prompt}{_ENDOFPROMPT}{text}"


def _resolve_reference(voice: str, speaker: Optional[str], ref_audio: Optional[str]) -> Optional[str]:
    for candidate in (ref_audio, speaker, voice, os.getenv("COSYVOICE3_REF_AUDIO")):
        if not candidate:
            continue
        c = str(candidate).strip()
        if not c:
            continue
        p = Path(c)
        if p.suffix.lower() in _AUDIO_SUFFIXES and p.is_file():
            return str(p)
    return None


def _spk_id_for(cosy, ref_clip: str, conditioning_text: str) -> str:
    """Extract + cache the reference speaker ONCE, keyed by (clip, conditioning text).

    CosyVoice's frontend otherwise re-runs prompt extraction (speech tokenizer, campplus
    speaker embedding, speech feat) for every sentence; caching via add_zero_shot_spk turns
    that into a one-time cost per voice. Features are identical, so voice similarity is
    unchanged.

    ``conditioning_text`` is the reference transcript for plain cloning, or the INSTRUCT
    text in instruct mode — see ``_synthesize_sync`` for why the instruction has to be
    baked into the cache entry rather than passed at call time."""
    sig = hashlib.sha1(f"{ref_clip}\x00{conditioning_text}".encode("utf-8")).hexdigest()[:16]
    cached = _SPK_CACHE.get(sig)
    if cached is not None:
        return cached
    spk_id = f"spk_{sig}"
    cosy.add_zero_shot_spk(conditioning_text, ref_clip, spk_id)
    _SPK_CACHE[sig] = spk_id
    return spk_id


# An instruction is a short natural-language delivery direction ("speak in a calm,
# reassuring tone"). It is NOT the text to be spoken. Keep it short: it occupies the same
# LLM prompt slot the reference transcript normally would, and a long instruction crowds
# out the conditioning that keeps the clone on-voice.
INSTRUCT_MAX_CHARS = 200


def _clean_instruct(instruct_text: Optional[str]) -> str:
    """Normalize a user-supplied instruction: collapse whitespace, cap the length."""
    import re
    t = re.sub(r"\s+", " ", (instruct_text or "").strip())
    return t[:INSTRUCT_MAX_CHARS]


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
    instruct_text: Optional[str] = None,
) -> None:
    # CosyVoice clones from the speaker clip + its transcript. rate/volume/language/style
    # and emo_* are accepted for interface parity but not used by the model.
    _ = (rate, volume, language, style, emo_alpha, emo_vector)
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty text; nothing to synthesize.")

    ref_clip = _resolve_reference(voice, speaker, ref_audio)
    if not ref_clip:
        ref_clip = _env("COSYVOICE3_DEFAULT_REF")
    if not ref_clip:
        raise RuntimeError(
            "CosyVoice needs a reference clip. Select a cloned voice, or set "
            "COSYVOICE3_DEFAULT_REF for the default voice."
        )

    # The transcript of the reference clip. CosyVoice conditions on it; for the default
    # voice fall back to COSYVOICE3_DEFAULT_REF_TEXT.
    p_text = (prompt_text or "").strip() or _env("COSYVOICE3_DEFAULT_REF_TEXT")
    if not p_text:
        raise RuntimeError(
            "CosyVoice needs the reference clip's transcript (prompt_text). Cloned voices "
            "store it as ref_text; for the default voice set COSYVOICE3_DEFAULT_REF_TEXT."
        )
    # Demarcate the prompt boundary so the model doesn't leak prompt audio at the start.
    p_text = _ensure_prompt_terminal(p_text)

    ref_clip, _ref_tmp = _ensure_ref_length(ref_clip)

    speed = max(SPEED_MIN, min(SPEED_MAX, float(
        speed if speed is not None else _env_float("COSYVOICE3_SPEED", SPEED_DEFAULT))))
    max_chars = _env_int("COSYVOICE3_MAX_CHARS", MAX_CHARS_DEFAULT)

    cosy = _load_cosy()

    import torch  # type: ignore
    import torchaudio  # type: ignore

    out_path.parent.mkdir(parents=True, exist_ok=True)
    chunks = _chunk_text(text, max_chars)

    # Instruct mode: a delivery direction steers prosody while the clip still supplies the
    # voice. CosyVoice implements this by swapping the reference transcript out of the LLM
    # prompt for the instruction and dropping llm_prompt_speech_token (so the LLM stops
    # copying the prompt's delivery), while flow_prompt_speech_token + the speaker embedding
    # — what actually carry timbre — are untouched.
    #
    # SUBTLE: upstream's frontend_zero_shot ignores its prompt_text argument entirely when a
    # cached zero_shot_spk_id is supplied (it returns spk2info[spk_id] wholesale), and
    # frontend_instruct2 just delegates to it. So passing the instruction as the call
    # argument would SILENTLY DO NOTHING against our speaker cache. Instead we bake the
    # instruction in as the cache entry's conditioning text — _SPK_CACHE is keyed on it, so
    # each instruction gets its own entry and we keep the one-extraction-per-voice win. We
    # still pass it positionally for correctness if the cache ever misses.
    #
    # <|endofprompt|> goes on HERE, on the string that becomes the cache key — for exactly
    # the reason above. Applying it to the call argument instead would leave the cached
    # conditioning without the marker and CosyVoice3 would assert on every request.
    instruct = _clean_instruct(instruct_text)
    conditioning = _ensure_endofprompt(instruct or p_text)

    try:
        with _INFER_LOCK:
            spk_id = _spk_id_for(cosy, ref_clip, conditioning)
            parts: list["torch.Tensor"] = []
            for chunk in chunks:
                # Both paths run off the cached speaker id (prompt_text/prompt_wav unused).
                # We pass `conditioning`, not the bare instruction: it is ignored on a cache
                # HIT, but on a miss it is what reaches the LLM — and without the
                # <|endofprompt|> marker that path asserts.
                if instruct:
                    stream = cosy.inference_instruct2(
                        chunk, conditioning, "", zero_shot_spk_id=spk_id, stream=False, speed=speed
                    )
                else:
                    stream = cosy.inference_zero_shot(
                        chunk, "", "", zero_shot_spk_id=spk_id, stream=False, speed=speed
                    )
                for out in stream:
                    parts.append(out["tts_speech"])
            if not parts:
                raise RuntimeError("CosyVoice synthesis produced no audio.")
            wav = torch.cat(parts, dim=1)  # each tts_speech is [1, samples]
        torchaudio.save(str(out_path), wav, cosy.sample_rate)
        if not out_path.is_file() or out_path.stat().st_size == 0:
            raise RuntimeError("CosyVoice synthesis produced no audio.")
    finally:
        if _ref_tmp:
            Path(_ref_tmp).unlink(missing_ok=True)


async def synthesize_to_file(
    text: str,
    out_path: Path,
    voice: str,  # reference-clip path for cloning, or a preset name ("default")
    use_gpu: bool,
    rate: str = "+0%",  # accepted for interface parity; ignored by CosyVoice
    volume: str = "+0%",  # accepted for interface parity; ignored by CosyVoice
    speaker: Optional[str] = None,  # optional alias; treated like voice
    language: Optional[str] = None,  # en/zh/ja/ko/yue/auto; used for ref transcription/metadata
    ref_audio: Optional[str] = None,  # explicit reference clip for cloning
    prompt_text: Optional[str] = None,  # transcript of the reference clip (REQUIRED by CosyVoice)
    speed: Optional[float] = None,  # 0.5-2.0 (native CosyVoice mel time-scaling)
    style: Optional[str] = None,  # accepted for parity; not used
    emo_alpha: Optional[float] = None,  # accepted for parity; CosyVoice has no emotion control
    emo_vector: Optional[Sequence[float]] = None,  # accepted for parity; ignored
    instruct_text: Optional[str] = None,  # delivery direction, e.g. "speak in a calm tone"
    **_ignored,
) -> None:
    # Spoken-form normalization happens here, at the engine boundary, so every caller gets
    # it without repeating itself — Narrava Studio narration, SpokenVerse txt-to-audio,
    # Book Alchemy, create-voice previews and the CLI all funnel through this function.
    # It must run BEFORE _chunk_text: its main job is removing abbreviation periods that
    # the chunker would otherwise mistake for sentence ends. Deterministic and idempotent,
    # so the pipeline applying it first (document level) costs nothing here.
    #
    # prompt_text is deliberately NOT normalized — it is the reference clip's verbatim
    # transcript, and CosyVoice conditions on it matching the audio.
    text = normalize_for_tts(text)
    await asyncio.to_thread(
        _synthesize_sync,
        text, out_path, voice, use_gpu, rate, volume, speaker, language, ref_audio,
        prompt_text, speed, style, emo_alpha, emo_vector, instruct_text,
    )


def _discover_voices_sync() -> list[str]:
    return list(_DEFAULT_VOICES)


async def list_voices() -> None:
    for voice in await asyncio.to_thread(_discover_voices_sync):
        print(voice)
    print(
        "\n(CosyVoice clones a voice from a reference clip + its transcript — pass "
        "--ref-audio /path/to/sample.wav and the clip's text. Speed 0.5-2.0 is native.)"
    )
    print("Languages: " + ", ".join(LANGUAGES))
