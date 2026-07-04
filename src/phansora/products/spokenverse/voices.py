"""Per-user custom voices for GPT-SoVITS voice cloning.

A "voice" is a short reference audio clip the user uploads; GPT-SoVITS clones it
at synthesis time (the clip path is passed as the reference). Uploads are trimmed
to at most ``MAX_SECONDS`` and normalized to 24 kHz mono WAV. Clips and a small
per-user JSON manifest live under the runtime data root:

    <runtime_root>/voices/<user_id>/<voice_id>.wav                (reference clip; TTS clones from this)
    <runtime_root>/voices/<user_id>/<voice_id>.sample.wav         (synthesized sample; played back in My Voices)
    <runtime_root>/voices/<user_id>/_pending/<token>.wav          (reference clip awaiting approval)
    <runtime_root>/voices/<user_id>/_pending/<token>.sample.wav   (engine-synthesized preview)
    <runtime_root>/voices/<user_id>/voices.json                   (manifest)

On upload the clip is normalized to a pending reference clip; the caller then runs
it through the engine to synthesize a short sample the user previews. On approval
both are saved: the reference clip (what text-to-speech later clones from) and the
sample (what the user hears when playing the voice back).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import List, Optional

from phansora.shared.paths import runtime_dir

# GPT-SoVITS requires a 3-10 second reference clip (longer errors out). We keep
# the first 9s (safely inside the range) — this clip is what it clones from, and
# it's also what gets auto-transcribed, so the two stay in sync.
MAX_SECONDS = 9
# GPT-SoVITS reference clips: 24 kHz mono.
_SAMPLE_RATE = 24000
# Pending clips (uploaded but never approved or discarded) are pruned after this
# long, so abandoned previews don't accumulate on disk forever.
PENDING_TTL_SECONDS = 6 * 3600

# Per-voice generation knobs, mirroring the options GPT-SoVITS supports. These are
# captured at approval and reapplied when the voice is later used for TTS. The
# reference transcript (``ref_text``) is stored alongside them — GPT-SoVITS clones
# best when it knows what the reference clip says.
from phansora.products.spokenverse.txt_to_voice.adapters.gptsovits_client import (
    LANGUAGES, LANGUAGE_DEFAULT,
    SPEED_MIN, SPEED_MAX, SPEED_DEFAULT,
    TOP_K_MIN, TOP_K_MAX, TOP_K_DEFAULT,
    TOP_P_MIN, TOP_P_MAX, TOP_P_DEFAULT,
    TEMPERATURE_MIN, TEMPERATURE_MAX, TEMPERATURE_DEFAULT,
    REPETITION_PENALTY_MIN, REPETITION_PENALTY_MAX, REPETITION_PENALTY_DEFAULT,
)

# All persisted setting keys (used to backfill/read from the manifest + pending JSON).
SETTING_KEYS = ("language", "speed", "top_k", "top_p", "temperature", "repetition_penalty")

_FLOAT_KNOBS = (
    ("speed", SPEED_MIN, SPEED_MAX, SPEED_DEFAULT),
    ("top_p", TOP_P_MIN, TOP_P_MAX, TOP_P_DEFAULT),
    ("temperature", TEMPERATURE_MIN, TEMPERATURE_MAX, TEMPERATURE_DEFAULT),
    ("repetition_penalty", REPETITION_PENALTY_MIN, REPETITION_PENALTY_MAX, REPETITION_PENALTY_DEFAULT),
)


def clamp_settings(
    language=None, speed=None, top_k=None, top_p=None, temperature=None, repetition_penalty=None,
) -> dict:
    """Coerce/clamp GPT-SoVITS generation knobs to supported ranges, filling defaults."""
    lang = (str(language).strip().lower() if language else "")
    out = {"language": lang if lang in LANGUAGES else LANGUAGE_DEFAULT}
    try:
        tk = int(top_k) if top_k is not None else TOP_K_DEFAULT
    except (TypeError, ValueError):
        tk = TOP_K_DEFAULT
    out["top_k"] = max(TOP_K_MIN, min(TOP_K_MAX, tk))
    given = {"speed": speed, "top_p": top_p, "temperature": temperature, "repetition_penalty": repetition_penalty}
    for name, lo, hi, default in _FLOAT_KNOBS:
        try:
            v = float(given[name]) if given[name] is not None else default
        except (TypeError, ValueError):
            v = default
        out[name] = max(lo, min(hi, round(v, 3)))
    return out


def _safe_id(value: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]", "_", str(value or ""))
    return s[:64] or "anon"


def _safe_token(token: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(token or ""))[:32]


def _user_dir(user_id: str) -> Path:
    d = runtime_dir("voices", _safe_id(user_id))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pending_dir(user_id: str) -> Path:
    d = _user_dir(user_id) / "_pending"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _manifest_path(user_id: str) -> Path:
    return _user_dir(user_id) / "voices.json"


def _load_manifest(user_id: str) -> List[dict]:
    p = _manifest_path(user_id)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []
    return []


def _save_manifest(user_id: str, items: List[dict]) -> None:
    _manifest_path(user_id).write_text(json.dumps(items, indent=2), encoding="utf-8")


def _probe_duration(path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return float((out.stdout or "0").strip() or 0)
    except Exception:
        return 0.0


def _process_to_wav(src: Path, dst: Path, max_seconds: int = MAX_SECONDS) -> None:
    """Trim to at most ``max_seconds`` (keeps the start, drops the tail) and
    convert to 24 kHz mono WAV."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-t", str(max_seconds),
         "-ac", "1", "-ar", str(_SAMPLE_RATE), str(dst)],
        check=True, capture_output=True, timeout=180,
    )


def create_pending(user_id: str, upload_path: Path) -> dict:
    """Process an uploaded clip into a pending voice awaiting approval.

    Returns ``{token, duration_seconds, trimmed}``. ``trimmed`` is True when the
    original ran past MAX_SECONDS and the tail was cut.
    """
    token = uuid.uuid4().hex[:16]
    raw_duration = _probe_duration(upload_path)
    dst = _pending_dir(user_id) / f"{token}.wav"
    _process_to_wav(upload_path, dst)
    return {
        "token": token,
        "duration_seconds": round(_probe_duration(dst), 2),
        "trimmed": raw_duration > MAX_SECONDS,
    }


def pending_path(user_id: str, token: str) -> Optional[Path]:
    """The pending reference clip (what GPT-SoVITS clones from)."""
    p = _pending_dir(user_id) / f"{_safe_token(token)}.wav"
    return p if p.exists() else None


def pending_sample_path(user_id: str, token: str) -> Path:
    """Where the engine-synthesized preview sample for a pending clip is stored.

    Returns the path unconditionally (the caller writes to or checks it). On
    approval this sample is kept as the saved voice's playback clip.
    """
    return _pending_dir(user_id) / f"{_safe_token(token)}.sample.wav"


def pending_settings_path(user_id: str, token: str) -> Path:
    return _pending_dir(user_id) / f"{_safe_token(token)}.json"


def save_pending_settings(user_id: str, token: str, ref_text: str = "", **knobs) -> dict:
    """Record the generation knobs + reference transcript used for a pending sample
    so approval can persist them onto the saved voice. Returns the settings."""
    settings = clamp_settings(**knobs)
    if ref_text:
        settings["ref_text"] = ref_text
    pending_settings_path(user_id, token).write_text(json.dumps(settings), encoding="utf-8")
    return settings


def _read_pending_settings(user_id: str, token: str) -> dict:
    p = pending_settings_path(user_id, token)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                out = clamp_settings(**{k: data.get(k) for k in SETTING_KEYS})
                if data.get("ref_text"):
                    out["ref_text"] = data["ref_text"]
                return out
        except Exception:
            pass
    return clamp_settings()


def approve(user_id: str, token: str, name: str) -> Optional[dict]:
    """Promote a pending reference clip to a saved voice.

    Saves two files: ``<id>.wav`` (the normalized upload, which TTS clones from)
    and ``<id>.sample.wav`` (the synthesized sample, played back in My Voices).
    Returns the record, or None if the pending clip is gone. Raises ``ValueError``
    if ``name`` is blank -- a voice name is required.
    """
    clean_name = (name or "").strip()[:80]
    if not clean_name:
        raise ValueError("A voice name is required.")
    tok = _safe_token(token)
    pending = _pending_dir(user_id) / f"{tok}.wav"
    if not pending.exists():
        return None
    dst = _user_dir(user_id) / f"{tok}.wav"
    shutil.move(str(pending), str(dst))
    # Keep the synthesized sample as the playback clip; the reference clip above is
    # what TTS clones from.
    sample = pending_sample_path(user_id, token)
    if sample.exists():
        shutil.move(str(sample), str(voice_sample_path(user_id, tok)))
    record = {
        "id": tok,
        "name": clean_name,
        "created_at": time.time(),
        # Persist the knobs used for the approved sample so TTS reuses them later.
        **_read_pending_settings(user_id, token),
    }
    pending_settings_path(user_id, token).unlink(missing_ok=True)
    items = _load_manifest(user_id)
    items.append(record)
    _save_manifest(user_id, items)
    return record


def discard_pending(user_id: str, token: str) -> None:
    p = pending_path(user_id, token)
    if p is not None:
        p.unlink(missing_ok=True)
    pending_sample_path(user_id, token).unlink(missing_ok=True)
    pending_settings_path(user_id, token).unlink(missing_ok=True)


def _prune_dir(pending_dir: Path, cutoff: float) -> int:
    """Delete files in ``pending_dir`` last modified before ``cutoff``. Best-effort;
    returns the count removed."""
    removed = 0
    try:
        entries = list(pending_dir.glob("*"))
    except Exception:
        return 0
    for p in entries:
        try:
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
                removed += 1
        except Exception:
            pass
    return removed


def prune_pending(user_id: str, max_age_seconds: int = PENDING_TTL_SECONDS) -> int:
    """Prune one user's stale pending clips (called opportunistically on upload)."""
    return _prune_dir(_pending_dir(user_id), time.time() - max_age_seconds)


def prune_all_pending(max_age_seconds: int = PENDING_TTL_SECONDS) -> int:
    """Prune stale pending clips for every user (called once at startup)."""
    base = runtime_dir("voices")
    if not base.is_dir():
        return 0
    cutoff = time.time() - max_age_seconds
    total = 0
    for user_dir in base.iterdir():
        if user_dir.is_dir():
            total += _prune_dir(user_dir / "_pending", cutoff)
    return total


def list_voices(user_id: str) -> List[dict]:
    # Only return records whose clip still exists on disk. Backfill settings so the
    # client always receives valid knobs, even for voices saved before they existed.
    out = []
    for v in _load_manifest(user_id):
        if voice_path(user_id, v.get("id", "")) is None:
            continue
        out.append({**v, **clamp_settings(**{k: v.get(k) for k in SETTING_KEYS})})
    return out


def voice_path(user_id: str, voice_id: str) -> Optional[Path]:
    """The saved reference clip TTS clones from."""
    p = _user_dir(user_id) / f"{_safe_token(voice_id)}.wav"
    return p if p.exists() else None


def voice_sample_path(user_id: str, voice_id: str) -> Path:
    """The saved playback clip (synthesized sample). Returned unconditionally;
    callers check ``.exists()`` (older voices predate samples)."""
    return _user_dir(user_id) / f"{_safe_token(voice_id)}.sample.wav"


def delete_voice(user_id: str, voice_id: str) -> bool:
    vid = _safe_token(voice_id)
    p = _user_dir(user_id) / f"{vid}.wav"
    existed = p.exists()
    p.unlink(missing_ok=True)
    voice_sample_path(user_id, vid).unlink(missing_ok=True)
    items = [v for v in _load_manifest(user_id) if v.get("id") != vid]
    _save_manifest(user_id, items)
    return existed
