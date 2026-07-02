"""Per-user custom voices for StyleTTS2 voice cloning.

A "voice" is a short reference audio clip the user uploads; StyleTTS2 clones it
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

# Uploads longer than this are trimmed from the END before processing.
MAX_SECONDS = 90
# StyleTTS2 reference clips: 24 kHz mono.
_SAMPLE_RATE = 24000
# Pending clips (uploaded but never approved or discarded) are pruned after this
# long, so abandoned previews don't accumulate on disk forever.
PENDING_TTL_SECONDS = 6 * 3600

# Per-voice synthesis knobs, mirroring the ranges StyleTTS2 supports. These are
# captured at approval and reapplied when the voice is later used for TTS.
DIFFUSION_STEPS_MIN, DIFFUSION_STEPS_MAX, DIFFUSION_STEPS_DEFAULT = 3, 20, 10
EMBEDDING_SCALE_MIN, EMBEDDING_SCALE_MAX, EMBEDDING_SCALE_DEFAULT = 0.5, 3.0, 1.0


def clamp_settings(diffusion_steps=None, embedding_scale=None) -> dict:
    """Coerce/clamp synthesis knobs to supported ranges, filling defaults."""
    try:
        steps = int(diffusion_steps) if diffusion_steps is not None else DIFFUSION_STEPS_DEFAULT
    except (TypeError, ValueError):
        steps = DIFFUSION_STEPS_DEFAULT
    try:
        scale = float(embedding_scale) if embedding_scale is not None else EMBEDDING_SCALE_DEFAULT
    except (TypeError, ValueError):
        scale = EMBEDDING_SCALE_DEFAULT
    steps = max(DIFFUSION_STEPS_MIN, min(DIFFUSION_STEPS_MAX, steps))
    scale = max(EMBEDDING_SCALE_MIN, min(EMBEDDING_SCALE_MAX, round(scale, 2)))
    return {"diffusion_steps": steps, "embedding_scale": scale}


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
    """The pending reference clip (what StyleTTS2 clones from)."""
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


def save_pending_settings(user_id: str, token: str, diffusion_steps=None, embedding_scale=None) -> dict:
    """Record the synthesis knobs used for a pending sample so approval can persist
    them onto the saved voice. Returns the clamped settings."""
    settings = clamp_settings(diffusion_steps, embedding_scale)
    pending_settings_path(user_id, token).write_text(json.dumps(settings), encoding="utf-8")
    return settings


def _read_pending_settings(user_id: str, token: str) -> dict:
    p = pending_settings_path(user_id, token)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return clamp_settings(data.get("diffusion_steps"), data.get("embedding_scale"))
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
        out.append({**v, **clamp_settings(v.get("diffusion_steps"), v.get("embedding_scale"))})
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
