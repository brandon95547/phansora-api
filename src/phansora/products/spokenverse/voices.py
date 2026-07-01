"""Per-user custom voices for StyleTTS2 voice cloning.

A "voice" is a short reference audio clip the user uploads; StyleTTS2 clones it
at synthesis time (the clip path is passed as the reference). Uploads are trimmed
to at most ``MAX_SECONDS`` and normalized to 24 kHz mono WAV. Clips and a small
per-user JSON manifest live under the runtime data root:

    <runtime_root>/voices/<user_id>/<voice_id>.wav
    <runtime_root>/voices/<user_id>/_pending/<token>.wav   (awaiting approval)
    <runtime_root>/voices/<user_id>/voices.json            (manifest)
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
    p = _pending_dir(user_id) / f"{_safe_token(token)}.wav"
    return p if p.exists() else None


def approve(user_id: str, token: str, name: str) -> Optional[dict]:
    """Promote a pending clip to a saved voice. Returns the record or None."""
    tok = _safe_token(token)
    pending = _pending_dir(user_id) / f"{tok}.wav"
    if not pending.exists():
        return None
    dst = _user_dir(user_id) / f"{tok}.wav"
    shutil.move(str(pending), str(dst))
    record = {
        "id": tok,
        "name": (name or "My Voice").strip()[:80] or "My Voice",
        "created_at": time.time(),
    }
    items = _load_manifest(user_id)
    items.append(record)
    _save_manifest(user_id, items)
    return record


def discard_pending(user_id: str, token: str) -> None:
    p = pending_path(user_id, token)
    if p is not None:
        p.unlink(missing_ok=True)


def list_voices(user_id: str) -> List[dict]:
    # Only return records whose clip still exists on disk.
    return [v for v in _load_manifest(user_id) if voice_path(user_id, v.get("id", "")) is not None]


def voice_path(user_id: str, voice_id: str) -> Optional[Path]:
    p = _user_dir(user_id) / f"{_safe_token(voice_id)}.wav"
    return p if p.exists() else None


def delete_voice(user_id: str, voice_id: str) -> bool:
    vid = _safe_token(voice_id)
    p = _user_dir(user_id) / f"{vid}.wav"
    existed = p.exists()
    p.unlink(missing_ok=True)
    items = [v for v in _load_manifest(user_id) if v.get("id") != vid]
    _save_manifest(user_id, items)
    return existed
