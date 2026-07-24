"""Per-user custom voices for CosyVoice2 voice cloning.

A "voice" is a short reference audio clip the user uploads; CosyVoice2 clones it
at synthesis time (the clip path + its transcript ``ref_text`` are passed as the
reference — CosyVoice conditions on the transcript). Uploads are trimmed
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

# CosyVoice2 works best with a 3-10 second reference clip (longer errors out). We keep
# the first 9s (safely inside the range) — this clip is what it clones from, and
# it's also what gets auto-transcribed, so the two stay in sync.
MAX_SECONDS = 9
# We never end the reference on a hard mid-word cut: that leaves the prompt audio with no
# "the speaker stopped" boundary (CosyVoice2 then leaks ~1s of prompt-like audio at the
# start of every clone) and yields a half-sentence transcript with no closing punctuation.
# Instead we cut at the last natural pause within the window and pad a little trailing
# silence, so the clip ends cleanly after a whole word. Only accept a pause once we've kept
# at least MIN_SECONDS of reference (a too-early pause would starve the clone of audio); if
# there's no usable pause, fall back to the hard cut but still pad trailing silence.
MIN_SECONDS = 4.0
TRAILING_SILENCE_SECONDS = 0.5
# silencedetect thresholds: treat < -35 dB for >= 0.3s as a pause (inter-word/sentence gap).
_SILENCE_NOISE_DB = "-35dB"
_SILENCE_MIN_DUR = 0.3
# CosyVoice2 reference clips: 24 kHz mono.
_SAMPLE_RATE = 24000
# Pending clips (uploaded but never approved or discarded) are pruned after this
# long, so abandoned previews don't accumulate on disk forever.
PENDING_TTL_SECONDS = 6 * 3600

# Per-voice generation knobs, mirroring the options CosyVoice2 supports. These are
# captured at approval and reapplied when the voice is later used for TTS. CosyVoice2
# clones from the clip PLUS its transcript, so the reference transcript (``ref_text``)
# stored on the voice record is required at synthesis time (passed as prompt_text).
from phansora.products.spokenverse.txt_to_voice.adapters.cosyvoice2_client import (
    LANGUAGES, LANGUAGE_DEFAULT,
    SPEED_MIN, SPEED_MAX, SPEED_DEFAULT,
    INSTRUCT_MAX_CHARS,
)

# All persisted setting keys (used to backfill/read from the manifest + pending JSON).
SETTING_KEYS = ("language", "speed", "instruct_text")

# Reserved store of app-wide DEFAULT voices, shown to every user in addition to their
# own saved voices. A regular user_id can never resolve to this id (the server's
# _safe_user_id strips leading/trailing "._-"), so the defaults are effectively
# read-only through the per-user API and are managed out-of-band (admin/filesystem).
DEFAULTS_ID = "_defaults"


def clamp_settings(language=None, speed=None, instruct_text=None) -> dict:
    """Coerce/clamp CosyVoice2 generation knobs to supported ranges, filling defaults."""
    lang = (str(language).strip().lower() if language else "")
    out = {"language": lang if lang in LANGUAGES else LANGUAGE_DEFAULT}
    try:
        sp = float(speed) if speed is not None else SPEED_DEFAULT
    except (TypeError, ValueError):
        sp = SPEED_DEFAULT
    out["speed"] = max(SPEED_MIN, min(SPEED_MAX, round(sp, 3)))
    # Delivery direction saved with the voice; "" means plain cloning. Whitespace is
    # collapsed and the length capped to match what the engine will accept.
    instruct = re.sub(r"\s+", " ", str(instruct_text or "").strip())[:INSTRUCT_MAX_CHARS]
    out["instruct_text"] = instruct
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


def _last_pause_before(path: Path, cap: float) -> Optional[float]:
    """Return the end-of-speech time of the last natural pause at or before ``cap``
    seconds (and at/after ``MIN_SECONDS``), or None if there's no usable pause.

    Uses ffmpeg ``silencedetect``; each detected silence's ``silence_start`` marks where
    the preceding word ended, which is exactly where we want to cut so the clip finishes
    on a whole word rather than mid-syllable."""
    proc = subprocess.run(
        ["ffmpeg", "-i", str(path), "-af",
         f"silencedetect=noise={_SILENCE_NOISE_DB}:d={_SILENCE_MIN_DUR}", "-f", "null", "-"],
        capture_output=True, text=True, timeout=60,
    )
    best: Optional[float] = None
    for m in re.finditer(r"silence_start:\s*([0-9.]+)", proc.stderr or ""):
        try:
            t = float(m.group(1))
        except ValueError:
            continue
        if MIN_SECONDS <= t <= cap:
            best = t  # keep the latest qualifying pause
    return best


def _process_to_wav(src: Path, dst: Path, max_seconds: int = MAX_SECONDS) -> None:
    """Normalize an uploaded clip to a 24 kHz mono WAV reference, ending it on a clean
    boundary: cut at the last natural pause within ``max_seconds`` (falling back to a hard
    ``max_seconds`` cut) and pad ``TRAILING_SILENCE_SECONDS`` of silence, so the clip
    finishes after a whole word with a clear "speaker stopped" gap."""
    # 1. Normalize + cap to the working window (mono 24 kHz, at most max_seconds).
    tmp = dst.with_suffix(".norm.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-t", str(max_seconds),
         "-ac", "1", "-ar", str(_SAMPLE_RATE), str(tmp)],
        check=True, capture_output=True, timeout=180,
    )
    try:
        # 2. End at the last natural pause (whole word) if we have one; else the hard cap.
        end = _last_pause_before(tmp, float(max_seconds)) or float(max_seconds)
        # 3. Trim to that boundary and append trailing silence.
        af = (f"atrim=end={end:.3f},asetpts=PTS-STARTPTS,"
              f"apad=pad_dur={TRAILING_SILENCE_SECONDS}")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp), "-af", af,
             "-ac", "1", "-ar", str(_SAMPLE_RATE), str(dst)],
            check=True, capture_output=True, timeout=180,
        )
    finally:
        tmp.unlink(missing_ok=True)


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
    """The pending reference clip (what CosyVoice2 clones from)."""
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
    if ``name`` is blank or already taken by another of this user's voices -- names
    must be unique (case-insensitively).
    """
    clean_name = (name or "").strip()[:80]
    if not clean_name:
        raise ValueError("A voice name is required.")
    tok = _safe_token(token)
    pending = _pending_dir(user_id) / f"{tok}.wav"
    if not pending.exists():
        return None
    # Names must be unique per user (case-insensitive). Check before moving any
    # files so a rejected approval leaves the pending clip intact for a retry.
    items = _load_manifest(user_id)
    if any((v.get("name") or "").strip().casefold() == clean_name.casefold() for v in items):
        raise ValueError(f'You already have a voice named "{clean_name}". Please choose another name.')
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
    items.append(record)  # already loaded above for the duplicate-name check
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


def _own_voice_path(user_id: str, voice_id: str) -> Optional[Path]:
    """Reference clip in this user's own dir only (no fallback to defaults)."""
    p = _user_dir(user_id) / f"{_safe_token(voice_id)}.wav"
    return p if p.exists() else None


def _own_sample_path(user_id: str, voice_id: str) -> Optional[Path]:
    p = _user_dir(user_id) / f"{_safe_token(voice_id)}.sample.wav"
    return p if p.exists() else None


def _list_own(user_id: str) -> List[dict]:
    # Only return records whose clip still exists on disk. Backfill settings so the
    # client always receives valid knobs, even for voices saved before they existed.
    out = []
    for v in _load_manifest(user_id):
        if _own_voice_path(user_id, v.get("id", "")) is None:
            continue
        out.append({**v, **clamp_settings(**{k: v.get(k) for k in SETTING_KEYS})})
    return out


def list_default_voices() -> List[dict]:
    """App-wide default voices shown to every user, sorted alphabetically by name
    (so they appear in order everywhere). Flagged ``default: True`` so the client can
    present them distinctly (and keep them out of the user's "My Voices")."""
    items = [{**v, "default": True} for v in _list_own(DEFAULTS_ID)]
    items.sort(key=lambda v: (v.get("name") or "").casefold())
    return items


def list_voices(user_id: str) -> List[dict]:
    """The voices a user picks from: the shared app defaults first, then this user's
    own saved voices (own entries that duplicate a default id are dropped)."""
    if str(user_id) == DEFAULTS_ID:
        return list_default_voices()
    defaults = list_default_voices()
    seen = {v["id"] for v in defaults}
    own = [v for v in _list_own(user_id) if v.get("id") not in seen]
    return defaults + own


def voice_path(user_id: str, voice_id: str) -> Optional[Path]:
    """The reference clip TTS clones from — the user's own, else a shared default."""
    return _own_voice_path(user_id, voice_id) or _own_voice_path(DEFAULTS_ID, voice_id)


def voice_sample_path(user_id: str, voice_id: str) -> Path:
    """This user's own sample path, returned unconditionally (approve() writes here).
    For playback that should also fall back to defaults, use resolve_sample_path()."""
    return _user_dir(user_id) / f"{_safe_token(voice_id)}.sample.wav"


def resolve_sample_path(user_id: str, voice_id: str) -> Optional[Path]:
    """Existing playback sample — the user's own, else a shared default's, else None."""
    return _own_sample_path(user_id, voice_id) or _own_sample_path(DEFAULTS_ID, voice_id)


def delete_voice(user_id: str, voice_id: str) -> bool:
    vid = _safe_token(voice_id)
    p = _user_dir(user_id) / f"{vid}.wav"
    existed = p.exists()
    p.unlink(missing_ok=True)
    voice_sample_path(user_id, vid).unlink(missing_ok=True)
    items = [v for v in _load_manifest(user_id) if v.get("id") != vid]
    _save_manifest(user_id, items)
    return existed


def rename_voice(user_id: str, voice_id: str, name: str) -> Optional[dict]:
    """Rename a saved voice. Returns the updated record (with clamped settings), or
    None if no such voice. Raises ``ValueError`` if the name is blank or already
    used by another of this user's voices (case-insensitive) — same rule as approve.
    """
    clean_name = (name or "").strip()[:80]
    if not clean_name:
        raise ValueError("A voice name is required.")
    vid = _safe_token(voice_id)
    items = _load_manifest(user_id)
    target = next((v for v in items if v.get("id") == vid), None)
    if target is None:
        return None
    if any(
        v.get("id") != vid and (v.get("name") or "").strip().casefold() == clean_name.casefold()
        for v in items
    ):
        raise ValueError(f'You already have a voice named "{clean_name}". Please choose another name.')
    target["name"] = clean_name
    _save_manifest(user_id, items)
    return {**target, **clamp_settings(**{k: target.get(k) for k in SETTING_KEYS})}
