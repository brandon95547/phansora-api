"""Simple file-based JSON cache keyed by a normalized title."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

from ..config import get_settings


def normalize_title(title: str) -> str:
    t = title.strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s\-]", "", t)
    return t


def _cache_path(key: str) -> Optional[Path]:
    settings = get_settings()
    if not settings.chrono_cache_dir:
        return None
    base = Path(settings.chrono_cache_dir)
    base.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^a-z0-9\-]+", "-", key)[:60].strip("-") or "trace"
    return base / f"{safe}-{digest}.json"


def get_cached(key: str) -> Optional[dict[str, Any]]:
    path = _cache_path(key)
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_cached(key: str, payload: dict[str, Any]) -> None:
    path = _cache_path(key)
    if path is None:
        return
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def delete_cached(key: str) -> bool:
    """Remove a cached trace by key. Returns True if a file was actually deleted."""
    path = _cache_path(key)
    if path is None or not path.exists():
        return False
    try:
        path.unlink()
        return True
    except Exception:
        return False
