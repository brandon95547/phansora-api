"""Simple file-based JSON cache keyed by a normalized title."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

from ..config import get_settings


# Bump whenever the trace SHAPE changes — new fields, new stages, reworked
# prompts. The cache has no TTL, so without a version a title traced under the
# old pipeline is served unchanged forever: users would keep getting timelines
# with no connections and no claim classes, and there would be no signal that
# anything was stale. Bumping partitions the new traces from the old ones rather
# than deleting anything, so a rollback still finds its own cache intact.
#
# v2: evidence dossiers, claim classes, evaluated connections, read source pages.
SCHEMA_VERSION = "v2"


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
    versioned = f"{SCHEMA_VERSION}:{key}"
    digest = hashlib.sha1(versioned.encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^a-z0-9\-]+", "-", key)[:60].strip("-") or "trace"
    return base / f"{SCHEMA_VERSION}-{safe}-{digest}.json"


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
