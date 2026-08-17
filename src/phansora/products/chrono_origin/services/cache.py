"""File-based JSON cache for completed traces.

DELIBERATELY SHARED between users. A trace answers an objective question — where a
story first appears — so two people asking it deserve the same answer, and a full run
costs a grounded-search fan-out plus several model calls. Caching per user would
multiply that bill to tell two people the same thing.

What it must NOT do is answer a different question with a stored one. The key used to
be the title alone, while a request also carries `context`, `max_depth`,
`max_sources_per_stage` and `language` — every one of which changes the result. So
"Moses" with context "biblical figure" at depth 6 and "Moses" with context "Egyptian
mythology" at depth 2 collided, and whichever ran first answered both. That is not a
cross-user problem: one person re-tracing with the controls moved got their old answer
back, and nothing on screen said the settings had been ignored.

The key is now the whole request. The FILENAME still leads with the title slug, which
is what lets invalidation work from a title alone (see delete_cached) and what keeps
the directory readable.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
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
# v3: node types, per-claim source ranks, the citation chase.
# v4: strand coverage reaches synthesis, open strands buy their own searches, and
#     a claimed tier can no longer promote a lead. Every one of those changes what
#     a trace CONTAINS rather than what it looks like, so a v3 answer served for a
#     v4 request would be the old research wearing the new pipeline's name.
SCHEMA_VERSION = "v4"


def normalize_title(title: str) -> str:
    t = title.strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s\-]", "", t)
    return t


def request_key(
    title: str,
    *,
    context: Optional[str] = None,
    max_depth: Optional[int] = None,
    max_sources_per_stage: Optional[int] = None,
    language: Optional[str] = None,
) -> str:
    """Everything about a request that changes its answer, as one string.

    Kept explicit rather than hashing the whole model: a field added to TraceRequest
    that does NOT affect the result (a display preference, a client id) should not
    silently split the cache, and one that does should be a deliberate line here.
    """
    parts = [
        normalize_title(title),
        normalize_title(context or ""),
        str(max_depth or ""),
        str(max_sources_per_stage or ""),
        (language or "en").strip().lower(),
    ]
    return "\x1f".join(parts)


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9\-]+", "-", normalize_title(title))[:60].strip("-") or "trace"


def _cache_path(title: str, key: str) -> Optional[Path]:
    settings = get_settings()
    if not settings.chrono_cache_dir:
        return None
    base = Path(settings.chrono_cache_dir)
    base.mkdir(parents=True, exist_ok=True)
    # Both halves matter. SCHEMA_VERSION partitions traces whose SHAPE changed; the
    # key partitions traces whose QUESTION differs. The filename slug is built from the
    # title alone — not from the key — because delete_cached has to find every variant
    # of a title, and a slug carrying the context and depth could not be matched from a
    # title on its own.
    versioned = f"{SCHEMA_VERSION}:{key}"
    digest = hashlib.sha1(versioned.encode("utf-8")).hexdigest()[:16]
    return base / f"{SCHEMA_VERSION}-{_slug(title)}-{digest}.json"


def _expired(path: Path) -> bool:
    ttl_days = get_settings().chrono_cache_ttl_days
    if not ttl_days or ttl_days <= 0:
        return False
    return (time.time() - path.stat().st_mtime) > (ttl_days * 86400)


def get_cached(title: str, key: str) -> Optional[dict[str, Any]]:
    path = _cache_path(title, key)
    if path is None or not path.exists():
        return None
    try:
        if _expired(path):
            # Dropped on read rather than by a sweep: this is the only moment anyone
            # cares, and it keeps the cache with no background job to own.
            path.unlink(missing_ok=True)
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_cached(title: str, key: str, payload: dict[str, Any]) -> None:
    path = _cache_path(title, key)
    if path is None:
        return
    # Written via a temp file and moved into place: a reader must never catch a
    # half-written trace, and a crash mid-write must not leave one behind.
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)


def delete_cached(title: str) -> int:
    """Drop every cached variant of a title. Returns how many files were removed.

    By title, because that is all the caller knows: the Node app invalidates when a user
    deletes a trace, and it holds the title, not the depth and context that produced it.
    Now that one title can have several entries, removing only an exact-key match would
    leave the others servable — so this clears the whole family.
    """
    settings = get_settings()
    if not settings.chrono_cache_dir:
        return 0
    base = Path(settings.chrono_cache_dir)
    if not base.exists():
        return 0
    # Filenames are `{version}-{slug}-{digest}.json`, so the slug is parsed out and
    # compared EXACTLY rather than globbed. A glob of `*-moses-*.json` would also match
    # `v2-moses-parting-the-red-sea-<digest>.json` and delete a different title's trace.
    #
    # Every version, not just the current one: the user asked for that title to be gone.
    want = _slug(title)
    removed = 0
    for path in base.glob("*.json"):
        stem = path.stem
        if "-" not in stem:
            continue
        _version, _, rest = stem.partition("-")
        slug_part, sep, _digest = rest.rpartition("-")
        if not sep or slug_part != want:
            continue
        try:
            path.unlink()
            removed += 1
        except Exception:
            pass
    return removed
