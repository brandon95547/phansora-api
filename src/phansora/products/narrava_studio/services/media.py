"""Fair-use media sourcing.

Default provider is Openverse (https://openverse.org) — a search index of openly
and Creative-Commons-licensed media. We request only results licensed for
commercial use and modification, and carry each result's license + attribution
through to the timeline so the finished video can credit sources properly.

Openverse indexes images (and audio), not video, so video requests currently fall
back to imagery. ``search_media`` never raises: on any network/parse failure it
returns a deterministic placeholder so the preliminary timeline still builds
(important for offline/dev and for keeping the UX resilient).
"""
from __future__ import annotations

import uuid
from typing import Dict, List, Optional
from urllib.parse import quote_plus

import httpx

from .. import config
from ..models import MediaClip

_OPENVERSE_IMAGES = "https://api.openverse.org/v1/images/"


def search_media(
    query: str,
    *,
    segment_id: str,
    media_type: str = "image",
    limit: int = 1,
) -> List[MediaClip]:
    """Return up to ``limit`` fair-use clips for ``query`` (best-effort)."""
    query = (query or "").strip()
    settings = config.get_settings()

    if query and settings.narrava_media_provider == "openverse":
        # LLM-written visual queries are often long/specific ("great white shark
        # dorsal fins open blue water"); the CC index matches short queries far
        # better, so retry with progressively shorter variants before giving up.
        for variant in _query_variants(query):
            try:
                clips = _search_openverse(variant, segment_id=segment_id, media_type=media_type, limit=limit)
                if clips:
                    return clips
            except Exception:
                # Never let media sourcing break a build — try the next variant.
                continue

    return [_placeholder(query, segment_id=segment_id, media_type=media_type)]


def _query_variants(query: str) -> List[str]:
    """Full query, then its first 3 and first 2 words (deduped, order preserved)."""
    words = query.split()
    variants = [query]
    if len(words) > 3:
        variants.append(" ".join(words[:3]))
    if len(words) > 2:
        variants.append(" ".join(words[:2]))
    seen: set[str] = set()
    out: List[str] = []
    for v in variants:
        v = v.strip()
        if v and v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out


def _search_openverse(query: str, *, segment_id: str, media_type: str, limit: int) -> List[MediaClip]:
    settings = config.get_settings()
    params = {
        "q": query,
        "page_size": max(1, min(limit, 20)),
        # Only media that may be used commercially and modified — safe for editing
        # into a produced video with attribution.
        "license_type": "commercial,modification",
        "mature": "false",
    }
    resp = httpx.get(
        _OPENVERSE_IMAGES,
        params=params,
        timeout=settings.narrava_media_timeout_s,
        headers={"User-Agent": "NarravaStudio/0.1 (+phansora.com)"},
    )
    resp.raise_for_status()
    results = (resp.json() or {}).get("results") or []

    clips: List[MediaClip] = []
    for r in results[:limit]:
        url = r.get("url") or r.get("thumbnail")
        if not url:
            continue
        clips.append(
            MediaClip(
                id=f"clip_{uuid.uuid4().hex[:8]}",
                segment_id=segment_id,
                type="image",  # Openverse imagery even when video was requested
                url=url,
                thumbnail_url=r.get("thumbnail") or url,
                source=r.get("source") or "openverse",
                license=_license_label(r),
                license_url=r.get("license_url"),
                attribution=_attribution(r),
                title=r.get("title"),
                query=query,
            )
        )
    return clips


def _license_label(r: Dict) -> Optional[str]:
    lic = r.get("license")
    ver = r.get("license_version")
    if not lic:
        return None
    return f"{str(lic).upper()} {ver}".strip()


def _attribution(r: Dict) -> str:
    creator = r.get("creator") or "Unknown"
    title = r.get("title") or "Untitled"
    lic = _license_label(r) or "CC"
    return f'"{title}" by {creator} ({lic}) via Openverse'


def _placeholder(query: str, *, segment_id: str, media_type: str) -> MediaClip:
    label = quote_plus((query or "media")[:40])
    return MediaClip(
        id=f"clip_{uuid.uuid4().hex[:8]}",
        segment_id=segment_id,
        type="video" if media_type == "video" else "image",
        url=f"https://placehold.co/1280x720/1e293b/94a3b8?text={label}",
        thumbnail_url=f"https://placehold.co/320x180/1e293b/94a3b8?text={label}",
        source="placeholder",
        license="Placeholder",
        attribution="Placeholder — replace before publishing.",
        title=query or "Placeholder",
        query=query,
    )
