"""Fair-use media sourcing across several stock providers.

Queries multiple providers concurrently and merges the results so the storyboard
sidebar shows a varied grid of images and video. Providers:

  - Pixabay            images + video   (needs PIXABAY_API_KEY)
  - Pexels             images + video   (needs PEXELS_API_KEY)
  - Internet Archive   images + video   (no key)
  - NASA Image/Video   images + video   (no key)
  - Wikimedia Commons  images           (no key)

Each provider function is best-effort: on a missing key, network error, or parse
failure it returns ``[]`` rather than raising, so one slow or unconfigured source
never breaks search. If everything is empty (e.g. offline dev, or no keys set) we
fall back to a deterministic placeholder so the UX stays resilient. Every result
carries source + attribution (+ license where the provider gives one) so the
finished video can credit its sources.
"""
from __future__ import annotations

import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional
from urllib.parse import quote, quote_plus

import httpx

from .. import config
from ..models import MediaClip

_UA = {"User-Agent": "NarravaStudio/0.1 (+phansora.com)"}
_TAG_RE = re.compile(r"<[^>]+>")


def _key(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _strip_html(s: Optional[str]) -> str:
    return _TAG_RE.sub("", s or "").strip()


def _timeout() -> float:
    return float(config.get_settings().narrava_media_timeout_s)


def _page(limit: int) -> int:
    """Clamp a provider's page size to something it will actually accept (every provider's
    own per-page maximum is >= 50)."""
    return max(3, min(int(limit or 1), 50))


def _per_provider(total: int, providers: int) -> int:
    """How many results to ask ONE provider for, given the total we intend to show.

    Each provider gets roughly its share of the total, doubled as headroom so the grid still
    fills when some sources come back thin. Deliberately NOT the full total: asking five
    providers for 48 each fetches ~240 results to display 48, and for NASA/Internet Archive
    (a follow-up request per hit) that difference is a pile of wasted round-trips.
    """
    share = -(-max(1, int(total or 1)) // max(1, providers))  # ceil division
    return _page(min(share * 2, total))


# Providers that need a follow-up request per hit (NASA, Internet Archive) resolve those
# in parallel — serially, a 32-result page would mean 32 round-trips back to back.
_RESOLVE_WORKERS = 12
# Internet Archive's metadata endpoint is slow (~5s/item), so a full page of hits takes
# ~23s to resolve even in parallel. Cap how many we resolve: the other providers fill the
# rest of the grid, and IA still contributes a solid share without stalling the search.
_ARCHIVE_RESOLVE_MAX = 12


def search_media(
    query: str,
    *,
    segment_id: str,
    media_type: str = "image",
    limit: int = 1,
    allow_placeholder: bool = True,
) -> List[MediaClip]:
    """Return up to ``limit`` fair-use clips for ``query`` (best-effort, never raises).

    ``allow_placeholder`` controls the empty case: the timeline builder needs *something*
    per beat, so it keeps the placeholder; interactive search passes False and gets an
    empty list, which the UI reports as "no results" rather than a fake tile.
    """
    query = (query or "").strip()
    settings = config.get_settings()
    if not query or settings.narrava_media_provider == "placeholder":
        return [_placeholder(query, segment_id=segment_id, media_type=media_type)] if allow_placeholder else []

    want_video = media_type == "video"
    providers = _providers_for(want_video)

    # Long user-typed queries return more when trimmed to their first few words, so try
    # the full query first, then a shortened variant, before giving up.
    for variant in _query_variants(query):
        merged = _fan_out(providers, variant, segment_id=segment_id, limit=limit)
        if merged:
            return merged[:limit]

    return [_placeholder(query, segment_id=segment_id, media_type=media_type)] if allow_placeholder else []


# ── provider fan-out + merge ─────────────────────────────────────────────────
Provider = Callable[..., List[MediaClip]]


def _providers_for(want_video: bool) -> List[Provider]:
    """The enabled providers for the requested media type — a provider is enabled only
    when its API key is present (Wikimedia needs none). Order sets round-robin priority."""
    out: List[Provider] = []
    if want_video:
        if _key("PIXABAY_API_KEY"):
            out.append(_pixabay_videos)
        if _key("PEXELS_API_KEY"):
            out.append(_pexels_videos)
        # Keyless archives — great for historical/documentary footage.
        out.append(_nasa_videos)
        out.append(_archive_videos)
    else:
        if _key("PIXABAY_API_KEY"):
            out.append(_pixabay_images)
        if _key("PEXELS_API_KEY"):
            out.append(_pexels_images)
        out.append(_nasa_images)
        out.append(_archive_images)
        out.append(_wikimedia_images)  # keyless — always available
    return out


def _fan_out(providers: List[Provider], query: str, *, segment_id: str, limit: int) -> List[MediaClip]:
    if not providers:
        return []
    # Each provider is asked for its share of the total (not the whole thing) — see
    # _per_provider — while the merge below still fills up to `limit` overall.
    per = _per_provider(limit, len(providers))
    # Providers are blocking HTTP; run them in parallel so total latency ≈ the slowest one.
    with ThreadPoolExecutor(max_workers=len(providers)) as pool:
        futures = [pool.submit(_safe, fn, query, segment_id, per) for fn in providers]
        lists = [f.result() for f in futures]
    return _interleave(lists, limit)


def _safe(fn: Provider, query: str, segment_id: str, limit: int) -> List[MediaClip]:
    try:
        return fn(query, segment_id=segment_id, limit=limit)
    except Exception:  # noqa: BLE001 — one bad provider must not break search
        return []


def _interleave(lists: List[List[MediaClip]], limit: int) -> List[MediaClip]:
    """Round-robin across providers so the grid mixes sources instead of one dominating."""
    out: List[MediaClip] = []
    i = 0
    while len(out) < limit and any(i < len(lst) for lst in lists):
        for lst in lists:
            if i < len(lst):
                out.append(lst[i])
                if len(out) >= limit:
                    break
        i += 1
    return out


def _query_variants(query: str) -> List[str]:
    words = query.split()
    variants = [query]
    if len(words) > 3:
        variants.append(" ".join(words[:3]))
    seen: set = set()
    out: List[str] = []
    for v in variants:
        v = v.strip()
        if v and v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out


def _clip(**kw) -> MediaClip:
    return MediaClip(id=f"clip_{uuid.uuid4().hex[:8]}", **kw)


# ── Pixabay ──────────────────────────────────────────────────────────────────
def _pixabay_images(query: str, *, segment_id: str, limit: int) -> List[MediaClip]:
    resp = httpx.get(
        "https://pixabay.com/api/",
        params={"key": _key("PIXABAY_API_KEY"), "q": query, "image_type": "photo",
                "per_page": _page(limit), "safesearch": "true"},
        timeout=_timeout(), headers=_UA,
    )
    resp.raise_for_status()
    hits = (resp.json() or {}).get("hits") or []
    out: List[MediaClip] = []
    for h in hits[:limit]:
        url = h.get("largeImageURL") or h.get("webformatURL")
        if not url:
            continue
        who = h.get("user") or "Unknown"
        out.append(_clip(
            segment_id=segment_id, type="image", url=url,
            thumbnail_url=h.get("webformatURL") or h.get("previewURL") or url,
            source="Pixabay", license="Pixabay License", license_url=h.get("pageURL"),
            attribution=f"{who} via Pixabay", title=(h.get("tags") or query), query=query,
        ))
    return out


def _pixabay_videos(query: str, *, segment_id: str, limit: int) -> List[MediaClip]:
    resp = httpx.get(
        "https://pixabay.com/api/videos/",
        params={"key": _key("PIXABAY_API_KEY"), "q": query,
                "per_page": _page(limit), "safesearch": "true"},
        timeout=_timeout(), headers=_UA,
    )
    resp.raise_for_status()
    hits = (resp.json() or {}).get("hits") or []
    out: List[MediaClip] = []
    for h in hits[:limit]:
        vids = h.get("videos") or {}
        pick = vids.get("medium") or vids.get("small") or vids.get("large") or vids.get("tiny") or {}
        url = pick.get("url")
        if not url:
            continue
        who = h.get("user") or "Unknown"
        out.append(_clip(
            segment_id=segment_id, type="video", url=url,
            thumbnail_url=pick.get("thumbnail") or h.get("userImageURL") or "",
            source="Pixabay", license="Pixabay License", license_url=h.get("pageURL"),
            attribution=f"{who} via Pixabay", title=(h.get("tags") or query), query=query,
        ))
    return out


# ── Pexels ───────────────────────────────────────────────────────────────────
def _pexels_images(query: str, *, segment_id: str, limit: int) -> List[MediaClip]:
    resp = httpx.get(
        "https://api.pexels.com/v1/search",
        params={"query": query, "per_page": _page(limit)},
        timeout=_timeout(), headers={**_UA, "Authorization": _key("PEXELS_API_KEY")},
    )
    resp.raise_for_status()
    photos = (resp.json() or {}).get("photos") or []
    out: List[MediaClip] = []
    for p in photos[:limit]:
        src = p.get("src") or {}
        url = src.get("large") or src.get("original") or src.get("medium")
        if not url:
            continue
        who = p.get("photographer") or "Unknown"
        out.append(_clip(
            segment_id=segment_id, type="image", url=url,
            thumbnail_url=src.get("medium") or src.get("small") or src.get("tiny") or url,
            source="Pexels", license="Pexels License", license_url=p.get("url"),
            attribution=f"Photo by {who} on Pexels", title=(p.get("alt") or query), query=query,
        ))
    return out


def _pexels_videos(query: str, *, segment_id: str, limit: int) -> List[MediaClip]:
    resp = httpx.get(
        "https://api.pexels.com/videos/search",
        params={"query": query, "per_page": _page(limit)},
        timeout=_timeout(), headers={**_UA, "Authorization": _key("PEXELS_API_KEY")},
    )
    resp.raise_for_status()
    videos = (resp.json() or {}).get("videos") or []
    out: List[MediaClip] = []
    for v in videos[:limit]:
        url = _best_pexels_file(v.get("video_files") or [])
        if not url:
            continue
        who = (v.get("user") or {}).get("name") or "Unknown"
        out.append(_clip(
            segment_id=segment_id, type="video", url=url,
            thumbnail_url=v.get("image") or "",
            source="Pexels", license="Pexels License", license_url=v.get("url"),
            attribution=f"Video by {who} on Pexels", title=query, query=query,
        ))
    return out


def _best_pexels_file(files: List[Dict]) -> Optional[str]:
    """Pick the largest MP4 up to ~1080p so the clip is crisp but not enormous."""
    mp4s = [f for f in files if (f.get("file_type") == "video/mp4") and f.get("link")]
    if not mp4s:
        mp4s = [f for f in files if f.get("link")]
    if not mp4s:
        return None
    usable = [f for f in mp4s if (f.get("width") or 0) <= 1920] or mp4s
    usable.sort(key=lambda f: (f.get("width") or 0), reverse=True)
    return usable[0].get("link")


# ── NASA Image & Video Library (keyless) ─────────────────────────────────────
# images.nasa.gov search returns collection items; each item's `href` is a JSON
# manifest listing the actual asset files, so a playable video needs one extra
# fetch per hit. Public-domain, which makes it ideal for documentary work.
_NASA_SEARCH = "https://images-api.nasa.gov/search"


def _nasa_images(query: str, *, segment_id: str, limit: int) -> List[MediaClip]:
    return _nasa(query, segment_id=segment_id, limit=limit, media_type="image")


def _nasa_videos(query: str, *, segment_id: str, limit: int) -> List[MediaClip]:
    return _nasa(query, segment_id=segment_id, limit=limit, media_type="video")


def _nasa(query: str, *, segment_id: str, limit: int, media_type: str) -> List[MediaClip]:
    resp = httpx.get(
        _NASA_SEARCH,
        params={"q": query, "media_type": media_type, "page_size": _page(limit)},
        timeout=_timeout(), headers=_UA,
    )
    resp.raise_for_status()
    items = ((resp.json() or {}).get("collection") or {}).get("items") or []

    items = items[:limit]
    if not items:
        return []

    # Video items only expose their playable file through a per-item manifest — resolve
    # those in parallel, or a full page of results would mean that many serial round-trips.
    resolved: List[Optional[str]] = [None] * len(items)
    if media_type != "image":
        with ThreadPoolExecutor(max_workers=min(len(items), _RESOLVE_WORKERS)) as pool:
            resolved = list(pool.map(lambda it: _nasa_asset(it.get("href"), (".mp4",)), items))

    out: List[MediaClip] = []
    for it, asset_url in zip(items, resolved):
        data = (it.get("data") or [{}])[0]
        title = data.get("title") or query
        nasa_id = data.get("nasa_id")
        # `links` carries the preview thumbnail without another request.
        thumb = next((l.get("href") for l in (it.get("links") or []) if l.get("href")), "")
        url = thumb if media_type == "image" else asset_url
        if not url:
            continue
        out.append(_clip(
            segment_id=segment_id,
            type="image" if media_type == "image" else "video",
            url=url, thumbnail_url=thumb,
            source="NASA", license="Public Domain (NASA)",
            license_url=f"https://images.nasa.gov/details/{quote(nasa_id)}" if nasa_id else None,
            attribution=f"{data.get('center') or 'NASA'} — NASA Image and Video Library",
            title=title, query=query,
        ))
    return out


def _nasa_asset(manifest_url: Optional[str], exts: tuple) -> Optional[str]:
    """Resolve an item's asset manifest to a playable file URL (best-effort)."""
    if not manifest_url:
        return None
    try:
        r = httpx.get(_safe_url(manifest_url), timeout=_timeout(), headers=_UA)
        r.raise_for_status()
        files = r.json() or []
    except Exception:  # noqa: BLE001
        return None
    picks = [f for f in files if isinstance(f, str) and f.lower().endswith(exts)]
    if not picks:
        return None
    # Prefer a mid-size encoding over the huge "orig" master.
    for tag in ("~medium", "medium", "~mobile", "small"):
        for f in picks:
            if tag in f.lower():
                return _safe_url(f)
    return _safe_url(picks[0])


# Characters that must survive percent-encoding so a URL keeps its structure. Kept as a
# module constant rather than inline: it contains a quote character, and a backslash inside
# an f-string expression is a SyntaxError on Python 3.10 (the runtime prod uses).
_URL_SAFE = "/?&=%#+,;@!$~*'()[]"


def _safe_url(url: str) -> str:
    """NASA asset paths contain raw spaces (e.g. ".../NASA Aims to Land.../collection.json"),
    which HTTP clients reject outright — percent-encode the path while leaving the URL's
    structure (scheme/host/separators) intact. Also upgrades http -> https."""
    url = url.replace("http://", "https://", 1) if url.startswith("http://") else url
    scheme, sep, rest = url.partition("://")
    if not sep:
        return quote(url, safe=":" + _URL_SAFE)
    host, slash, path = rest.partition("/")
    encoded = quote(path, safe=_URL_SAFE)
    return f"{scheme}://{host}{slash}{encoded}"


# ── Internet Archive (keyless) ───────────────────────────────────────────────
# advancedsearch returns item identifiers; the files in an item come from its
# metadata endpoint, so (like NASA) a playable file needs one lookup per hit.
_IA_SEARCH = "https://archive.org/advancedsearch.php"
_IA_META = "https://archive.org/metadata"


def _archive_images(query: str, *, segment_id: str, limit: int) -> List[MediaClip]:
    return _archive(query, segment_id=segment_id, limit=limit, want_video=False)


def _archive_videos(query: str, *, segment_id: str, limit: int) -> List[MediaClip]:
    return _archive(query, segment_id=segment_id, limit=limit, want_video=True)


def _archive(query: str, *, segment_id: str, limit: int, want_video: bool) -> List[MediaClip]:
    mediatype = "movies" if want_video else "image"
    # Match the title/description rather than full text, and exclude the TV News Archive —
    # an untargeted query otherwise surfaces late-night talk-show recordings that merely
    # mention the topic, instead of usable documentary footage.
    q = f'title:({query}) AND mediatype:{mediatype} AND -collection:tvarchive'
    resp = httpx.get(
        _IA_SEARCH,
        params={
            "q": q,
            "fl[]": ["identifier", "title", "creator", "licenseurl"],
            "rows": _page(limit), "page": 1, "output": "json",
        },
        timeout=_timeout(), headers=_UA,
    )
    resp.raise_for_status()
    docs = (((resp.json() or {}).get("response") or {}).get("docs")) or []
    if not docs:
        return []

    exts = (".mp4", ".m4v", ".ogv") if want_video else (".jpg", ".jpeg", ".png")
    # Resolve each item's file list in parallel — one HTTP call per hit otherwise serializes.
    # Bounded by _ARCHIVE_RESOLVE_MAX: this endpoint is slow enough that a full page would
    # stall the whole (parallel) fan-out behind it.
    docs = docs[:min(limit, _ARCHIVE_RESOLVE_MAX)]
    with ThreadPoolExecutor(max_workers=min(len(docs), _RESOLVE_WORKERS)) as pool:
        resolved = list(pool.map(lambda d: _ia_file(d.get("identifier"), exts), docs))

    out: List[MediaClip] = []
    for doc, url in zip(docs, resolved):
        if not url:
            continue
        creator = doc.get("creator") or "Internet Archive"
        if isinstance(creator, list):
            creator = creator[0] if creator else "Internet Archive"
        ident = doc.get("identifier")
        out.append(_clip(
            segment_id=segment_id, type="video" if want_video else "image", url=url,
            thumbnail_url=f"https://archive.org/services/img/{quote(str(ident))}" if ident else "",
            source="Internet Archive", license=doc.get("licenseurl") and "See source" or "Internet Archive",
            license_url=f"https://archive.org/details/{quote(str(ident))}" if ident else None,
            attribution=f"{creator} via Internet Archive",
            title=str(doc.get("title") or query)[:120], query=query,
        ))
    return out


def _ia_file(identifier: Optional[str], exts: tuple) -> Optional[str]:
    """Pick a playable/viewable file from an Internet Archive item (best-effort)."""
    if not identifier:
        return None
    try:
        r = httpx.get(f"{_IA_META}/{quote(str(identifier))}", timeout=_timeout(), headers=_UA)
        r.raise_for_status()
        files = (r.json() or {}).get("files") or []
    except Exception:  # noqa: BLE001
        return None
    names = [f.get("name") for f in files if isinstance(f, dict) and f.get("name")]
    picks = [n for n in names if str(n).lower().endswith(exts)]
    if not picks:
        return None
    # Prefer a derivative (smaller) over the original master when one exists.
    picks.sort(key=lambda n: ("512kb" not in n.lower() and "256kb" not in n.lower(), len(n)))
    return f"https://archive.org/download/{quote(str(identifier))}/{quote(str(picks[0]))}"


# ── Wikimedia Commons (keyless) ──────────────────────────────────────────────
def _wikimedia_images(query: str, *, segment_id: str, limit: int) -> List[MediaClip]:
    resp = httpx.get(
        "https://commons.wikimedia.org/w/api.php",
        params={
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": query, "gsrnamespace": 6, "gsrlimit": _page(limit),
            "prop": "imageinfo", "iiprop": "url|extmetadata|mime", "iiurlwidth": 800,
        },
        timeout=_timeout(), headers=_UA,
    )
    resp.raise_for_status()
    pages = ((resp.json() or {}).get("query") or {}).get("pages") or {}
    out: List[MediaClip] = []
    for page in list(pages.values())[:limit]:
        info = (page.get("imageinfo") or [{}])[0]
        if not str(info.get("mime") or "").startswith("image/"):
            continue
        url = info.get("thumburl") or info.get("url")
        if not url:
            continue
        meta = info.get("extmetadata") or {}
        artist = _strip_html((meta.get("Artist") or {}).get("value")) or "Unknown"
        lic = _strip_html((meta.get("LicenseShortName") or {}).get("value")) or "See source"
        out.append(_clip(
            segment_id=segment_id, type="image",
            url=info.get("url") or url, thumbnail_url=url,
            source="Wikimedia Commons", license=lic, license_url=info.get("descriptionurl"),
            attribution=f"{artist} ({lic}) via Wikimedia Commons",
            title=str(page.get("title") or query).replace("File:", ""), query=query,
        ))
    return out


# ── Placeholder (offline / no results) ───────────────────────────────────────
def _placeholder(query: str, *, segment_id: str, media_type: str) -> MediaClip:
    label = quote_plus((query or "media")[:40])
    return _clip(
        segment_id=segment_id,
        type="video" if media_type == "video" else "image",
        url=f"https://placehold.co/1280x720/1e293b/94a3b8?text={label}",
        thumbnail_url=f"https://placehold.co/320x180/1e293b/94a3b8?text={label}",
        source="placeholder", license="Placeholder",
        attribution="Placeholder — replace before publishing.",
        title=query or "Placeholder", query=query,
    )
