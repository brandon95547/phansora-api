"""Image search across the open web, via DuckDuckGo.

The Media panel's "Search Web Images". Everything in services/media.py is free-to-use
stock; nothing here is. These are ordinary web pictures — most of them belong to someone —
so every result says where it was found and that its rights are unknown, and that rides
onto the asset row and out into the project's credits list like any stock licence.

DuckDuckGo has no image API. This is the same two requests its own results page makes: the
search page, for the `vqd` token that authorises a query, then ``i.js`` for the JSON. The
pictures themselves come from Bing's index.

Why not the ``ddgs`` package, which does exactly this and is already in the venv: it turns
every non-200 into an empty list. DuckDuckGo answers a throttled client with 202 on the
page and 403 on ``i.js``, so under ddgs "you are being rate limited" and "nothing matches"
are the same empty grid. That is the failure that got ddgs taken out of Chrono Origin (see
shared/ai/search.py). Here the two stay apart: a genuine miss is ``[]``, anything else
raises WebSearchUnavailable and the editor says so.
"""
from __future__ import annotations

import re
import threading
import uuid
from typing import List
from urllib.parse import urlsplit

import httpx

from .. import config
from ..models import MediaClip

_UA = {"User-Agent": "NarravaStudio/0.1 (+phansora.com)"}
_PAGE = "https://duckduckgo.com/"
_IMAGES = "https://duckduckgo.com/i.js"
_VQD_RE = re.compile(r"""vqd=["']?([\d-]+)""")

# Every search leaves from the one server address, so one person holding Enter could get
# that address throttled for everybody. Node rate-limits per user; this bounds how many
# are on the wire at once, whoever sent them.
_GATE = threading.Semaphore(2)

# The import route stores jpg/png/gif/webp and nothing else, so a picture in any other
# format would be a tile you can select and then cannot add.
_UNSTORABLE = (".svg", ".avif", ".bmp", ".tif", ".tiff", ".ico", ".heic")

# Recorded as the licence so the credits list says it in so many words. The link beside it
# (license_url) is the page the picture was found on, which is where the rights are.
RIGHTS_UNKNOWN = "Rights unknown — check the source page before publishing"


class WebSearchUnavailable(Exception):
    """DuckDuckGo did not answer the search — throttled, blocked, down or changed shape.

    Deliberately not an empty result: see the module docstring.
    """


def search_web_images(query: str, *, limit: int = 48) -> List[MediaClip]:
    """Up to ``limit`` web images for ``query``. ``[]`` means DuckDuckGo found nothing;
    every other failure raises WebSearchUnavailable with a sentence the editor can show."""
    query = (query or "").strip()
    if not query:
        return []
    timeout = float(config.get_settings().narrava_media_timeout_s)
    with _GATE, httpx.Client(timeout=timeout, headers=_UA) as client:
        try:
            page = client.get(_PAGE, params={"q": query})
            match = _VQD_RE.search(page.text) if page.status_code == 200 else None
            if not match:
                raise WebSearchUnavailable(_refused(page.status_code))
            resp = client.get(
                _IMAGES,
                params={"o": "json", "q": query, "vqd": match.group(1), "l": "us-en", "p": "1"},
                headers={"Referer": _PAGE, "Accept": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise WebSearchUnavailable("Web image search could not reach DuckDuckGo. Try again in a moment.") from exc
        if resp.status_code != 200:
            raise WebSearchUnavailable(_refused(resp.status_code))
        try:
            items = resp.json().get("results")
        except ValueError as exc:
            raise WebSearchUnavailable("DuckDuckGo sent back something that was not search results.") from exc
    if not isinstance(items, list):
        raise WebSearchUnavailable("DuckDuckGo sent back something that was not search results.")
    return _clips(items, query=query, limit=limit)


def _refused(status: int) -> str:
    # 202 and 403 are how DuckDuckGo throttles; anything else is an outage or a change on
    # their side, and waiting a minute is still the right advice for both.
    if status in (202, 403, 429):
        return "DuckDuckGo is limiting searches from our server right now. Wait a minute and try again."
    return f"DuckDuckGo did not answer the search (HTTP {status}). Try again in a moment."


def _clips(items: list, *, query: str, limit: int) -> List[MediaClip]:
    out: List[MediaClip] = []
    seen: set = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        image = str(it.get("image") or "").strip()
        parts = urlsplit(image)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            continue
        if parts.path.lower().endswith(_UNSTORABLE) or image in seen:
            continue
        seen.add(image)
        page_url = str(it.get("url") or "").strip()
        host = _host(page_url) or _host(image)
        out.append(MediaClip(
            id=f"clip_{uuid.uuid4().hex[:8]}",
            segment_id="search",
            type="image",
            url=image,
            thumbnail_url=str(it.get("thumbnail") or "") or None,
            source=host,
            license=RIGHTS_UNKNOWN,
            license_url=page_url or None,
            attribution=f"Found on {host}" if host else "Found on the web",
            title=str(it.get("title") or "").strip() or None,
            query=query,
        ))
        if len(out) >= limit:
            break
    return out


def _host(url: str) -> str:
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host
