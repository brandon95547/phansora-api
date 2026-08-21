"""Open the page behind a citation and read it.

Search returns a title and a URL. Neither can tell you whether a manuscript is
the original or a ninth-century copy, and neither can be followed backwards to
what a secondary source actually cites — which are the two questions this
product exists to answer. Without this stage, every provenance field in the
dossier is the model's recollection dressed as a finding; the OpenAI path does
not even return snippets, so nothing the pipeline says about a source has ever
been checked against the source.

The budget is spent narrowly and deliberately: a handful of the highest-tier
pages per trace, capped in length, fetched concurrently because the wait is
network time rather than tokens. Anything that fails is reported as unread
rather than quietly skipped — a page we could not open is a gap in the evidence,
and the dossier should say so.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional

from phansora.shared.net.safe_fetch import UnsafeUrlError, fetch_text

logger = logging.getLogger(__name__)

# Enough page for provenance discussion, which is usually near the top, without
# letting one long article crowd out the other sources in the prompt.
DEFAULT_MAX_CHARS = 6000
FETCH_TIMEOUT_S = 15.0
# Pages are text; 3 MB is generous for an article and stops a hostile server
# streaming us out of memory before safe_fetch's own cap matters.
MAX_BYTES = 3 * 1024 * 1024

_SCRIPT_STYLE = re.compile(r"<(script|style|noscript|svg)\b[^>]*>.*?</\1>", re.I | re.S)
_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")
_BLANKS = re.compile(r"\n{3,}")


@dataclass
class PageRead:
    """One attempt to read a source. `text` is empty when the read failed."""

    url: str
    title: str = ""
    text: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())


def _strip_html(html: str, *, max_chars: int) -> str:
    """Last-resort extraction when trafilatura is unavailable or finds nothing."""
    body = _SCRIPT_STYLE.sub(" ", html or "")
    body = _TAGS.sub("\n", body)
    body = (
        body.replace("&nbsp;", " ").replace("&amp;", "&")
        .replace("&lt;", "<").replace("&gt;", ">").replace("&#39;", "'")
        .replace("&quot;", '"')
    )
    body = _WS.sub(" ", body)
    body = "\n".join(line.strip() for line in body.split("\n"))
    body = _BLANKS.sub("\n\n", body).strip()
    return body[:max_chars]


def _extract(html: str, url: str, *, max_chars: int) -> str:
    """Main article text, preferring trafilatura's boilerplate removal.

    trafilatura is already a dependency of Book Alchemy's parsers, so this adds
    no new install. If it is missing or returns nothing usable (some archive and
    museum pages are mostly tables), fall back to tag-stripping rather than
    treating the page as unreadable.
    """
    try:
        import trafilatura  # noqa: PLC0415 - optional, resolved per call site

        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )
        if text and text.strip():
            return text.strip()[:max_chars]
    except ImportError:
        logger.debug("trafilatura unavailable; using tag-stripping fallback")
    except Exception as exc:  # noqa: BLE001 - extraction quality must not break a trace
        logger.debug("trafilatura failed on %s: %s", url, exc)
    return _strip_html(html, max_chars=max_chars)


_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)


def read_page(url: str, *, max_chars: int = DEFAULT_MAX_CHARS) -> PageRead:
    """Fetch and extract one page. Never raises."""
    try:
        html = fetch_text(url, timeout=FETCH_TIMEOUT_S, max_bytes=MAX_BYTES)
    except UnsafeUrlError as exc:
        return PageRead(url=url, error=str(exc))
    except Exception as exc:  # noqa: BLE001 - a dead link is a finding, not a crash
        return PageRead(url=url, error=f"{type(exc).__name__}: {exc}"[:200])

    title = ""
    m = _TITLE.search(html or "")
    if m:
        title = _WS.sub(" ", _TAGS.sub("", m.group(1))).strip()[:200]

    text = _extract(html, url, max_chars=max_chars)
    if not text.strip():
        return PageRead(url=url, title=title, error="No readable text extracted.")
    return PageRead(url=url, title=title, text=text)


# ------------------------------------------------------------ mining leads
# A lead page's value is not its prose — the policy is explicit that a wiki must
# never be the evidentiary basis of anything — it is the list of real sources at
# the bottom of it. Wikipedia footnotes point overwhelmingly at JSTOR, DOIs,
# archive.org, Perseus and university presses, which is exactly the material a
# trace is supposed to rest on, and until now the pipeline threw the page away
# without ever looking. So: open the page, take ONLY the reference block, and
# never let a word of its body text reach the model.
_REF_HEADING = re.compile(
    r"<h[23][^>]*>\s*(?:<[^>]+>\s*)*(references|notes|footnotes|bibliography|sources|works cited|citations)\b",
    re.I,
)
_REF_LIST_ATTR = re.compile(
    r"<(?:ol|ul|div)[^>]*(?:class|id)=\"[^\"]*(?:reference|reflist|footnote|bibliograph|citation)[^\"]*\"[^>]*>",
    re.I,
)
_ANCHOR = re.compile(r"<a\b[^>]*href=\"(https?://[^\"#]+)\"[^>]*>(.*?)</a>", re.I | re.S)
# Wikipedia's own furniture, plus the licence and tooling links every wiki page
# carries. Harvesting these would just feed the pipeline more of the same tier.
_REF_NOISE = (
    "wikipedia.org", "wikimedia.org", "wikidata.org", "wiktionary.org",
    "creativecommons.org", "mediawiki.org", "wikisource.org", "archive.today",
    "web.archive.org/save", "/wiki/", "action=edit", "javascript:",
)
MAX_REFERENCES = 40


def parse_references(html: str) -> List[Dict[str, str]]:
    """Outbound links from a page's reference/footnote block, and nothing else.

    Returns ``[{"url":…, "text":…}]``. The refusal to return body text is the
    property that makes this policy-compliant rather than a loophole, and it is
    asserted in the tests: mining a lead page must not become a way of reading one.
    """
    if not html:
        return []

    # Start at whichever comes first: a heading that says "References", or a list
    # whose class says so. Anything before that point is the page itself.
    starts = [m.start() for m in (_REF_HEADING.search(html), _REF_LIST_ATTR.search(html)) if m]
    if not starts:
        return []
    block = html[min(starts):]

    out: List[Dict[str, str]] = []
    seen: set = set()
    for m in _ANCHOR.finditer(block):
        url = m.group(1).strip()
        low = url.lower()
        if any(n in low for n in _REF_NOISE) or url in seen:
            continue
        seen.add(url)
        text = _WS.sub(" ", _TAGS.sub(" ", m.group(2) or "")).strip()[:200]
        out.append({"url": url, "text": text})
        if len(out) >= MAX_REFERENCES:
            break
    return out



def read_pages(
    urls: List[str],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_workers: int = 4,
) -> List[PageRead]:
    """Read several pages concurrently, preserving the order they were requested in.

    Concurrency here is free in token terms and saves most of the wall clock: a
    15-second worst case repeated in sequence is a minute of a user's trace spent
    waiting on other people's servers. Callers that fetch a batch should pass
    `max_workers=len(urls)` — the default is a floor, not a recommendation.
    """
    targets = [u for u in dict.fromkeys(urls or []) if u]
    if not targets:
        return []
    workers = max(1, min(max_workers, len(targets)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda u: read_page(u, max_chars=max_chars), targets))


def read_best(
    candidates: List[str],
    *,
    want: int,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_failures_reported: int = 2,
) -> List[PageRead]:
    """Read down a ranked candidate list until `want` pages actually come back.

    Major repositories are the sources most worth reading and the most likely to
    refuse: the British Museum returns 403 to any server-side fetch and the Met
    rate-limits, regardless of user agent. Asking for exactly `want` pages means
    one blocked museum silently costs a whole read slot and the trace falls back
    to summaries without anyone noticing.

    So over-fetch and keep the first `want` successes. Fetching is concurrent and
    costs no tokens; only successful reads reach a prompt. A couple of failures
    are kept deliberately, because "the British Museum record could not be
    opened" is a real gap in the evidence and the synthesis stage should see it
    rather than infer silence.
    """
    if want <= 0 or not candidates:
        return []
    # One worker per candidate. The caller over-fetches (2x `want`), so this is
    # typically twelve URLs against a pool that defaulted to four — three waves of a
    # 15-second timeout, up to 45s of a user's trace spent waiting on other people's
    # servers in sequence. Fetching costs no tokens, so width here is free.
    reads = read_pages(candidates, max_chars=max_chars, max_workers=len(candidates))
    good = [r for r in reads if r.ok][:want]
    bad = [r for r in reads if not r.ok][:max_failures_reported]
    return good + bad


def format_reads_block(reads: List[PageRead], *, per_source_chars: Optional[int] = None) -> str:
    """Render read pages for a prompt, keeping failures visible.

    Failed reads are listed too. "We could not open the British Museum record"
    is information the synthesis stage should have — it is the difference between
    an unverified claim and a verified absence.
    """
    if not reads:
        return "(no source pages were read)"
    parts: List[str] = []
    for r in reads:
        if r.ok:
            body = r.text if per_source_chars is None else r.text[:per_source_chars]
            parts.append(f"--- SOURCE PAGE: {r.title or r.url}\nURL: {r.url}\n{body}")
        else:
            parts.append(f"--- COULD NOT READ: {r.url}\nReason: {r.error or 'unknown'}")
    return "\n\n".join(parts)
