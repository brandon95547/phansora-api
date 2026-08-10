"""Source-format parsers for Book Alchemy.

Every parser returns a :class:`ParsedDoc` — a uniform shape of normalized text
blocks, each carrying best-effort provenance (chapter / section / page range).
This uniform shape is what makes the rest of the pipeline format-agnostic and
lets new input types (transcripts, doc sets, etc.) slot in later.

Heavy / optional dependencies are imported lazily inside each parser so that a
missing optional package (e.g. MOBI support) never breaks importing this module.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Block:
    text: str
    chapter: Optional[str] = None
    section: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    kind: str = "prose"          # "prose" | "apparatus" — see classify_block


@dataclass
class ParsedDoc:
    title: str
    blocks: list[Block] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.text.strip())


class UnsupportedSourceError(Exception):
    """Raised when a source can't be parsed (e.g. MOBI without the optional lib)."""


class ScannedPdfError(UnsupportedSourceError):
    """The PDF has little/no extractable text and needs OCR to recover content.

    Distinct from UnsupportedSourceError so the pipeline can fall back to the
    OCR path instead of failing the job."""


# ----------------------------------------------------------------- apparatus
# Front and back matter a course must never narrate: tables of contents, indexes,
# page numbers, running heads, copyright pages. It is NAVIGATION, not material —
# a lecturer does not read it aloud. Before this filter existed, a Bible course
# opened by reciting how many chapters each book has, because every line in the
# file was treated as something to teach.
#
# Deliberately deterministic and deliberately conservative: everything caught
# here is SHAPED like a reference line. Prose that merely talks about structure
# reads as prose and survives; catching that is the job of the `teachable`
# judgment in the analyze phase, which can actually read the sentence. And
# build_chunks discards the whole verdict if it would drop more than a quarter of
# a source (APPARATUS_MAX_SHARE), so a mis-tuned pattern here cannot silently gut
# a book — it can only fail to help.

_DOT_LEADER = re.compile(r"\.\s*\.\s*\.|…")
_ENDS_IN_PAGE = re.compile(r"\S\s+\d{1,4}$")
_BARE_PAGE = re.compile(r"^(?:page\s+)?\d{1,4}$", re.I)
_ROMAN_PAGE = re.compile(r"^[ivxlcdm]{1,7}$", re.I)
# "Abraham, 14, 22, 31" / "covenant, 88-90, 145" — two or more page refs trailing.
_INDEX_ENTRY = re.compile(
    r",\s*\d{1,4}(?:\s*[-–]\s*\d{1,4})?"
    r"(?:\s*,\s*\d{1,4}(?:\s*[-–]\s*\d{1,4})?)+\s*$"
)
_COPYRIGHT = re.compile(
    r"all rights reserved|isbn\b|library of congress|printed in the "
    r"united states|no part of this (?:book|publication)|"
    r"cataloging-in-publication",
    re.I,
)
_SENTENCE_END = re.compile(r"[.!?][\"')\]]?$")

_REFERENCE_LINE_MAX_CHARS = 90    # longer than this and it is a sentence
_REFERENCE_LINE_MAX_WORDS = 12

APPARATUS_LINE_SHARE = 0.6        # of the lines in a multi-line block
APPARATUS_MIN_LINES = 3           # below this, one stray match proves nothing

RUNNING_HEAD_MIN_PAGES = 5
RUNNING_HEAD_SHARE = 0.4          # of pages carrying the same head


def _is_reference_line(line: str) -> bool:
    """Does this single line point at content rather than carry it?"""
    s = line.strip()
    if not s:
        return False
    if _BARE_PAGE.match(s) or _ROMAN_PAGE.match(s):
        return True
    if len(s) > _REFERENCE_LINE_MAX_CHARS:
        return False
    if _DOT_LEADER.search(s) and re.search(r"\d\s*$", s):
        return True
    if _INDEX_ENTRY.search(s):
        return True
    # "Genesis            12" — a label and a page number with no sentence between
    # them. The sentence-end and word-count guards keep ordinary prose that
    # happens to close on a number ("...founded in 1948.") out of this.
    return (
        bool(_ENDS_IN_PAGE.search(s))
        and not _SENTENCE_END.search(s)
        and len(s.split()) <= _REFERENCE_LINE_MAX_WORDS
    )


def classify_block(text: str) -> str:
    """``"apparatus"`` if this block navigates the book, ``"prose"`` otherwise."""
    s = (text or "").strip()
    if not s:
        return "prose"
    if _COPYRIGHT.search(s) and len(s) < 1200:
        return "apparatus"

    lines = [ln for ln in s.split("\n") if ln.strip()]
    if len(lines) == 1:
        return "apparatus" if _is_reference_line(lines[0]) else "prose"
    if len(lines) < APPARATUS_MIN_LINES:
        return "prose"

    hits = sum(1 for ln in lines if _is_reference_line(ln))
    return "apparatus" if hits / len(lines) >= APPARATUS_LINE_SHARE else "prose"


def _head_key(line: str) -> str:
    """A running head with its page number generalized away.

    "GENESIS 14" and "GENESIS 15" are the same head, so digits collapse to a
    placeholder before the lines are counted.
    """
    return re.sub(r"\d+", "#", line).strip()


def _running_heads(pages: list[str]) -> set[str]:
    """Keys of the lines repeated at the top or bottom of most pages.

    PyMuPDF's ``get_text("text")`` carries no coordinates, so position is
    approximated by "first or last line of the page" — which is where a running
    head lands in practice. Left in, these are narrated once per page.
    """
    if len(pages) < RUNNING_HEAD_MIN_PAGES:
        return set()

    counts: dict[str, int] = {}
    for text in pages:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if not lines:
            continue
        # A set: on a one-line page the first and last line are the same line.
        for candidate in {lines[0], lines[-1]}:
            if 0 < len(candidate) <= _REFERENCE_LINE_MAX_CHARS:
                key = _head_key(candidate)
                counts[key] = counts.get(key, 0) + 1

    threshold = max(RUNNING_HEAD_MIN_PAGES, int(len(pages) * RUNNING_HEAD_SHARE))
    return {key for key, n in counts.items() if n >= threshold}


def _strip_running_heads(text: str, heads: set[str]) -> str:
    if not heads:
        return text
    kept = [ln for ln in text.split("\n") if _head_key(ln.strip()) not in heads]
    return "\n".join(kept)


# ----------------------------------------------------------------- dispatch
def parse_source(
    *,
    source_format: str,
    path: Optional[str] = None,
    url: Optional[str] = None,
    text: Optional[str] = None,
    title_hint: Optional[str] = None,
) -> ParsedDoc:
    fmt = (source_format or "").lower().lstrip(".")
    title = title_hint or (Path(path).stem if path else (url or "Untitled"))

    if fmt in ("txt", "text"):
        return _parse_plain(text if text is not None else _read(path), title)
    if fmt in ("md", "markdown"):
        return _parse_markdown(text if text is not None else _read(path), title)
    if fmt in ("html", "htm"):
        return _parse_html(text if text is not None else _read(path), title)
    if fmt == "url":
        return _parse_url(url or "", title)
    if fmt == "pdf":
        return _parse_pdf(path, title)
    if fmt == "docx":
        return _parse_docx(path, title)
    if fmt == "epub":
        return _parse_epub(path, title)
    if fmt in ("mobi", "azw", "azw3"):
        return _parse_mobi(path, title)
    raise UnsupportedSourceError(f"Unsupported source format: {source_format!r}")


# ----------------------------------------------------------------- plain / md
def _parse_plain(raw: str, title: str) -> ParsedDoc:
    raw = _normalize_ws(raw or "")
    paras = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]
    blocks = [Block(text=p, kind=classify_block(p)) for p in paras]
    return ParsedDoc(title=title, blocks=blocks or [Block(text=raw)])


def _parse_markdown(raw: str, title: str) -> ParsedDoc:
    # Track the current heading as the section/chapter for provenance, then
    # strip markdown to readable prose.
    blocks: list[Block] = []
    current_heading: Optional[str] = None
    buf: list[str] = []

    def flush() -> None:
        if buf:
            blocks.append(Block(text="\n\n".join(buf).strip(), chapter=current_heading))
            buf.clear()

    for line in (raw or "").splitlines():
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush()
            current_heading = _strip_md(m.group(2).strip())
            continue
        if line.strip():
            buf.append(line.rstrip())
        else:
            flush()
    flush()
    cleaned = [
        Block(text=stripped, chapter=b.chapter, kind=classify_block(stripped))
        for b in blocks
        if (stripped := _strip_md(b.text)).strip()
    ]
    return ParsedDoc(title=title, blocks=cleaned or [Block(text=_strip_md(raw))])


# ----------------------------------------------------------------- html / url
def _parse_html(raw: str, title: str) -> ParsedDoc:
    from bs4 import BeautifulSoup  # lazy

    soup = BeautifulSoup(raw or "", "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    page_title = (soup.title.get_text(strip=True) if soup.title else "") or title

    blocks: list[Block] = []
    current_heading: Optional[str] = None
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote"]):
        txt = _normalize_ws(el.get_text(" ", strip=True))
        if not txt:
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            current_heading = txt
        else:
            blocks.append(Block(text=txt, chapter=current_heading, kind=classify_block(txt)))
    if not blocks:
        blocks = _parse_plain(soup.get_text(" ", strip=True), page_title).blocks
    return ParsedDoc(title=page_title, blocks=blocks)


def _parse_url(url: str, title: str) -> ParsedDoc:
    if not url:
        raise UnsupportedSourceError("No URL provided.")
    # Prefer trafilatura's main-content extraction; fall back to raw HTML parse.
    html = None
    try:
        import trafilatura  # lazy

        html = trafilatura.fetch_url(url)
        if html:
            extracted = trafilatura.extract(
                html, include_comments=False, include_tables=False, favor_recall=True
            )
            if extracted and extracted.strip():
                doc = _parse_plain(extracted, title)
                doc.title = title or url
                return doc
    except Exception:
        pass

    if html is None:
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 BookAlchemy"})
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            html = resp.read().decode("utf-8", errors="ignore")
    return _parse_html(html, title or url)


# ----------------------------------------------------------------- pdf
def _parse_pdf(path: Optional[str], title: str) -> ParsedDoc:
    import fitz  # PyMuPDF, lazy

    if not path:
        raise UnsupportedSourceError("No PDF path provided.")
    # Read every page first: running heads can only be identified by comparing
    # pages against each other, so blocks cannot be built in the same pass.
    pages: list[str] = []
    with fitz.open(path) as pdf:
        page_count = pdf.page_count
        for pno in range(page_count):
            pages.append(_normalize_ws(pdf.load_page(pno).get_text("text") or ""))

    if page_count == 0:
        raise UnsupportedSourceError("PDF has no pages.")

    total_chars = sum(len(t) for t in pages)
    text_pages = sum(1 for t in pages if len(t) >= 20)   # pages with real text

    heads = _running_heads(pages)
    blocks: list[Block] = []
    for pno, txt in enumerate(pages):
        for para in re.split(r"\n{2,}", _strip_running_heads(txt, heads)):
            para = para.strip()
            if para:
                blocks.append(Block(
                    text=para, page_start=pno + 1, page_end=pno + 1,
                    kind=classify_block(para),
                ))

    # Decide text-based vs scanned/image-based. A normal digital PDF has text on
    # (nearly) every page; a scanned book extracts ~nothing. Only when the doc is
    # predominantly image-based do we hand off to OCR — text PDFs are used as-is.
    text_ratio = text_pages / page_count
    if text_pages == 0 or text_ratio < 0.3 or total_chars < 100:
        raise ScannedPdfError(
            "PDF appears to be scanned/image-based; OCR required to extract content."
        )
    return ParsedDoc(title=title, blocks=blocks)


# ----------------------------------------------------------------- docx
def _parse_docx(path: Optional[str], title: str) -> ParsedDoc:
    import docx  # python-docx, lazy

    if not path:
        raise UnsupportedSourceError("No DOCX path provided.")
    document = docx.Document(path)
    blocks: list[Block] = []
    current_heading: Optional[str] = None
    for para in document.paragraphs:
        txt = _normalize_ws(para.text or "")
        if not txt:
            continue
        style = (para.style.name or "").lower() if para.style else ""
        if "heading" in style or "title" in style:
            current_heading = txt
        else:
            blocks.append(Block(text=txt, chapter=current_heading, kind=classify_block(txt)))
    return ParsedDoc(title=title, blocks=blocks or [Block(text="")])


# ----------------------------------------------------------------- epub
def _parse_epub(path: Optional[str], title: str) -> ParsedDoc:
    import ebooklib  # lazy
    from ebooklib import epub
    from bs4 import BeautifulSoup

    if not path:
        raise UnsupportedSourceError("No EPUB path provided.")
    book = epub.read_epub(path)
    meta_title = title
    try:
        t = book.get_metadata("DC", "title")
        if t and t[0]:
            meta_title = t[0][0]
    except Exception:
        pass

    blocks: list[Block] = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        # An EPUB3 navigation document IS an ITEM_DOCUMENT, so the table of
        # contents arrives here looking exactly like a chapter. Narrating it is
        # what produced "the book of Job has 42 chapters" courses.
        if _is_epub_nav(item, soup):
            continue
        chapter = None
        h = soup.find(["h1", "h2", "h3"])
        if h:
            chapter = _normalize_ws(h.get_text(" ", strip=True))
        for el in soup.find_all(["p", "li", "blockquote"]):
            txt = _normalize_ws(el.get_text(" ", strip=True))
            if txt:
                blocks.append(Block(text=txt, chapter=chapter, kind=classify_block(txt)))
    return ParsedDoc(title=meta_title, blocks=blocks or [Block(text="")])


# How much of a document's text may sit inside links before it is navigation
# rather than content. Real prose carries the occasional link; a contents page is
# almost nothing but.
EPUB_NAV_LINK_SHARE = 0.6


def _is_epub_nav(item, soup) -> bool:
    """Is this EPUB document the table of contents rather than a chapter?

    Three independent signals, because EPUB3 and EPUB2 declare it differently and
    plenty of files declare it not at all:
      * the spine marks the item ``properties="nav"``;
      * the markup carries ``<nav epub:type="toc">`` (or a ``toc``/``landmarks`` id);
      * the document is mostly link text, which is what a hand-rolled contents
        page looks like when nothing is declared.
    """
    props = getattr(item, "properties", None) or []
    if "nav" in props:
        return True

    for nav in soup.find_all("nav"):
        epub_type = nav.get("epub:type") or nav.get("type") or ""
        if "toc" in str(epub_type).lower() or "landmarks" in str(epub_type).lower():
            return True
    if soup.find(attrs={"id": re.compile(r"^(toc|contents|nav)$", re.I)}):
        return True

    name = (getattr(item, "file_name", "") or "").lower()
    if re.search(r"(^|/)(nav|toc|contents)[^/]*\.x?html?$", name):
        return True

    text_len = len(_normalize_ws(soup.get_text(" ", strip=True)))
    if text_len < 40:
        return False
    link_len = sum(
        len(_normalize_ws(a.get_text(" ", strip=True))) for a in soup.find_all("a")
    )
    return (link_len / text_len) >= EPUB_NAV_LINK_SHARE


# ----------------------------------------------------------------- mobi (best-effort)
def _parse_mobi(path: Optional[str], title: str) -> ParsedDoc:
    if not path:
        raise UnsupportedSourceError("No MOBI path provided.")
    try:
        import mobi  # optional, lazy
    except Exception as exc:  # noqa: BLE001
        raise UnsupportedSourceError(
            "MOBI support is not installed. Please convert to EPUB or PDF."
        ) from exc
    try:
        tmpdir, extracted_path = mobi.extract(path)
        ext = Path(extracted_path).suffix.lower()
        if ext in (".epub",):
            return _parse_epub(extracted_path, title)
        if ext in (".html", ".htm"):
            return _parse_html(_read(extracted_path), title)
        return _parse_plain(_read(extracted_path), title)
    except UnsupportedSourceError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise UnsupportedSourceError(
            "Could not parse MOBI file. Please convert to EPUB or PDF."
        ) from exc


# ----------------------------------------------------------------- helpers
def _read(path: Optional[str]) -> str:
    if not path:
        return ""
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def _normalize_ws(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    # Join hyphenated line breaks, collapse intra-paragraph single newlines into
    # spaces, but keep blank-line paragraph breaks.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _strip_md(text: str) -> str:
    text = re.sub(r"`{1,3}([^`]*)`{1,3}", r"\1", text)        # inline/code
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)            # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)        # links -> text
    text = re.sub(r"[*_~>#]+", "", text)                         # emphasis/marks
    return _normalize_ws(text)
