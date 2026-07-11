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
    return ParsedDoc(title=title, blocks=[Block(text=p) for p in paras] or [Block(text=raw)])


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
    cleaned = [Block(text=_strip_md(b.text), chapter=b.chapter) for b in blocks if b.text.strip()]
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
            blocks.append(Block(text=txt, chapter=current_heading))
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
    blocks: list[Block] = []
    total_chars = 0
    text_pages = 0
    page_count = 0
    with fitz.open(path) as pdf:
        page_count = pdf.page_count
        for pno in range(page_count):
            page = pdf.load_page(pno)
            txt = _normalize_ws(page.get_text("text") or "")
            if len(txt) >= 20:           # a page with real, extractable text
                text_pages += 1
            total_chars += len(txt)
            for para in re.split(r"\n{2,}", txt):
                para = para.strip()
                if para:
                    blocks.append(Block(text=para, page_start=pno + 1, page_end=pno + 1))

    if page_count == 0:
        raise UnsupportedSourceError("PDF has no pages.")

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
            blocks.append(Block(text=txt, chapter=current_heading))
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
        chapter = None
        h = soup.find(["h1", "h2", "h3"])
        if h:
            chapter = _normalize_ws(h.get_text(" ", strip=True))
        for el in soup.find_all(["p", "li", "blockquote"]):
            txt = _normalize_ws(el.get_text(" ", strip=True))
            if txt:
                blocks.append(Block(text=txt, chapter=chapter))
    return ParsedDoc(title=meta_title, blocks=blocks or [Block(text="")])


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
