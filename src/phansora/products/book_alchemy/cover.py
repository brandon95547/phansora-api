"""A course's cover and author, read out of the uploaded file while it still exists.

The dashboard shows a course as a book — a cover, a title, who wrote it — and the
only moment any of that can be read is the parse step: the source file is deleted
as soon as parsing is behind us (worker._cleanup_source). So this runs there,
once. The cover is parked beside the lesson audio and the author goes on the
project's ``options``; no schema changes, and a project made before this existed
simply has neither.

Everything here is best-effort. A book with no cover is still a book — the
dashboard draws one of its own — so nothing in this module may fail a parse:
:func:`extract` returns what it found and logs what it could not read.
"""
from __future__ import annotations

import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Optional

log = logging.getLogger("book_alchemy.cover")

# One name, so the route that serves a cover never has to trust a filename read
# back out of the database.
COVER_NAME = "cover.jpg"

# Twice the size the dashboard draws it at, for dense screens, and no more: this
# is a thumbnail on a card, and it is fetched every time the list is opened.
COVER_MAX_WIDTH = 600
COVER_MAX_HEIGHT = 900
# Smaller than this is a publisher's mark or a spacer image, not a cover.
COVER_MIN_SIDE = 96

# A PDF has no "cover" of its own, only a first page. That page is worth showing
# when it is a cover or a title page — an image, or a few lines of display type —
# and not when it is the opening page of an article, which shrinks to a grey
# rectangle of illegible text. The amount of text on it tells the two apart.
PDF_COVER_MAX_TEXT_CHARS = 400

AUTHOR_MAX_CHARS = 120
AUTHORS_SHOWN = 3

# What ends up in a document's author field when no person filled it in: the
# account that was logged in, or the program that made the file. Shown under a
# course title it reads as a mistake, so it is better left blank.
_NOT_A_PERSON = re.compile(
    r"\b(unknown|anonymous|administrator|admin|user|owner|author|untitled|default|"
    r"microsoft|adobe|acrobat|calibre|scanner|converter|office|pdf|word)\b",
    re.I,
)


def clean_author(raw) -> Optional[str]:
    """A name fit to print under a title, or None."""
    text = re.sub(r"\s+", " ", str(raw or "")).strip().strip(",;").strip()
    if not text or len(text) > AUTHOR_MAX_CHARS:
        return None
    if "@" in text or "/" in text or "\\" in text:      # an address or a path
        return None
    if not re.search(r"[^\W\d_]", text):                 # no letters at all
        return None
    if _NOT_A_PERSON.search(text):
        return None
    return text


def cover_file(project_dir: Path, name) -> Optional[Path]:
    """The stored cover for a project, if it has one."""
    if name != COVER_NAME:
        return None
    path = Path(project_dir) / COVER_NAME
    return path if path.is_file() else None


def extract(source_format: str, path: Optional[str], dest_dir: Path) -> dict:
    """Read ``{"author": ..., "cover": ...}`` out of a source file. Never raises.

    Either key is absent when the file has nothing usable for it. The cover is
    written into ``dest_dir`` as :data:`COVER_NAME`.
    """
    fmt = (source_format or "").lower().lstrip(".")
    if fmt not in ("pdf", "epub") or not path or not Path(path).is_file():
        return {}
    try:
        author, image = _from_pdf(path) if fmt == "pdf" else _from_epub(path)
    except Exception:  # noqa: BLE001 — see the module docstring
        log.warning("Could not read cover/author from %s", Path(path).name, exc_info=True)
        return {}

    found: dict = {}
    name = clean_author(author)
    if name:
        found["author"] = name
    if image:
        try:
            if _save_cover(image, Path(dest_dir) / COVER_NAME):
                found["cover"] = COVER_NAME
        except Exception:  # noqa: BLE001
            log.warning("Could not store a cover for %s", Path(path).name, exc_info=True)
    return found


def _from_pdf(path: str) -> tuple[Optional[str], Optional[bytes]]:
    import fitz  # PyMuPDF, lazy — same as parsers._parse_pdf

    with fitz.open(path) as pdf:
        author = (pdf.metadata or {}).get("author")
        if not pdf.page_count:
            return author, None
        page = pdf.load_page(0)
        if len((page.get_text("text") or "").strip()) > PDF_COVER_MAX_TEXT_CHARS:
            return author, None
        longest = max(page.rect.width, page.rect.height) or 1
        scale = min(3.0, COVER_MAX_HEIGHT / longest)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        return author, pixmap.tobytes("png")


def _from_epub(path: str) -> tuple[Optional[str], Optional[bytes]]:
    import ebooklib  # lazy — same as parsers._parse_epub
    from ebooklib import epub

    book = epub.read_epub(path)

    names: list[str] = []
    for entry in book.get_metadata("DC", "creator") or []:
        name = clean_author(entry[0] if entry else None)
        if name and name not in names:
            names.append(name)
    author = ", ".join(names[:AUTHORS_SHOWN]) or None

    return author, _epub_cover_bytes(book, ebooklib)


def _epub_cover_bytes(book, ebooklib) -> Optional[bytes]:
    """The cover image, by whichever of three declarations this file uses.

    EPUB3 marks the item itself; EPUB2 names its id in a <meta name="cover">; and
    a good many files do neither and just call the image "cover".
    """
    for item in book.get_items_of_type(ebooklib.ITEM_COVER):
        return item.get_content()

    for _value, attrs in book.get_metadata("OPF", "cover") or []:
        item = book.get_item_with_id((attrs or {}).get("content") or "")
        if item is not None:
            return item.get_content()

    for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
        label = f"{getattr(item, 'id', '')} {getattr(item, 'file_name', '')}".lower()
        if "cover" in label:
            return item.get_content()
    return None


def _save_cover(data: bytes, dest: Path) -> bool:
    """Normalize an image to a bounded JPEG. False if it is too small to be a cover."""
    from PIL import Image  # lazy; a declared dependency (requirements.txt)

    with Image.open(BytesIO(data)) as image:
        if min(image.size) < COVER_MIN_SIDE:
            return False
        # RGB: a JPEG has no alpha, and a palette or CMYK cover would not save as one.
        image = image.convert("RGB")
        image.thumbnail((COVER_MAX_WIDTH, COVER_MAX_HEIGHT))
        dest.parent.mkdir(parents=True, exist_ok=True)
        image.save(dest, "JPEG", quality=82, optimize=True)
    return True
