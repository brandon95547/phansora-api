"""An uploaded book or document, read into chapters (POST /sources/extract).

Parsing is Book Alchemy's (``book_alchemy.parsers.parse_source``): it returns the text as
blocks carrying chapter and page provenance, and marks navigation — tables of contents,
indexes, running heads — as "apparatus", which is dropped here. What this module adds is
the grouping into chapters:

  * DOCX, Markdown and EPUB name their chapters themselves (headings, documents).
  * A PDF's chapters come from its outline (bookmarks) when it has one.
  * Otherwise, in a PDF or a plain text file, a line such as "Chapter 3" or "Prologue"
    at the start of a block opens a chapter.

A scanned PDF (no text to select) is read first by the shared OCR (phansora.shared.ocr).

When none of that finds at least two chapters, a DOCX is read again by its top heading
level, and a PDF by its larger type (material.py); only when that finds nothing either is
the text cut into parts of about PART_WORDS words, at paragraph boundaries, titled
"Part 1", "Part 2"…

material.py also reads what the chapters leave out: the sections under them, and the
pictures, drawings and tables, kept as they are.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import UploadFile

from phansora.products.book_alchemy.parsers import ParsedDoc, ScannedPdfError, UnsupportedSourceError, parse_source

from . import material
from .errors import ServiceError
from .schemas import ExtractedAsset, ExtractedChapter, ExtractedSection, ExtractResponse

logger = logging.getLogger("talari")

FORMATS = {
    ".pdf": "pdf",
    ".epub": "epub",
    ".docx": "docx",
    ".txt": "txt",
    ".text": "txt",
    ".md": "md",
    ".markdown": "md",
}
MAX_UPLOAD_BYTES = 60 * 1024 * 1024
PART_WORDS = 3000
# A last part shorter than this share of PART_WORDS joins the one before it.
SHORT_LAST_PART = 0.3
UNTITLED_OPENING = "Opening"
# Untitled text before the first chapter shorter than this is front matter, and dropped.
FRONT_MATTER_WORDS = 40

# A scanned PDF (no text to select) is read by the shared OCR (phansora.shared.ocr): its pages
# recognized by Tesseract, OCR_WORKERS at a time, and tidied by DeepSeek, OCR_CLEAN batches
# at a time, the model TALARI_MODEL's. Minutes for a long book; the import job waits.
OCR_WORKERS = max(1, int(os.getenv("TALARI_OCR_WORKERS", "6")))
OCR_CLEAN = max(1, int(os.getenv("TALARI_OCR_CLEAN_CONCURRENCY", "4")))
# The page markers in OCR'd text ("--- PAGE 3 ---", or "PAGE 3" once they are plain): taken
# out, so a "Chapter 3" at the top of a page opens its block and is found as a chapter.
PAGE_MARK = re.compile(r"^[ \t]*(?:-{2,}[ \t]*)?PAGE[ \t]+\d+(?:[ \t]*-{2,})?[ \t]*$", re.MULTILINE | re.IGNORECASE)
UNSUPPORTED = "Talari can read PDF, EPUB, DOCX, TXT and Markdown (.md) files."

_NUMBER = (
    r"(?:\d{1,3}|[IVXLCDM]{1,7}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty(?:[- ]\w+)?|thirty(?:[- ]\w+)?)"
)
# "Chapter 3", "CHAPTER III. The Storm", "Part One", "Prologue", "Appendix B: Tables".
CHAPTER_HEADING = re.compile(
    rf"^(?:(?:chapter|part|book)\s+{_NUMBER}\b|prologue|epilogue|introduction|preface|foreword|"
    rf"afterword|conclusion|interlude|appendix(?:\s+[A-Z0-9]{{1,3}})?)(?:[.:—–-]?\s+\S.*|[.:]?)$",
    re.IGNORECASE,
)
HEADING_MAX_CHARS = 70
SUBTITLE_MAX_WORDS = 10
DOT_LEADER = re.compile(r"\.\s*\.\s*\.|…")


def source_format(filename: str | None) -> str | None:
    return FORMATS.get(Path(filename or "").suffix.lower())


async def save_upload(upload: UploadFile, path: Path) -> int:
    """Copy an upload to ``path`` in pieces, refusing one over the size limit (413)."""
    size = 0
    with path.open("wb") as out:
        while chunk := await upload.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise ServiceError(f"That file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.", status=413)
            out.write(chunk)
    if size == 0:
        raise ServiceError("That file is empty.", status=422)
    return size


def _decode(data: bytes) -> str:
    """A text file's characters. UTF-8 first (with or without a BOM), then UTF-16 if it says
    so, then Windows-1252, which is what a text file that is not UTF-8 nearly always is."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def _title_from_filename(filename: str | None) -> str:
    stem = Path(filename or "").stem
    return " ".join(re.sub(r"[_]+", " ", stem).split()) or "Untitled"


@dataclass
class _Chapter:
    title: str | None
    paragraphs: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    # The pages each paragraph is on, in step with ``paragraphs``, so a part cut from
    # them knows its own pages rather than its whole chapter's.
    spans: list[list[int]] = field(default_factory=list)

    @property
    def words(self) -> int:
        return sum(len(p.split()) for p in self.paragraphs)


def _heading(text: str) -> tuple[str, str] | None:
    """A chapter heading at the top of a block: (title, the rest of the block)."""
    lines = [line.strip() for line in text.split("\n")]
    first = lines[0]
    if len(first) > HEADING_MAX_CHARS or DOT_LEADER.search(first) or not CHAPTER_HEADING.match(first):
        return None
    title, rest = first.rstrip(".:"), lines[1:]
    # "CHAPTER 3" on one line and "The Storm" on the next are one heading.
    if rest and re.fullmatch(rf"(?:chapter|part|book)\s+{_NUMBER}", title, re.IGNORECASE):
        subtitle = rest[0]
        if subtitle and len(subtitle.split()) <= SUBTITLE_MAX_WORDS and not re.search(r"[.,;!?]$", subtitle):
            title, rest = f"{title}: {subtitle}", rest[1:]
    return title, "\n".join(rest).strip()


def _is_contents(text: str) -> bool:
    """A block that is a list of chapter headings is a table of contents, page numbers or
    not ("Chapter 1: Beginnings" / "Chapter 2: The Road" / …), not the start of chapter 1."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if len(lines) < 3:
        return False
    return sum(1 for line in lines if CHAPTER_HEADING.match(line)) / len(lines) >= 0.6


def _pdf_outline(path: Path) -> tuple[int | None, list[tuple[int, str]]]:
    """The PDF's page count, and its outline as (first page, title), at the shallowest
    level that names at least two chapters (a lone level-1 entry is usually the book)."""
    import fitz  # PyMuPDF; parse_source has already opened this file with it

    try:
        with fitz.open(path) as pdf:
            pages = pdf.page_count
            toc = pdf.get_toc(simple=True)
    except Exception:  # noqa: BLE001 — an unreadable outline only means no outline
        return None, []
    entries = [(level, " ".join(str(title).split()), page) for level, title, page in toc if page and page >= 1]
    for level in sorted({level for level, _, _ in entries}):
        chosen = sorted({page: title for lvl, title, page in reversed(entries) if lvl == level and title}.items())
        if len(chosen) >= 2:
            return pages, chosen
    return pages, []


def _outline_title(outline: list[tuple[int, str]], page: int | None) -> str | None:
    found = None
    for start, title in outline:
        if page is None or start > page:
            break
        found = title
    return found


def _group(doc: ParsedDoc, outline: list[tuple[int, str]]) -> list[_Chapter]:
    """The document's prose, grouped into its own chapters (when it has any)."""
    chapters: list[_Chapter] = []
    for block in doc.blocks:
        text = (block.text or "").strip()
        if not text:
            continue
        label: str | None = None
        found = None
        if block.chapter:
            label = block.chapter
        elif outline:
            label = _outline_title(outline, block.page_start)
        elif block.kind != "apparatus" or "\n" not in text:
            # "Chapter 12" on a line of its own reads to the apparatus filter like a
            # contents entry ("Genesis 12"), so headings are looked for first.
            found = _heading(text)
            if found:
                label, text = found
        if (block.kind == "apparatus" and not found) or (not block.chapter and _is_contents(block.text)):
            continue
        # A block with no chapter of its own belongs to the one before it; so does one
        # repeating that chapter's heading, which is what a running head looks like.
        if label and (not chapters or chapters[-1].title != label):
            chapters.append(_Chapter(title=label))
        elif not chapters:
            chapters.append(_Chapter(title=None))
        if text and block.kind != "apparatus":
            pages = [p for p in (block.page_start, block.page_end) if p]
            chapters[-1].paragraphs.append(text)
            chapters[-1].pages.extend(pages)
            chapters[-1].spans.append(pages)
    return [chapter for chapter in chapters if chapter.paragraphs]


def _split_long(paragraph: str) -> list[str]:
    """A paragraph too long to sit inside one part, at line breaks or else sentence ends."""
    if len(paragraph.split()) <= PART_WORDS * 2:
        return [paragraph]
    pieces = paragraph.split("\n") if "\n" in paragraph else re.split(r"(?<=[.!?])\s+", paragraph)
    out: list[str] = []
    current: list[str] = []
    words = 0
    for piece in pieces:
        current.append(piece)
        words += len(piece.split())
        if words >= PART_WORDS:
            out.append(("\n" if "\n" in paragraph else " ").join(current))
            current, words = [], 0
    if current:
        out.append(("\n" if "\n" in paragraph else " ").join(current))
    return out


def _parts(chapters: list[_Chapter]) -> list[_Chapter]:
    """All the prose, cut into parts of about PART_WORDS words at paragraph boundaries."""
    parts: list[_Chapter] = []
    current = _Chapter(title=None)
    for chapter in chapters:
        for number, paragraph in enumerate(chapter.paragraphs):
            pages = chapter.spans[number] if number < len(chapter.spans) else chapter.pages
            for piece in _split_long(paragraph):
                current.paragraphs.append(piece)
                current.pages.extend(pages)
                if current.words >= PART_WORDS:
                    parts.append(current)
                    current = _Chapter(title=None)
    if current.paragraphs:
        if parts and current.words < PART_WORDS * SHORT_LAST_PART:
            parts[-1].paragraphs.extend(current.paragraphs)
            parts[-1].pages.extend(current.pages)
        else:
            parts.append(current)
    for number, part in enumerate(parts, start=1):
        part.title = f"Part {number}"
    return parts


def _read_scanned(path: Path, filename: str | None) -> ParsedDoc:
    """A scanned PDF's text, by the shared OCR, parsed as plain text (its chapters then come
    from its headings, or its parts). Blocking — extract already runs in a thread, with no
    event loop of its own. Its pages are kept, as they are read, in a folder named after the
    file, so a reading that is cut short and asked again picks up where it stopped."""
    from phansora.shared.ocr.pdf import PdfConverter, PdfToTxtConfig, explain_failure  # lazy: Tesseract is a system dependency

    work = Path(tempfile.gettempdir()) / "talari-ocr" / path.stem
    out = work / "text.txt"
    cfg = PdfToTxtConfig(keep_page_breaks=False, to_chapters=False, ocr_concurrency=OCR_WORKERS, clean_concurrency=OCR_CLEAN, model_var="TALARI_MODEL")
    try:
        asyncio.run(PdfConverter(cfg).convert_pdf_to_txt_async(path, out, cache_dir=work / "pages"))
        text = PAGE_MARK.sub("\n", out.read_text(encoding="utf-8", errors="ignore")).strip()
        if not text:
            raise ServiceError("This PDF looks scanned, and no text could be read from its pages.", status=422)
        doc = parse_source(source_format="text", text=text, title_hint=_title_from_filename(filename))
    except ServiceError:
        raise
    except Exception as error:  # noqa: BLE001 — Tesseract, DeepSeek or the file itself
        logger.exception("Talari: could not read a scanned PDF")
        raise ServiceError(explain_failure(error), status=422) from None
    shutil.rmtree(work, ignore_errors=True)
    return doc


def extract(
    path: Path, fmt: str, *, filename: str | None, title: str | None, pictures: material.Pictures | None = None,
    assets: bool = True, budget: float = material.ASSET_SECONDS,
) -> ExtractResponse:
    """Read a saved upload into chapters. Blocking (PDF parsing is CPU work): run it in a thread.

    With ``pictures`` (where to keep them), the response also carries the sections under
    the chapters and the pictures, drawings and tables found in the file — a PDF's only
    with ``assets``, since finding them can take minutes (``assetsPending`` says they were
    left for later). The same file read twice is read the same, so the chapter and section
    numbers of one reading hold for the other."""
    try:
        if fmt in ("txt", "md"):
            doc = parse_source(source_format=fmt, text=_decode(path.read_bytes()), title_hint=_title_from_filename(filename))
        else:
            doc = parse_source(source_format=fmt, path=str(path), title_hint=_title_from_filename(filename))
    except ScannedPdfError:
        doc = _read_scanned(path, filename)
    except UnsupportedSourceError as error:
        logger.info("Talari: could not read a %s upload: %s", fmt, error)
        raise ServiceError(f"This {fmt.upper()} file could not be read.", status=422) from None
    except Exception:  # noqa: BLE001 — a damaged file fails inside a third-party parser, any way it likes
        logger.exception("Talari: could not parse a %s upload", fmt)
        raise ServiceError(
            f"This {fmt.upper()} file could not be read. It may be damaged, or not really a {fmt.upper()} file.",
            status=422,
        ) from None

    pages, outline = _pdf_outline(path) if fmt == "pdf" else (None, [])
    grouped = _group(doc, outline)
    if not grouped:
        raise ServiceError("No text could be found in this file.", status=422)

    titled = [chapter for chapter in grouped if chapter.title]
    # A PDF is read for its structure once, for both its chapters and its library.
    reading = None
    if fmt == "pdf" and (pictures is not None or (not outline and len(titled) < 2)):
        try:
            reading = material.PdfReading(path)
        except Exception:  # noqa: BLE001 — no structure is a fallback, never a failure
            logger.exception("Talari: could not read a PDF's structure")
    # A DOCX by its top heading level (Heading 1, with 2 and 3 as its sections), and a PDF
    # with no outline and no "Chapter 3" lines by its largest recurring type.
    docx_sections, docx_assets = None, None
    restructured: list[_Chapter] = []
    if fmt == "docx" and pictures is not None:
        found = material.docx_structure(path, pictures)
        if found:
            structured, docx_sections, docx_assets = found
            if len([c for c in structured if c.title]) >= 2:
                restructured = [_Chapter(title=c.title or None, paragraphs=c.paragraphs) for c in structured]
    elif fmt == "pdf" and not outline and len(titled) < 2:
        restructured = [
            _Chapter(title=c.title, paragraphs=c.paragraphs, pages=c.pages, spans=[[]] * len(c.paragraphs))
            for c in (material.pdf_chapters(path, reading) if reading else [])
        ]
    if restructured:
        chapters = restructured
        if chapters[0].title is None and chapters[0].words < FRONT_MATTER_WORDS:
            chapters = chapters[1:]
        elif chapters[0].title is None:
            chapters[0].title = UNTITLED_OPENING
    elif len(titled) >= 2:
        chapters = grouped
        if chapters[0].title is None:
            if chapters[0].words < FRONT_MATTER_WORDS:
                # A title page, a "Contents" heading whose entries were apparatus, a
                # dedication: front matter, not a chapter anyone would write from.
                chapters = chapters[1:]
            else:
                chapters[0].title = UNTITLED_OPENING
    else:
        chapters = _parts(grouped)
        if len(chapters) == 1 and len(titled) == 1:
            chapters[0].title = titled[0].title

    out: list[ExtractedChapter] = []
    for index, chapter in enumerate(chapters):
        start, end = (min(chapter.pages), max(chapter.pages)) if chapter.pages else (None, None)
        out.append(
            ExtractedChapter(
                index=index,
                title=chapter.title or f"Part {index + 1}",
                words=chapter.words,
                page_start=start,
                page_end=end,
                text="\n\n".join(chapter.paragraphs),
            )
        )
    found = material.Material()
    if pictures is not None:
        found = material.read(
            path, fmt, out, outline=outline, pictures=pictures, docx_sections=docx_sections, docx_assets=docx_assets,
            assets=assets, budget=budget, reading=reading,
        )
    return ExtractResponse(
        title=" ".join((title or "").split()) or doc.title or _title_from_filename(filename),
        format=fmt,
        pages=pages,
        words=sum(chapter.words for chapter in out),
        chapters=out,
        sections=[
            ExtractedSection(
                index=index, chapter_index=s.chapter_index, level=s.level, title=s.title,
                page_start=s.page_start, page_end=s.page_end, text=s.text, words=len(s.text.split()),
            )
            for index, s in enumerate(found.sections)
        ],
        assets=[
            ExtractedAsset(
                index=index, kind=a.kind, src=a.src, width=a.width, height=a.height, page=a.page, caption=a.caption,
                rows=a.rows, header=a.header, chapter_index=a.chapter_index, section_index=a.section_index,
            )
            for index, a in enumerate(found.assets)
        ],
        assets_pending=fmt == "pdf" and pictures is not None and not assets,
    )
