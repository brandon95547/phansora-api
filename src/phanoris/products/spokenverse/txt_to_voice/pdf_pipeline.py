# src/txt_to_voice/pdf_pipeline.py
#
# New flow (Option A):
#   PDF -> render pages to images (PyMuPDF) -> OCR each page (Tesseract) ->
#   batch OCR text -> DeepSeek Chat cleans/merges -> write final .txt
#
# System deps (Pop!_OS/Ubuntu):
#   sudo apt update
#   sudo apt install -y tesseract-ocr
#
# Python deps (venv):
#   python -m pip install pymupdf pillow pytesseract aiohttp python-dotenv

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

from phanoris.products.spokenverse.services.pdf_render import PdfRenderConfig, render_pdf_to_images_bytes
from phanoris.products.spokenverse.services.ocr_tesseract import TesseractOCRConfig, ocr_image_bytes
from phanoris.shared.ai.deepseek import DeepSeekChatConfig, clean_ocr_text
from phanoris.shared.utils.naming import sanitize_stem

LOG = logging.getLogger("txt_to_voice")


@dataclass(frozen=True)
class PdfToTxtConfig:
    # Rendering
    render_dpi: int = 250
    image_format: str = "jpeg"  # "jpeg" or "png"
    jpeg_quality: int = 85

    # OCR (Tesseract)
    tesseract_lang: str = "eng"
    tesseract_psm: int = 3
    tesseract_oem: int = 3

    # DeepSeek cleaning batching
    batch_pages: int = 5              # how many pages to send per LLM call
    batch_max_chars: int = 20000      # safety cap per LLM call input
    max_output_tokens: int = 3500     # response cap per call (adjust if needed)
    clean_concurrency: int = 2        # parallel DeepSeek clean requests

    # Output formatting
    keep_page_breaks: bool = True     # keep page markers in the final text
    to_chapters: bool = False         # write one txt file per chapter
    target_chapter_chars: int = 18000 # fallback synthetic chapter target size

    # OCR speed
    ocr_concurrency: int = 4          # parallel OCR worker count


class PdfConverter:
    def __init__(self, cfg: PdfToTxtConfig | None = None) -> None:
        # Load .env from project root (spokenverse/.env)
        project_root = Path(__file__).resolve().parents[2]
        load_dotenv(dotenv_path=project_root / ".env")

        self.cfg = cfg or PdfToTxtConfig()
        self.chat_cfg = DeepSeekChatConfig.from_env()

        self.ocr_cfg = TesseractOCRConfig(
            lang=self.cfg.tesseract_lang,
            psm=self.cfg.tesseract_psm,
            oem=self.cfg.tesseract_oem,
        )

        self.render_cfg = PdfRenderConfig(
            dpi=self.cfg.render_dpi,
            image_format=self.cfg.image_format,
            jpeg_quality=self.cfg.jpeg_quality,
        )

    @staticmethod
    def _normalize_ws(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "")).strip()

    def _looks_like_toc_page(self, lines: List[str]) -> bool:
        """
        Detect obvious table-of-contents style pages to avoid false chapter starts.
        """
        if not lines:
            return False
        sample = [self._normalize_ws(ln) for ln in lines[:80]]
        toc_like = 0
        for ln in sample:
            if re.search(r"\.{2,}\s*\d+\s*$", ln):
                toc_like += 1
            elif re.match(r"^\d{1,3}\s+.+\s+\d{1,4}\s*$", ln):
                toc_like += 1
        return toc_like >= 5

    def _detect_chapter_title(self, page_text: str) -> str | None:
        """
        Detect likely chapter heading from top-of-page text.
        """
        lines = [ln.strip() for ln in (page_text or "").splitlines() if ln.strip()]
        if not lines:
            return None

        if self._looks_like_toc_page(lines):
            return None

        head = [self._normalize_ws(ln) for ln in lines[:24]]
        chapter_number_only = re.compile(
            r"^\s*(chapter|chap\.)\s+([0-9]+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten)\s*$",
            re.IGNORECASE,
        )
        patterns = [
            re.compile(
                r"^\s*(chapter|chap\.)\s+([0-9]+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten)\b.*$",
                re.IGNORECASE,
            ),
            re.compile(r"^\s*part\s+([0-9]+|[ivxlcdm]+)\b.*$", re.IGNORECASE),
            re.compile(r"^\s*(introduction|prologue|epilogue|conclusion)\b.*$", re.IGNORECASE),
            re.compile(
                r"^\s*([0-9]{1,3}|[ivxlcdm]{1,8})\s+['\"(A-Za-z].{6,120}$",
                re.IGNORECASE,
            ),
        ]

        for idx, line in enumerate(head):
            candidate = line
            if len(candidate) < 6 or len(candidate) > 140:
                continue
            if candidate.endswith("."):
                continue

            # Handle common style:
            #   CHAPTER 5
            #   The Title
            # where chapter number and title are split across lines.
            m = chapter_number_only.match(candidate)
            if m and (idx + 1) < len(head):
                next_line = head[idx + 1]
                if 4 <= len(next_line) <= 100 and not next_line.lower().startswith("page "):
                    combined = f"{candidate}: {next_line}"
                    return combined[:140]

            for pat in patterns:
                if pat.match(candidate):
                    return candidate
        return None

    def _build_chapter_ranges(self, page_texts: List[Tuple[int, str]]) -> List[Tuple[str, List[Tuple[int, str]]]]:
        """
        Returns chapter ranges as:
        [(chapter_title, [(page_num, page_text), ...]), ...]

        If no reliable headings are found, synthesize organized chapter segments by size.
        """
        starts: List[Tuple[int, str]] = []
        for page_num, txt in page_texts:
            title = self._detect_chapter_title(txt)
            if not title:
                continue
            if starts and page_num - starts[-1][0] < 4:
                continue
            starts.append((page_num, title))

        looks_reliable = len(starts) >= 4
        if looks_reliable:
            page_map: Dict[int, str] = dict(starts)
            chapter_ranges: List[Tuple[str, List[Tuple[int, str]]]] = []
            current_title = page_map.get(page_texts[0][0], "Front Matter")
            current_pages: List[Tuple[int, str]] = []

            for page_num, txt in page_texts:
                if page_num in page_map and current_pages:
                    chapter_ranges.append((current_title, current_pages))
                    current_title = page_map[page_num]
                    current_pages = []
                elif page_num in page_map:
                    current_title = page_map[page_num]
                current_pages.append((page_num, txt))

            if current_pages:
                chapter_ranges.append((current_title, current_pages))
            return chapter_ranges

        # Fallback chapter synthesis when no chapter headings are detected.
        chapter_ranges = []
        buf_pages: List[Tuple[int, str]] = []
        buf_chars = 0
        chapter_idx = 1
        target = max(12000, int(self.cfg.target_chapter_chars))

        for page_num, txt in page_texts:
            clean_txt = (txt or "").strip()
            if not clean_txt:
                continue

            page_len = len(clean_txt)
            buf_pages.append((page_num, clean_txt))
            buf_chars += page_len

            # Keep chapter segments reasonably sized and page-coherent.
            if buf_chars >= target and len(buf_pages) >= 8:
                start_p = buf_pages[0][0]
                end_p = buf_pages[-1][0]
                title = f"Chapter {chapter_idx:02d} (Pages {start_p}-{end_p})"
                chapter_ranges.append((title, buf_pages))
                chapter_idx += 1
                buf_pages = []
                buf_chars = 0

        if buf_pages:
            start_p = buf_pages[0][0]
            end_p = buf_pages[-1][0]
            title = f"Chapter {chapter_idx:02d} (Pages {start_p}-{end_p})"
            chapter_ranges.append((title, buf_pages))

        return chapter_ranges

    def _build_batches(self, page_texts: List[Tuple[int, str]]) -> List[str]:
        """
        Create batches of OCR text to send to DeepSeek cleaner.
        We include page markers in the batch so the model can respect boundaries.
        """
        batches: List[str] = []
        buf_lines: List[str] = []
        buf_chars = 0
        buf_pages = 0

        def flush() -> None:
            nonlocal buf_lines, buf_chars, buf_pages
            if buf_lines:
                batches.append("\n".join(buf_lines).strip())
            buf_lines = []
            buf_chars = 0
            buf_pages = 0

        for page_num, txt in page_texts:
            txt = (txt or "").strip()
            if not txt:
                continue

            block = f"--- PAGE {page_num} ---\n{txt}\n"
            block_len = len(block)

            # If adding this block would exceed caps, flush first
            if (buf_pages >= self.cfg.batch_pages) or (buf_chars + block_len > self.cfg.batch_max_chars):
                flush()

            buf_lines.append(block)
            buf_chars += block_len
            buf_pages += 1

        flush()
        return batches

    async def _clean_batches(self, batches: List[str], pdf_name: str) -> List[str]:
        """
        Clean OCR text batches concurrently (bounded by clean_concurrency).
        """
        if not batches:
            return []

        sem = asyncio.Semaphore(max(1, int(self.cfg.clean_concurrency)))
        cleaned_by_idx: Dict[int, str] = {}

        async def run_one(i: int, batch: str) -> None:
            async with sem:
                LOG.info("DeepSeek clean batch %d/%d: %s", i + 1, len(batches), pdf_name)
                cleaned = await clean_ocr_text(
                    batch,
                    cfg=self.chat_cfg,
                    max_output_tokens=self.cfg.max_output_tokens,
                )
                cleaned_by_idx[i] = (cleaned or "").strip()

        tasks = [asyncio.create_task(run_one(i, batch)) for i, batch in enumerate(batches)]
        await asyncio.gather(*tasks)
        return [cleaned_by_idx[i] for i in range(len(batches)) if cleaned_by_idx.get(i)]

    async def _ocr_pages(self, pages: List[Tuple[int, bytes]], pdf_name: str) -> List[Tuple[int, str]]:
        """
        OCR page images concurrently and return sorted (page_num, raw_text).
        """
        sem = asyncio.Semaphore(max(1, int(self.cfg.ocr_concurrency)))
        raw_by_page: Dict[int, str] = {}
        total = len(pages)

        async def run_one(page_num: int, image_bytes: bytes) -> None:
            async with sem:
                LOG.info("OCR (tesseract) page %d/%d: %s", page_num, total, pdf_name)
                raw = await asyncio.to_thread(ocr_image_bytes, image_bytes, cfg=self.ocr_cfg)
                raw_by_page[page_num] = raw

        tasks = [asyncio.create_task(run_one(page_num, image_bytes)) for page_num, image_bytes in pages]
        await asyncio.gather(*tasks)
        return [(p, raw_by_page[p]) for p, _ in pages if p in raw_by_page]

    async def _clean_page_block(self, page_texts: List[Tuple[int, str]], pdf_name: str) -> str:
        batches = self._build_batches(page_texts)
        if not batches:
            return ""
        cleaned_parts = await self._clean_batches(batches, pdf_name)
        return "\n\n".join(cleaned_parts).strip()

    async def _write_chapter_outputs(
        self,
        page_texts: List[Tuple[int, str]],
        pdf_name: str,
        out_txt_path: Path,
    ) -> Path:
        chapter_ranges = self._build_chapter_ranges(page_texts)
        if not chapter_ranges:
            raise RuntimeError("Could not build chapter ranges from OCR pages.")

        out_root = out_txt_path.parent
        out_root.mkdir(parents=True, exist_ok=True)
        safe_pdf_stem = sanitize_stem(out_txt_path.stem)

        written = 0
        for idx, (_chapter_title, chapter_pages) in enumerate(chapter_ranges, start=1):
            cleaned = await self._clean_page_block(chapter_pages, pdf_name)
            if not cleaned:
                # If the cleaner drops a chapter entirely, keep raw OCR pages so
                # chapter count/order are preserved.
                cleaned = "\n\n".join(
                    (txt or "").strip() for _, txt in chapter_pages if (txt or "").strip()
                ).strip()
            if not cleaned:
                continue

            if not self.cfg.keep_page_breaks:
                cleaned = cleaned.replace("--- PAGE ", "PAGE ").replace(" ---", "")

            out_path = out_root / f"{safe_pdf_stem}__{idx:02d}_Part_{idx}.txt"
            out_path.write_text(cleaned, encoding="utf-8", errors="replace")
            written += 1

        if written == 0:
            raise RuntimeError("No chapter files were produced after cleaning.")

        LOG.info("PDF -> chapter TXT files: %s -> %s (%d files)", pdf_name, out_root, written)
        return out_root

    async def convert_pdf_to_txt_async(self, pdf_path: Path, out_txt_path: Path) -> Path:
        if not pdf_path.exists() or not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        out_txt_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_bytes = pdf_path.read_bytes()

        # 1) Render PDF pages -> images
        pages = render_pdf_to_images_bytes(pdf_bytes, cfg=self.render_cfg)
        if not pages:
            raise RuntimeError("No pages rendered from PDF.")

        LOG.info("Rendering complete: %s (%d pages)", pdf_path.name, len(pages))

        # 2) Tesseract OCR per page -> raw text (concurrent)
        raw_pages = await self._ocr_pages(pages, pdf_path.name)

        # 3) chapter mode
        if self.cfg.to_chapters:
            return await self._write_chapter_outputs(raw_pages, pdf_path.name, out_txt_path)

        # 4) classic single-output mode
        final_text = await self._clean_page_block(raw_pages, pdf_path.name)
        if not final_text:
            raise RuntimeError("DeepSeek cleaning returned empty output.")

        # Optional: if you DON'T want page markers in final output, reduce marker style.
        if not self.cfg.keep_page_breaks:
            final_text = final_text.replace("--- PAGE ", "PAGE ").replace(" ---", "")

        out_txt_path.write_text(final_text, encoding="utf-8", errors="replace")
        LOG.info("PDF -> TXT (Tesseract + DeepSeek): %s -> %s", pdf_path.name, out_txt_path.name)
        return out_txt_path

    async def convert_folder_async(self, in_dir: Path, out_dir: Path) -> int:
        if not in_dir.exists() or not in_dir.is_dir():
            LOG.error("PDF input directory not found or not a directory: %s", in_dir)
            return 2

        out_dir.mkdir(parents=True, exist_ok=True)

        pdf_files = sorted([p for p in in_dir.glob("*.pdf") if p.is_file()])
        if not pdf_files:
            LOG.info("No .pdf files found in %s", in_dir)
            return 0

        failures: List[Tuple[str, str]] = []

        for pdf_path in pdf_files:
            try:
                out_txt = out_dir / f"{pdf_path.stem}.txt"
                await self.convert_pdf_to_txt_async(pdf_path, out_txt)
            except Exception as e:
                LOG.error("Failed converting %s: %s", pdf_path.name, e)
                failures.append((pdf_path.name, str(e)))

        if failures:
            print("\nSome PDFs failed:")
            for name, err in failures:
                print(f"  - {name}: {err}")
            return 1

        return 0
