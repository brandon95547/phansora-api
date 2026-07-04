# src/txt_to_voice/cli.py
# Updated to expose --max-concurrency for faster chunk synthesis.

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Optional, Sequence

from .pipeline import BatchConverter, TTSConfig
from .pdf_pipeline import PdfConverter, PdfToTxtConfig
from .adapters.backend import list_voices, resolve_engine

LOG = logging.getLogger("txt_to_voice")


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert .txt files to audio (GPT-SoVITS) OR convert PDFs to .txt."
    )

    # --- PDF -> TXT mode ---
    parser.add_argument("--pdf-to-txt", action="store_true", help="Convert PDFs to .txt and exit")
    parser.add_argument("--pdf-to-chapters", action="store_true", help="Convert PDFs directly to chapter .txt files (implies --pdf-to-txt --to-chapters)")
    parser.add_argument("--pdf-in", dest="pdf_input_dir", default="./input_pdf", help="Input folder containing .pdf files")
    parser.add_argument("--txt-out", dest="txt_output_dir", default="./output_txt", help="Output folder for extracted .txt files")
    parser.add_argument("--no-page-breaks", action="store_true", help="Do not insert page break markers in output .txt")
    parser.add_argument("--dpi", type=int, default=250, help="Render DPI for OCR (200-300 typical)")
    parser.add_argument("--ocr-lang", default="eng", help='Tesseract language(s), e.g. "eng" or "eng+spa"')
    parser.add_argument("--batch-pages", type=int, default=5, help="How many pages to send per DeepSeek clean call")
    parser.add_argument("--to-chapters", action="store_true", help="Write one output .txt per chapter")
    parser.add_argument("--ocr-concurrency", type=int, default=4, help="How many PDF pages to OCR concurrently")
    parser.add_argument("--clean-concurrency", type=int, default=2, help="How many DeepSeek clean calls to run concurrently")
    parser.add_argument("--target-chapter-chars", type=int, default=18000, help="Fallback synthetic chapter size when no headings are found")

    # --- TXT -> AUDIO mode ---
    parser.add_argument("--in", dest="input_dir", default="./output_txt")
    parser.add_argument("--out", dest="output_dir", default="./output_audio")
    parser.add_argument(
        "--engine",
        default=None,
        choices=["gptsovits"],
        help="TTS engine to use (GPT-SoVITS is the only engine)",
    )
    parser.add_argument(
        "--voice",
        default="default",
        help="GPT-SoVITS: 'default' for the built-in voice, or a path to a reference clip to clone from",
    )
    parser.add_argument(
        "--ref-audio",
        dest="ref_audio",
        default=None,
        help="Path to a reference clip to clone the voice from (takes priority over --voice)",
    )
    parser.add_argument("--gpu", action="store_true", help="Use CUDA/GPU for inference (NVIDIA + CUDA PyTorch required)")
    parser.add_argument("--rate", default="+0%", help="Accepted for compatibility; ignored by GPT-SoVITS")
    parser.add_argument("--volume", default="+0%", help="Accepted for compatibility; ignored by GPT-SoVITS")
    parser.add_argument("--speaker", default=None, help="Optional alias for --voice (reference-clip path)")
    parser.add_argument("--language", default=None, help="Text language: en/zh/ja/ko/yue/auto (default en)")
    parser.add_argument("--format", dest="output_format", default="mp3", choices=["mp3", "wav"])
    parser.add_argument("--chunk-chars", type=int, default=2500)

    # --- GPT-SoVITS generation knobs ---
    parser.add_argument("--prompt-text", dest="prompt_text", default=None, help="Transcript of the reference clip (better quality; omit for reference-free)")
    parser.add_argument("--speed", type=float, default=None, help="0.6-1.65; speed_factor (default 1.0)")
    parser.add_argument("--top-k", dest="top_k", type=int, default=None, help="1-100; GPT top-k sampling (default 5)")
    parser.add_argument("--top-p", dest="top_p", type=float, default=None, help="0-1; nucleus sampling (default 1.0)")
    parser.add_argument("--temperature", type=float, default=None, help="0.01-1.0; sampling randomness (default 1.0)")
    parser.add_argument("--repetition-penalty", dest="repetition_penalty", type=float, default=None, help="0-2; discourage repetition (default 1.35)")

    # NEW: concurrency
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Max number of TTS chunks to synthesize concurrently per TXT file",
    )
    parser.add_argument(
        "--file-concurrency",
        type=int,
        default=1,
        help="Max number of TXT files to convert concurrently",
    )

    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--list-voices", action="store_true", help="Print available voices for the selected --engine")

    return parser.parse_args(argv)


async def main_async(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    if args.pdf_to_chapters:
        args.pdf_to_txt = True
        args.to_chapters = True

    if args.list_voices:
        await list_voices(args.engine)
        return 0

    # --- PDF -> TXT path (runs and exits; does NOT call TTS pipeline) ---
    if args.pdf_to_txt:
        pdf_in = Path(args.pdf_input_dir)
        txt_out = Path(args.txt_output_dir)

        pdf_cfg = PdfToTxtConfig(
            keep_page_breaks=not args.no_page_breaks,
            render_dpi=args.dpi,
            tesseract_lang=args.ocr_lang,
            batch_pages=args.batch_pages,
            to_chapters=args.to_chapters,
            ocr_concurrency=args.ocr_concurrency,
            clean_concurrency=args.clean_concurrency,
            target_chapter_chars=args.target_chapter_chars,
        )
        pdf_converter = PdfConverter(pdf_cfg)

        return await pdf_converter.convert_folder_async(pdf_in, txt_out)

    # --- TXT -> AUDIO path ---
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    if not in_dir.exists() or not in_dir.is_dir():
        LOG.error("Input directory not found or not a directory: %s", in_dir)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = TTSConfig(
        voice=(args.speaker or args.voice),
        use_gpu=args.gpu,
        rate=args.rate,
        volume=args.volume,
        output_format=args.output_format,
        chunk_chars=args.chunk_chars,
        speaker=args.speaker,
        language=args.language,
        engine=resolve_engine(args.engine),
        ref_audio=args.ref_audio,
        max_concurrency=args.max_concurrency,  # NEW
        file_concurrency=args.file_concurrency,
        prompt_text=args.prompt_text,
        speed=args.speed,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
    )

    converter = BatchConverter(cfg)
    return await converter.convert_folder(in_dir, out_dir)


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))
