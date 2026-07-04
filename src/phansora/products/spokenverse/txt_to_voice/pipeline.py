# src/txt_to_voice/pipeline.py
#
# Batch TXT -> Audio pipeline (GPT-SoVITS backend).
# Supports chunk-level and file-level concurrency to speed up larger batches.

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from phansora.shared.utils.chunking import chunk_text
from phansora.shared.utils.naming import sanitize_stem
from phansora.shared.utils.ffmpeg import (
    concat_audio_files_ffmpeg,
    require_ffmpeg_if_needed,
    transcode_audio_ffmpeg,
)
from .adapters.backend import get_synthesizer, resolve_engine

LOG = logging.getLogger("txt_to_voice")


def _default_engine() -> str:
    # Resolves to "gptsovits" (the only engine); honours TTS_ENGINE for validation.
    return resolve_engine(None)


@dataclass(frozen=True)
class TTSConfig:
    voice: str  # GPT-SoVITS reference-clip path, or "default" for the built-in voice
    use_gpu: bool  # whether to request CUDA in the TTS backend
    rate: str  # accepted for compatibility; ignored by GPT-SoVITS
    volume: str  # accepted for compatibility; ignored by GPT-SoVITS
    output_format: str  # "mp3" or "wav"
    chunk_chars: int = 2500
    speaker: Optional[str] = None  # optional alias; treated as a reference-clip path
    language: Optional[str] = None  # en/zh/ja/ko/yue/auto

    # TTS engine ("gptsovits"); kept for forward-compatibility.
    engine: str = field(default_factory=_default_engine)
    # GPT-SoVITS reference clip for voice cloning.
    ref_audio: Optional[str] = None

    # NEW: how many chunks to synthesize in parallel per TXT file
    max_concurrency: int = 4
    # How many TXT files to process concurrently.
    file_concurrency: int = 1

    # GPT-SoVITS knobs; None => engine/env defaults.
    prompt_text: Optional[str] = None  # reference transcript (better quality; None => ref-free)
    speed: Optional[float] = None  # 0.6-1.65; speed_factor
    top_k: Optional[int] = None  # 1-100; GPT sampling
    top_p: Optional[float] = None  # 0-1; nucleus sampling
    temperature: Optional[float] = None  # 0.01-1.0
    repetition_penalty: Optional[float] = None  # 0-2


class BatchConverter:
    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg
        self._synthesize = get_synthesizer(cfg.engine)
        if cfg.volume != "+0%":
            LOG.warning("volume is currently ignored by the TTS backend.")

    def _iter_txt_files(self, input_dir: Path) -> List[Path]:
        # Recursive search ensures chapter outputs like
        # output_txt/<pdf_stem>_chapters/*.txt are included.
        return sorted([p for p in input_dir.rglob("*.txt") if p.is_file()])

    def _make_output_stem(self, txt_path: Path, in_dir: Path) -> str:
        """
        Build a collision-resistant stem from the input-relative path.
        This prevents chapter names like '01_Part_1.txt' from different books
        overwriting each other in a single output folder.
        """
        rel_no_suffix = txt_path.relative_to(in_dir).with_suffix("")
        flat_rel = "__".join(rel_no_suffix.parts)
        return sanitize_stem(flat_rel)

    async def _convert_one_txt(self, txt_path: Path, out_dir: Path, in_dir: Path) -> Path:
        text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            raise ValueError(f"Text file is empty: {txt_path.name}")

        chunks = chunk_text(text, self.cfg.chunk_chars)
        if not chunks:
            raise ValueError(f"No usable text after parsing: {txt_path.name}")

        safe_name = self._make_output_stem(txt_path, in_dir)
        final_audio = out_dir / f"{safe_name}.{self.cfg.output_format}"

        needs_concat = len(chunks) > 1
        needs_transcode = self.cfg.output_format != "wav"
        require_ffmpeg_if_needed(needs_concat or needs_transcode)

        tmp_dir = out_dir / ".tmp_chunks" / safe_name
        tmp_dir.mkdir(parents=True, exist_ok=True)

        LOG.info("Converting: %s", txt_path.name)
        LOG.info("  Chunks: %d", len(chunks))
        LOG.info("  Concurrency: %d", max(1, int(self.cfg.max_concurrency)))
        LOG.info("  Engine: %s", self.cfg.engine)
        LOG.info("  Device: %s", "cuda" if self.cfg.use_gpu else "cpu")

        chunk_files: List[Path] = []
        for i in range(1, len(chunks) + 1):
            chunk_files.append(tmp_dir / f"chunk_{i:04d}.wav")

        # --- NEW: synthesize chunks concurrently ---
        sem = asyncio.Semaphore(max(1, int(self.cfg.max_concurrency)))

        async def run_one(i: int, chunk_text_str: str, chunk_file: Path) -> None:
            async with sem:
                LOG.debug("  Synthesizing chunk %d/%d -> %s", i, len(chunks), chunk_file.name)
                await self._synthesize(
                    text=chunk_text_str,
                    out_path=chunk_file,
                    voice=self.cfg.voice,
                    use_gpu=self.cfg.use_gpu,
                    rate=self.cfg.rate,
                    volume=self.cfg.volume,
                    speaker=self.cfg.speaker,
                    language=self.cfg.language,
                    ref_audio=self.cfg.ref_audio,
                    prompt_text=self.cfg.prompt_text,
                    speed=self.cfg.speed,
                    top_k=self.cfg.top_k,
                    top_p=self.cfg.top_p,
                    temperature=self.cfg.temperature,
                    repetition_penalty=self.cfg.repetition_penalty,
                )

        tasks: List[asyncio.Task[None]] = []
        for i, chunk in enumerate(chunks, start=1):
            tasks.append(asyncio.create_task(run_one(i, chunk, chunk_files[i - 1])))

        # If any chunk fails, gather will raise; this is desired.
        await asyncio.gather(*tasks)
        # --- end NEW ---

        merged_audio = chunk_files[0]
        if len(chunk_files) > 1:
            merged_audio = tmp_dir / "merged.wav"
            LOG.info("  Concatenating chunks with ffmpeg -> %s", final_audio.name)
            concat_audio_files_ffmpeg(chunk_files, merged_audio)

        if self.cfg.output_format == "wav":
            final_audio.parent.mkdir(parents=True, exist_ok=True)
            merged_audio.replace(final_audio)
        else:
            transcode_audio_ffmpeg(merged_audio, final_audio)

        # Cleanup temp chunks
        try:
            for f in chunk_files:
                f.unlink(missing_ok=True)
            if merged_audio.name == "merged.wav":
                merged_audio.unlink(missing_ok=True)
            tmp_dir.rmdir()
        except Exception:
            LOG.debug("Temp cleanup skipped/partial (not fatal).", exc_info=True)

        LOG.info("Done: %s", final_audio)
        return final_audio

    async def convert_folder(self, in_dir: Path, out_dir: Path) -> int:
        if not in_dir.exists() or not in_dir.is_dir():
            LOG.error("Input directory not found or not a directory: %s", in_dir)
            return 2

        out_dir.mkdir(parents=True, exist_ok=True)
        txt_files = self._iter_txt_files(in_dir)

        if not txt_files:
            LOG.info("No .txt files found in %s", in_dir)
            return 0

        failures: List[Tuple[str, str]] = []

        if self.cfg.file_concurrency > 1:
            file_sem = asyncio.Semaphore(max(1, int(self.cfg.file_concurrency)))

            async def run_file(p: Path) -> None:
                async with file_sem:
                    try:
                        await self._convert_one_txt(p, out_dir, in_dir)
                    except Exception as e:
                        LOG.error("Failed converting %s: %s", p.name, e)
                        failures.append((p.name, str(e)))

            LOG.info("File concurrency: %d", max(1, int(self.cfg.file_concurrency)))
            await asyncio.gather(*(run_file(p) for p in txt_files))
        else:
            for p in txt_files:
                try:
                    await self._convert_one_txt(p, out_dir, in_dir)
                except Exception as e:
                    LOG.error("Failed converting %s: %s", p.name, e)
                    failures.append((p.name, str(e)))

        if failures:
            print("\nSome files failed:")
            for name, err in failures:
                print(f"  - {name}: {err}")
            return 1

        return 0
