"""
main.py — Dossier Pipeline Orchestrator

Pipeline flow:
  1. Parse source-labeled text into individual source documents
  2. Profile each source (type, role, thesis, key claims) via LLM
  3. Clean extracted text (OCR artifacts, merged words)
  4. Generate dossier-style TOC from source profiles + topic headings
  5. Split into source-aware chunks (preserving === SOURCE: ... === headers)
  6. Organize chunks with dossier synthesis prompts (source attribution)
  7. Validate coverage, source balance, and duplication

Accepts two input modes:
  a) `sources` — list of {label, text} dicts from the API (preferred)
  b) `input_path` / `text` — single text blob with === SOURCE: ... === markers
     (backward-compatible)
"""

import argparse
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from .config import Config
from .embeddings import EmbeddingStore
from .toc_manager import TocManager
from .organizer import ChunkOrganizer, build_prompt_template
from .utils import load_text, split_text_into_chunks, split_text_semantic
from .toc_generator import TocGenerator
from .source_profiler import SourceProfile, profile_sources
from .validation import (
    compute_coverage,
    generate_loss_report,
    compute_source_balance,
    compute_duplication_ratio,
)
from .text_cleaner import clean_extracted_text
from .synthesis import synthesize_dossier, render_front_matter
from phansora.shared.paths import runtime_root

# Runtime data root (CWD / PHANSORA_DATA_DIR), not the installed package dir.
BASE_DIR = runtime_root()


# ------------------------------------------------------------------
# Source-text parser: split labeled text into individual source docs
# ------------------------------------------------------------------

_SOURCE_HEADER_RE = re.compile(r"^=== SOURCE:\s*(.+?)\s*===$", re.MULTILINE)


def parse_labeled_sources(text: str) -> List[Dict[str, str]]:
    """
    Split a text blob that contains === SOURCE: filename === markers
    into a list of {label, text} dicts.

    If no markers are found, the entire text is returned as a single
    source with label 'input'.
    """
    matches = list(_SOURCE_HEADER_RE.finditer(text))
    if not matches:
        return [{"label": "input", "text": text.strip()}]

    sources: List[Dict[str, str]] = []
    for i, m in enumerate(matches):
        label = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sources.append({"label": label, "text": body})

    return sources


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def run_pipeline(
    input_path: Optional[str] = None,
    toc_full_path: Optional[str] = None,
    max_chunk_chars: Optional[int] = None,
    sources: Optional[List[Dict[str, str]]] = None,
) -> dict:
    """
    Run the full dossier pipeline.

    Parameters
    ----------
    input_path : str, optional
        Path to a text file (may contain === SOURCE markers).
    toc_full_path : str, optional
        Output path for the dossier markdown.
    max_chunk_chars : int, optional
        Max characters per chunk for the organization pass.
    sources : list[dict], optional
        Pre-split sources as [{label, text}, ...].  If provided,
        input_path is ignored.
    """
    # 1. Setup configuration and services
    config = Config.from_env()
    embedding_store = EmbeddingStore(config)

    # Per-stage timing — prints where the wall-clock goes so we optimize the
    # proven hotspot (cleanup / TOC / organize) instead of guessing.
    _stage_times: List[tuple] = []

    def _timed(label, fn):
        _t = time.perf_counter()
        result = fn()
        dt = time.perf_counter() - _t
        _stage_times.append((label, dt))
        print(f"[TIMING] {label}: {dt:.1f}s")
        return result

    # 2. Resolve sources
    resolved_input_path: str = ""
    if sources:
        # Sources provided directly by the API
        print(f"[PIPELINE] Received {len(sources)} pre-split sources.")
    else:
        # Load from file and parse source markers
        resolved_input_path = str(Path(input_path or config.input_text_path))
        full_text = load_text(resolved_input_path)
        sources = parse_labeled_sources(full_text)
        print(f"[PIPELINE] Parsed {len(sources)} source(s) from input file.")

    # 3. AI text cleanup per source
    if config.clean_extracted_text:
        print("[PIPELINE] Running AI text cleanup on each source...")
        _t = time.perf_counter()
        for src in sources:
            src["text"] = clean_extracted_text(
                src["text"],
                client=config.deepseek_client,
                chunk_size=config.cleanup_chunk_size,
            )
        _stage_times.append(("text cleanup", time.perf_counter() - _t))
        total_chars = sum(len(s["text"]) for s in sources)
        print(f"[TIMING] text cleanup: {_stage_times[-1][1]:.1f}s")
        print(f"[PIPELINE] Text cleanup complete ({total_chars} chars across {len(sources)} sources).")

    # 4. Source profiling — classify type, role, thesis, claims
    source_profiles: List[SourceProfile] = []
    if config.enable_source_profiling and len(sources) > 1:
        print("[PIPELINE] Profiling sources...")
        source_profiles = profile_sources(
            sources=sources,
            client=config.deepseek_client,
            sample_chars=config.profile_sample_chars,
        )
        for p in source_profiles:
            print(f"  {p.source_label}: type={p.source_type}, role={p.rhetorical_role}")
    elif len(sources) == 1:
        # Single source — create a minimal profile without LLM call
        source_profiles = [
            SourceProfile(
                source_label=sources[0]["label"],
                source_type="unknown",
                rhetorical_role="mixed",
                central_argument="(single source — no profiling needed)",
                char_count=len(sources[0]["text"]),
            )
        ]

    # 4b. Cross-source intelligence synthesis (Phase 1). The one stage that sees
    # every source together: merges duplicate facts into single findings with
    # confidence + supporting sources, builds a timeline, cross-source agreement/
    # conflict analysis, evidence matrix, and an executive summary. Rendered as the
    # dossier's leading front matter (prepended after the body is assembled, below).
    # Best-effort: any failure leaves front_matter empty and the dossier proceeds.
    front_matter = ""
    if config.enable_correlation and len(sources) > 1:
        print("[PIPELINE] Running cross-source intelligence synthesis...")
        intel_model = _timed(
            "synthesis",
            lambda: synthesize_dossier(
                sources=sources,
                source_profiles=source_profiles,
                client=config.deepseek_client,
                sample_chars=config.synthesis_sample_chars,
            ),
        )
        if intel_model:
            source_labels = [s["label"] for s in sources]
            front_matter = render_front_matter(intel_model, source_labels)
            print(
                f"[PIPELINE] Synthesis produced {len(intel_model.get('findings') or [])} "
                f"findings, {len(intel_model.get('timeline') or [])} timeline events."
            )

    # 5. Build the merged text (with source headers) for TOC extraction
    merged_text = _build_source_labeled_text(sources)

    # 6. TOC generation — dossier-style
    resolved_toc_path = Path(toc_full_path or config.toc_full_path)
    toc_generator = TocGenerator(config, embedding_store)

    _t = time.perf_counter()
    if len(sources) > 1 and source_profiles:
        print("[PIPELINE] Generating dossier-style TOC...")
        toc_generator.generate_dossier_from_sources(
            merged_text,
            source_profiles=source_profiles,
            toc_full_path=str(resolved_toc_path),
        )
    else:
        print("[PIPELINE] Single source — generating topic-based TOC...")
        toc_generator.generate_from_text(
            merged_text,
            toc_full_path=str(resolved_toc_path),
        )
    _stage_times.append(("TOC generation", time.perf_counter() - _t))
    print(f"[TIMING] TOC generation: {_stage_times[-1][1]:.1f}s")

    # 7. Load TOC and run completeness pre-check
    toc = TocManager(str(resolved_toc_path))
    toc_completeness = toc_generator.check_toc_completeness(
        merged_text,
        headings=toc.get_heading_titles(),
    )

    # 8. Split the source-labeled text into chunks (preserving source headers)
    resolved_max_chunk_chars = int(max_chunk_chars or config.max_chunk_chars)
    chunks = _build_source_aware_chunks(sources, embedding_store, resolved_max_chunk_chars, config)
    prompt_template = build_prompt_template(toc)

    # 9. Organize chunks — dossier synthesis with source attribution
    organizer = ChunkOrganizer(
        client=config.deepseek_client,
        toc=toc,
        embedding_store=embedding_store,
        prompt_template=prompt_template,
        conservative_mode=config.conservative_mode,
        catchall_heading=config.catchall_heading,
        content_similarity_threshold=config.content_similarity_threshold,
        source_profiles=source_profiles,
        max_source_share=config.max_source_share,
        claim_dedup_threshold=config.claim_dedup_threshold,
    )
    organized_sections = _timed("organize chunks", lambda: organizer.organize_chunks(chunks))
    organizer.insert_sections(organized_sections)

    # 10. Validate coverage
    output_text = Path(resolved_toc_path).read_text(encoding="utf-8")
    coverage = compute_coverage(
        original_text=merged_text,
        output_text=output_text,
        embedding_store=embedding_store,
        threshold=config.coverage_threshold,
    )

    # 10b. Source balance and duplication checks
    source_balance = compute_source_balance(organized_sections)
    duplication_ratio = compute_duplication_ratio(
        organized_sections,
        embedding_store=embedding_store,
        threshold=config.claim_dedup_threshold,
    )

    # 10c. Write loss report
    loss_report = generate_loss_report(
        coverage,
        source_balance=source_balance,
        duplication_ratio=duplication_ratio,
        max_source_share=config.max_source_share,
        max_duplication_ratio=config.max_duplication_ratio,
    )
    report_path = Path(resolved_toc_path).parent / "loss_report.md"
    report_path.write_text(loss_report, encoding="utf-8")

    # 10d. Prepend the intelligence front matter so the dossier LEADS with the
    # executive summary, timeline, findings, cross-source analysis, and evidence
    # matrix — above the per-section synthesized body. Done after coverage/validation
    # so those metrics reflect the source-derived body, not the synthesized summary.
    if front_matter:
        dossier_body = Path(resolved_toc_path).read_text(encoding="utf-8")
        Path(resolved_toc_path).write_text(
            front_matter + "\n\n" + dossier_body, encoding="utf-8"
        )

    coverage_score = coverage["coverage_score"]
    if coverage_score < 0.95:
        print(f"⚠️  Coverage score: {coverage_score:.1%} — some content may have been lost.")
        print(f"   See {report_path} for details.")
    else:
        print(f"✅ Coverage score: {coverage_score:.1%}")

    if duplication_ratio > config.max_duplication_ratio:
        print(f"⚠️  Duplication ratio: {duplication_ratio:.1%} (max: {config.max_duplication_ratio:.0%})")

    _total = sum(dt for _, dt in _stage_times)
    print(
        f"[TIMING] LLM stages total: {_total:.1f}s — "
        + ", ".join(f"{label} {dt:.1f}s" for label, dt in _stage_times)
    )

    return {
        "ok": True,
        "input_path": resolved_input_path,
        "toc_full_path": str(resolved_toc_path),
        "chunk_count": len(chunks),
        "organized_section_count": len(organized_sections),
        "coverage_score": coverage_score,
        "loss_report_path": str(report_path),
        "toc_completeness_ratio": toc_completeness["coverage_ratio"],
        "toc_uncovered_paragraphs": len(toc_completeness["uncovered"]),
        "source_count": len(sources),
        "source_balance": source_balance,
        "duplication_ratio": duplication_ratio,
        "has_front_matter": bool(front_matter),
    }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_source_labeled_text(sources: List[Dict[str, str]]) -> str:
    """Join all sources into a single text with === SOURCE: ... === headers."""
    parts: List[str] = []
    for src in sources:
        label = src.get("label", "input")
        text = (src.get("text") or "").strip()
        if text:
            parts.append(f"=== SOURCE: {label} ===\n\n{text}")
    return "\n\n".join(parts)


def _build_source_aware_chunks(
    sources: List[Dict[str, str]],
    embedding_store: EmbeddingStore,
    max_chunk_chars: int,
    config,
) -> List[str]:
    """
    Split each source independently, then prepend the source header to each chunk
    so the organizer always knows which source a chunk came from.
    """
    all_chunks: List[str] = []
    for src in sources:
        label = src.get("label", "input")
        text = (src.get("text") or "").strip()
        if not text:
            continue

        # Split this source's text into semantically coherent chunks
        source_chunks = split_text_semantic(
            text,
            embedding_fn=embedding_store.get_embedding,
            max_chars=max_chunk_chars,
            overlap_paragraphs=config.overlap_paragraphs,
            similarity_drop=config.semantic_similarity_drop,
        )
        # Prepend the source header to each chunk
        for chunk in source_chunks:
            labeled_chunk = f"=== SOURCE: {label} ===\n\n{chunk}"
            all_chunks.append(labeled_chunk)

    print(f"[PIPELINE] Created {len(all_chunks)} source-labeled chunks.")
    return all_chunks


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description="Run Dossier Nova pipeline.")
    parser.add_argument(
        "--input",
        default=None,
        help="Input text file path. Defaults to INPUT_TEXT_PATH from env.",
    )
    parser.add_argument(
        "--toc",
        default=None,
        help="Output TOC markdown path. Defaults to TOC_FULL_PATH from env.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Max chars per content chunk. Defaults to MAX_CHUNK_CHARS from env.",
    )
    args = parser.parse_args(argv)

    result = run_pipeline(
        input_path=args.input,
        toc_full_path=args.toc,
        max_chunk_chars=args.max_chars,
    )
    print(
        "✅ Dossier generated "
        f"(sources={result['source_count']}, chunks={result['chunk_count']}, "
        f"sections={result['organized_section_count']})."
    )


if __name__ == "__main__":
    main()
