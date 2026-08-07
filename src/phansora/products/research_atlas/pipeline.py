"""Research Atlas end to end: supplied sources in, sixteen-section report out.

    clean -> profile -> chunk per source -> extract each chunk -> merge -> connect -> render -> audit

What is deliberately NOT here, compared with the dossier pipeline this replaces:

  No table-of-contents generation. Research Atlas has sixteen fixed sections (report.py),
  so the document's shape is the product's definition rather than a model output that
  varied between runs.

  No chunk-organizing pass. That stage existed to turn source prose into chapters. Here
  the chunks feed EXTRACTION and the report is assembled from the extracted record, so
  organizing prose into a narrative would just be a second, less traceable copy of the
  material.

  No semantic chunking. Topic-coherent boundaries mattered when a chunk became a section
  of prose; for extraction they do not, and paragraph packing avoids loading an embedding
  model for every run. An entity split across a boundary is recovered by the merge, which
  joins on normalized names.

The map stage runs concurrently because these calls are network wait, not CPU -- the same
reasoning as config.llm_max_workers elsewhere.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import extraction, neutrality, report
from .config import Config
from .source_profiler import profile_sources
from .text_cleaner import clean_extracted_text, needs_cleanup


def chunk_source(text: str, max_chars: int, overlap_paragraphs: int = 1) -> List[str]:
    """Pack paragraphs up to max_chars, overlapping a little at each seam.

    The overlap is what keeps a sentence that straddles a boundary from being read as
    two half-sentences by two separate extraction calls.
    """
    paragraphs = [p.strip() for p in str(text or "").split("\n\n") if p.strip()]
    if not paragraphs:
        return []
    chunks: List[str] = []
    current: List[str] = []
    size = 0
    for para in paragraphs:
        if current and size + len(para) > max_chars:
            chunks.append("\n\n".join(current))
            current = current[-overlap_paragraphs:] if overlap_paragraphs else []
            size = sum(len(p) for p in current)
        current.append(para)
        size += len(para)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def run_pipeline(
    input_path: Optional[str] = None,
    toc_full_path: Optional[str] = None,
    max_chunk_chars: Optional[int] = None,
    sources: Optional[List[Dict[str, str]]] = None,
    title: str = "Research Atlas Report",
) -> dict:
    """Build the report.

    Signature is kept compatible with the pipeline this replaces so the HTTP layer and
    the Node worker do not have to change shape: `sources` is [{label, text}, ...] and
    `toc_full_path` is where the finished Markdown lands.
    """
    started = time.time()
    config = Config()
    client = config.deepseek_client
    out_path = toc_full_path or config.toc_full_path
    max_chars = max_chunk_chars or config.max_chunk_chars

    # 1. Resolve sources
    resolved = _resolve_sources(sources, input_path)
    if not resolved:
        raise ValueError("No usable sources were supplied.")
    labels = [s["label"] for s in resolved]
    print(f"[ATLAS] {len(resolved)} source(s): {', '.join(labels)}")

    # 2. Repair extraction damage, but only where there is damage to repair. A clean
    #    article costs a regex scan instead of an LLM pass.
    if config.clean_extracted_text:
        for src in resolved:
            damaged, score = needs_cleanup(src["text"], src["label"])
            if damaged or not config.skip_clean_when_undamaged:
                src["text"] = clean_extracted_text(
                    src["text"], client, chunk_size=config.cleanup_chunk_size
                ) or src["text"]
            else:
                print(f"[ATLAS] {src['label']}: undamaged (score {score:.2f}), skipping cleanup")

    # 3. Describe each source's FORM (transcript, filing, article). Descriptive only --
    #    the Source Index prints it and nothing weighs it.
    profiles: List[Any] = []
    if config.enable_source_profiling:
        try:
            profiles = profile_sources(
                resolved, client,
                sample_chars=config.profile_sample_chars,
                max_workers=config.llm_max_workers,
            )
        except Exception as e:  # noqa: BLE001 -- a missing profile costs a label, not the run
            print(f"[ATLAS] source profiling failed, continuing without it: {e}")

    # 4. MAP. Every unit is (label, chunk); the label rides along so the extracted
    #    record can be attributed by code rather than by the model.
    units: List[Tuple[str, str]] = []
    for src in resolved:
        for chunk in chunk_source(src["text"], max_chars, config.overlap_paragraphs):
            units.append((src["label"], chunk))
    print(f"[ATLAS] extracting from {len(units)} chunk(s) across {len(resolved)} source(s)")

    per_chunk: List[Tuple[str, Dict]] = []
    with ThreadPoolExecutor(max_workers=max(1, config.llm_max_workers)) as pool:
        futures = [
            (label, pool.submit(extraction.extract_from_chunk, client, chunk, label))
            for label, chunk in units
        ]
        for label, fut in futures:
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[ATLAS] chunk extraction failed for {label}: {e}")
                continue
            if rec:
                per_chunk.append((label, rec))
    print(f"[ATLAS] {len(per_chunk)}/{len(units)} chunk(s) produced a record")

    # 5. REDUCE
    record = extraction.merge_extractions(per_chunk)
    print(
        f"[ATLAS] merged: {len(record['people'])} people, "
        f"{len(record['organizations'])} organizations, {len(record['places'])} places, "
        f"{len(record['timeline'])} timeline entries, {len(record['claims'])} claims, "
        f"{len(record['relationships'])} relationships, "
        f"{len(record['date_conflicts'])} date disagreement(s)"
    )

    # 6. Connective sections, over the merged skeleton rather than the raw text.
    connective, violations = extraction.build_connective_sections(client, record, labels)
    if violations:
        print(f"[ATLAS] {len(violations)} characterization(s) redacted from the connective pass")

    # 7. Render and audit. The audit runs on the finished document, so it also catches
    #    anything the renderer's own phrasing introduces.
    markdown = report.render_report(record, connective, labels, title=title, profiles=profiles)
    residual = report.audit_report(markdown, labels)
    if residual:
        print("[ATLAS] NEUTRALITY WARNINGS IN FINAL REPORT:")
        print(neutrality.summarize(residual))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(markdown, encoding="utf-8")

    elapsed = time.time() - started
    print(f"[ATLAS] report written to {out_path} in {elapsed:.1f}s")

    return {
        "output_path": str(out_path),
        "sections": list(report.SECTIONS),
        "source_labels": labels,
        "chunk_count": len(units),
        "chunks_extracted": len(per_chunk),
        "counts": {
            "people": len(record["people"]),
            "organizations": len(record["organizations"]),
            "places": len(record["places"]),
            "timeline": len(record["timeline"]),
            "events": len(record["events"]),
            "claims": len(record["claims"]),
            "documents": len(record["documents"]),
            "relationships": len(record["relationships"]),
            "date_conflicts": len(record["date_conflicts"]),
            "gaps": len(record["gaps"]),
        },
        # Surfaced rather than swallowed: a caller can show that redaction happened.
        "neutrality": {
            "redacted_in_connective": len(violations),
            "residual_warnings": [v.describe() for v in residual],
            "clean": not residual,
        },
        "record": record,
        "connective": connective,
        "elapsed_seconds": round(elapsed, 1),
    }


def _resolve_sources(sources, input_path) -> List[Dict[str, str]]:
    """Normalize the two accepted input shapes into [{label, text}, ...]."""
    if sources:
        out = [
            {"label": str(s.get("label") or "source").strip(),
             "text": str(s.get("text") or "").strip()}
            for s in sources
        ]
        return [s for s in out if s["text"]]
    if input_path and Path(input_path).exists():
        raw = Path(input_path).read_text(encoding="utf-8", errors="ignore")
        return _parse_labeled(raw) or [{"label": Path(input_path).name, "text": raw.strip()}]
    return []


def _parse_labeled(text: str) -> List[Dict[str, str]]:
    """Split a file written with `=== SOURCE: label ===` markers back into sources."""
    import re
    parts = re.split(r"^===\s*SOURCE:\s*(.+?)\s*===\s*$", text, flags=re.MULTILINE)
    if len(parts) < 3:
        return []
    out: List[Dict[str, str]] = []
    for i in range(1, len(parts) - 1, 2):
        body = parts[i + 1].strip()
        if body:
            out.append({"label": parts[i].strip(), "text": body})
    return out
