"""
validation.py — Dossier Edition

Post-assembly validation:
- Paragraph-level coverage scoring (unchanged)
- Source balance analysis — no single source should dominate
- Duplication ratio — catch paraphrased duplicates across the dossier
- Loss reports with dossier-specific diagnostics

Public API:
    compute_coverage(...)         — paragraph embedding coverage
    compute_source_balance(...)   — source share per label
    compute_duplication_ratio(...) — fraction of near-duplicate sections
    generate_loss_report(...)     — Markdown report (now includes balance + duplication)
"""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .embeddings import EmbeddingStore
from .utils import split_paragraphs


# ------------------------------------------------------------------
# 1. Paragraph-level coverage (unchanged from original)
# ------------------------------------------------------------------

def compute_coverage(
    original_text: str,
    output_text: str,
    embedding_store: EmbeddingStore,
    threshold: float = 0.85,
) -> dict:
    """
    Compute paragraph-level coverage of the original text in the output.

    For every paragraph in the original, we find the most similar paragraph
    in the output (by cosine similarity of embeddings).  A paragraph is
    "covered" if its best match exceeds *threshold*.

    Returns a dict with:
        coverage_score        – float in [0, 1]
        total_paragraphs      – int
        covered_paragraphs    – int
        uncovered_paragraphs  – list of (paragraph_text, best_similarity)
        paragraph_details     – list of (paragraph_text, best_similarity, matched)
    """
    orig_paragraphs = split_paragraphs(original_text)
    output_paragraphs = split_paragraphs(output_text)

    if not orig_paragraphs:
        return {
            "coverage_score": 1.0,
            "total_paragraphs": 0,
            "covered_paragraphs": 0,
            "uncovered_paragraphs": [],
            "paragraph_details": [],
        }

    # Embed all output paragraphs
    output_embeddings: List[np.ndarray] = []
    for p in output_paragraphs:
        emb = embedding_store.get_embedding(p)
        if emb is not None:
            output_embeddings.append(emb)

    if not output_embeddings:
        return {
            "coverage_score": 0.0,
            "total_paragraphs": len(orig_paragraphs),
            "covered_paragraphs": 0,
            "uncovered_paragraphs": [(p, 0.0) for p in orig_paragraphs],
            "paragraph_details": [(p, 0.0, False) for p in orig_paragraphs],
        }

    output_matrix = np.vstack(output_embeddings).astype(np.float32)

    covered = 0
    uncovered: List[Tuple[str, float]] = []
    details: List[Tuple[str, float, bool]] = []

    for para in orig_paragraphs:
        para_emb = embedding_store.get_embedding(para)
        if para_emb is None:
            uncovered.append((para, 0.0))
            details.append((para, 0.0, False))
            continue

        # Cosine similarity against all output paragraphs
        para_vec = para_emb.reshape(1, -1)
        norms_out = np.linalg.norm(output_matrix, axis=1) + 1e-9
        norm_para = float(np.linalg.norm(para_vec)) + 1e-9
        similarities = (output_matrix @ para_vec.T).flatten() / (norms_out * norm_para)

        best_sim = float(np.max(similarities))
        matched = best_sim >= threshold

        if matched:
            covered += 1
        else:
            uncovered.append((para, best_sim))

        details.append((para, best_sim, matched))

    score = covered / len(orig_paragraphs) if orig_paragraphs else 1.0

    return {
        "coverage_score": score,
        "total_paragraphs": len(orig_paragraphs),
        "covered_paragraphs": covered,
        "uncovered_paragraphs": uncovered,
        "paragraph_details": details,
    }


# ------------------------------------------------------------------
# 2. Source balance analysis (NEW)
# ------------------------------------------------------------------

def compute_source_balance(
    sections: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute the character-share of each source in the organized sections.

    Returns a dict mapping source_label -> share (0.0 – 1.0).
    """
    char_counter: Counter = Counter()
    for sec in sections:
        label = sec.get("source_label", "unknown") or "unknown"
        char_counter[label] += len(sec.get("text", ""))

    total = sum(char_counter.values()) or 1
    return {label: count / total for label, count in char_counter.items()}


# ------------------------------------------------------------------
# 3. Duplication ratio (NEW)
# ------------------------------------------------------------------

def compute_duplication_ratio(
    sections: List[Dict[str, Any]],
    embedding_store: EmbeddingStore,
    threshold: float = 0.90,
) -> float:
    """
    Compute the fraction of sections that are near-duplicates of an earlier section.

    Uses cosine similarity of section text embeddings.
    Returns a float in [0.0, 1.0].
    """
    if len(sections) < 2:
        return 0.0

    embeddings: List[np.ndarray] = []
    dup_count = 0
    total = 0

    for sec in sections:
        text = (sec.get("text") or "").strip()
        if not text:
            continue

        emb = embedding_store.get_embedding(text)
        if emb is None:
            total += 1
            continue

        is_dup = False
        for prev_emb in embeddings:
            sim = float(emb.dot(prev_emb))
            if sim >= threshold:
                is_dup = True
                break

        if is_dup:
            dup_count += 1

        embeddings.append(emb)
        total += 1

    return dup_count / total if total else 0.0


# ------------------------------------------------------------------
# 4. Loss report generation (extended)
# ------------------------------------------------------------------

def generate_loss_report(
    coverage_result: dict,
    source_balance: Optional[Dict[str, float]] = None,
    duplication_ratio: Optional[float] = None,
    max_source_share: float = 0.40,
    max_duplication_ratio: float = 0.15,
) -> str:
    """
    Generate a Markdown report covering:
    - Paragraph-level coverage
    - Source balance diagnostics
    - Duplication ratio
    """
    score = coverage_result["coverage_score"]
    total = coverage_result["total_paragraphs"]
    covered = coverage_result["covered_paragraphs"]
    uncovered = coverage_result["uncovered_paragraphs"]

    lines = [
        "# Dossier Quality Report",
        "",
        "## Coverage",
        "",
        f"**Coverage Score:** {score:.1%}",
        f"**Paragraphs Covered:** {covered} / {total}",
        "",
    ]

    if score >= 0.95:
        lines.append("✅ Excellent coverage. Minimal content loss detected.")
    elif score >= 0.85:
        lines.append(
            "⚠️  Good coverage, but some content may have been lost. "
            "Review uncovered paragraphs below."
        )
    else:
        lines.append(
            "❌ Significant content loss detected. "
            "Review and address uncovered paragraphs."
        )

    # --- Source Balance Section ---
    if source_balance:
        lines.extend(["", "## Source Balance", ""])
        lines.append("| Source | Share |  |")
        lines.append("|--------|-------|--|")
        for label, share in sorted(source_balance.items(), key=lambda x: -x[1]):
            flag = "⚠️ OVERWEIGHT" if share > max_source_share else "✅"
            lines.append(f"| {label} | {share:.1%} | {flag} |")
        lines.append("")
        overweight = [l for l, s in source_balance.items() if s > max_source_share]
        if overweight:
            lines.append(
                f"⚠️  {len(overweight)} source(s) exceed the {max_source_share:.0%} share limit: "
                f"{', '.join(overweight)}. The dossier may be imbalanced."
            )
        else:
            lines.append("✅ All sources are within balance limits.")

    # --- Duplication Ratio Section ---
    if duplication_ratio is not None:
        lines.extend(["", "## Duplication", ""])
        lines.append(f"**Duplication Ratio:** {duplication_ratio:.1%}")
        if duplication_ratio > max_duplication_ratio:
            lines.append(
                f"⚠️  Exceeds the {max_duplication_ratio:.0%} maximum. Some sections may contain "
                "near-identical content from different passes."
            )
        else:
            lines.append("✅ Duplication within acceptable limits.")

    # --- Uncovered Paragraphs ---
    if uncovered:
        lines.extend(["", "## Uncovered Paragraphs", ""])
        lines.append(
            "The following original paragraphs had no close match "
            "(similarity < threshold) in the output:"
        )
        lines.append("")

        for i, (para, sim) in enumerate(uncovered, 1):
            preview = para[:300] + "…" if len(para) > 300 else para
            lines.append(f"### {i}. (best similarity: {sim:.3f})")
            lines.append("")
            lines.append(f"> {preview}")
            lines.append("")

    return "\n".join(lines)
