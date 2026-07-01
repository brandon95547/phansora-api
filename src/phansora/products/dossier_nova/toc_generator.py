# toc_generator.py

"""
toc_generator.py — Dossier Edition

Builds a dossier-style Table of Contents from source material using DeepSeek
+ local embeddings.  The output is an investigative dossier structure, NOT
a simple topic outline.

Key difference from the old merger-style TOC:
- The TOC is organized into investigative dossier sections (Subject Overview,
  Timeline, Key Evidence, Institutional Findings, Interpretive Frameworks,
  Policy Significance, Source Perspectives, Open Questions, etc.)
- Source profiles inform which sections get created.
- Topic headings from the text are placed UNDER the dossier sections, not
  at the top level.

Still writes ONLY toc/full.md with stable hierarchical IDs.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import re

from .source_profiler import SourceProfile
from .utils import split_text_into_chunks, split_paragraphs
from .embeddings import EmbeddingStore


# --- Standard dossier sections (used as scaffolding) ---
DOSSIER_SECTIONS = [
    ("Subject Overview", "Background, context, and scope of the subject under investigation"),
    ("Timeline of Events", "Chronological sequence of key events, dates, and developments"),
    ("Key Evidence and Factual Findings", "Primary facts, data points, documented evidence, and verified information"),
    ("Institutional and Legal Findings", "Official findings, rulings, organizational actions, and regulatory outcomes"),
    ("Interpretive Frameworks", "Analytical perspectives, theories, scholarly interpretations, and competing explanations"),
    ("Policy and Social Significance", "Broader implications, advocacy positions, policy recommendations, and societal impact"),
    ("Source Perspectives", "Source-by-source perspective notes highlighting each document's unique contribution"),
    ("Open Questions and Contradictions", "Unresolved issues, tensions between sources, gaps in evidence, and areas needing further investigation"),
]


class TocGenerator:
    def __init__(self, config, embedding_store: EmbeddingStore):
        self.config = config
        self.embedding_store = embedding_store
        self.client = config.deepseek_client
        self.similarity_threshold = float(config.similarity_threshold)

    # ------------------------------------------------------------------
    # Step 1: Extract topic headings from the raw text (unchanged logic)
    # ------------------------------------------------------------------

    def extract_headings(
        self,
        full_text: str,
        max_chars_per_chunk: int = 10_000,
    ) -> List[str]:
        """
        Use DeepSeek to extract candidate headings from each text chunk,
        dedupe using cosine similarity, return a unique flat list.
        """
        chunks = split_text_into_chunks(full_text, max_chars=max_chars_per_chunk)
        target_heading_count = getattr(self.config, "toc_target_heading_count", 60)
        print(f"[TOC] Split text into {len(chunks)} chunks for heading extraction (target: {target_heading_count} headings).")

        seen_embeddings: List[np.ndarray] = []
        heading_texts: Set[str] = set()
        toc_sections: List[str] = []

        toc_prompt_template = (
            "You are extracting candidate section headings from a long non-fiction text.\n"
            "Given the following chunk, propose a short list of Markdown headings that "
            "describe the main topics.\n\n"
            "Rules:\n"
            "- Return ONLY Markdown headings, one per line.\n"
            "- Use only '##' or '###' at the start of each heading (sub-topic level).\n"
            "- No commentary or explanation, just headings.\n\n"
            "Chunk:\n"
            "{chunk}"
        )

        def _extract_from_chunk(idx_chunk):
            idx, chunk = idx_chunk
            print(f"[TOC] Processing chunk {idx}/{len(chunks)} for headings...")
            prompt = toc_prompt_template.format(chunk=chunk)
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            return idx, response.choices[0].message.content

        with ThreadPoolExecutor(max_workers=8) as executor:
            raw_results = list(executor.map(_extract_from_chunk, enumerate(chunks, start=1)))

        for _idx, result in raw_results:
            if not result:
                continue
            for line in result.splitlines():
                line = line.strip()
                if not line:
                    continue
                m = re.match(r"^(#{1,3})\s+(.*)$", line)
                if not m:
                    continue
                hashes, title = m.groups()
                title = title.strip()
                heading_text = f"{hashes} {title}"
                emb = self.embedding_store.get_embedding(heading_text)
                if emb is None:
                    continue
                if any(
                    self._cosine_similarity(emb, prev) >= self.similarity_threshold
                    for prev in seen_embeddings
                ):
                    continue
                if heading_text not in heading_texts:
                    heading_texts.add(heading_text)
                    seen_embeddings.append(emb)
                    toc_sections.append(heading_text)

        print(f"[TOC] Collected {len(toc_sections)} unique candidate sub-headings.")
        return toc_sections

    # ------------------------------------------------------------------
    # Step 2: Build dossier-style TOC
    # ------------------------------------------------------------------

    def build_dossier_toc(
        self,
        topic_headings: List[str],
        source_profiles: Optional[List[SourceProfile]] = None,
    ) -> str:
        """
        Build a dossier-structured TOC by:
        1. Starting with standard dossier sections as top-level (#) headings
        2. Asking DeepSeek to place extracted topic headings under the
           appropriate dossier sections as sub-headings (##/###)
        3. Adding source-specific perspective sections if profiles exist
        4. Deduplicating and assigning stable IDs
        """
        # Build the dossier scaffold
        scaffold_lines = []
        for section_name, _desc in DOSSIER_SECTIONS:
            scaffold_lines.append(f"# {section_name}")

        scaffold = "\n".join(scaffold_lines)

        # Ask DeepSeek to place topic headings under dossier sections
        topic_block = "\n".join(f"- {h}" for h in topic_headings) if topic_headings else "(no topic headings extracted)"

        # Build source profile context for the LLM
        profile_context = ""
        if source_profiles:
            profile_lines = []
            for p in source_profiles:
                profile_lines.append(
                    f"- {p.source_label}: type={p.source_type}, role={p.rhetorical_role}"
                )
                if p.central_argument:
                    profile_lines.append(f"  Thesis: {p.central_argument[:200]}")
            profile_context = (
                "\n\nSOURCE PROFILES:\n" + "\n".join(profile_lines) +
                "\n\nUse these profiles to decide which dossier sections need more "
                "sub-headings and which sources contribute to which sections."
            )

        organize_prompt = (
            "You are building the Table of Contents for an investigative dossier.\n\n"
            "DOSSIER STRUCTURE (top-level sections — keep ALL of these as # headings):\n\n"
            f"{scaffold}\n\n"
            "EXTRACTED TOPIC HEADINGS from the source material:\n\n"
            f"{topic_block}\n\n"
            f"{profile_context}\n\n"
            "TASK:\n"
            "1. Keep every # (top-level) dossier section exactly as listed above.\n"
            "2. Place each extracted topic heading under the MOST appropriate dossier section "
            "   as a ## or ### sub-heading.\n"
            "3. If a topic heading doesn't fit any section, place it under the closest match.\n"
            "4. You may add a few additional sub-headings if the source material clearly "
            "   warrants them, but do NOT invent content — only create structural headings.\n"
            "5. Under 'Source Perspectives', create one ## sub-heading per source document "
            "   (using the source filenames from the profiles above).\n"
            "6. Return ONLY Markdown headings (#, ##, ###). No bullets, no prose, no commentary.\n"
            "7. Do NOT change the wording of the top-level dossier sections.\n"
        )

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant building a dossier TOC."},
                {"role": "user", "content": organize_prompt},
            ],
            temperature=0.2,
            max_tokens=2048,
        )

        toc_md_raw = (response.choices[0].message.content or "").strip()
        if not toc_md_raw:
            raise ValueError("[TOC] DeepSeek returned empty dossier TOC.")

        # Parse, dedupe, and assign IDs
        final_headings: List[Tuple[str, str]] = []
        seen_norms: Set[str] = set()
        seen_embs: List[np.ndarray] = []

        for line in toc_md_raw.splitlines():
            line = line.rstrip()
            if not line:
                continue
            m = re.match(r"^(#{1,6})\s+(.*\S.*)$", line)
            if not m:
                continue
            hashes, title = m.groups()
            title = title.strip()
            heading_text_for_emb = f"{hashes} {title}"
            norm = self._normalize_heading_text(title)
            if norm in seen_norms:
                continue
            emb = self.embedding_store.get_embedding(heading_text_for_emb)
            if emb is None:
                continue
            if any(
                self._cosine_similarity(emb, prev) >= self.similarity_threshold
                for prev in seen_embs
            ):
                continue
            seen_norms.add(norm)
            seen_embs.append(emb)
            final_headings.append((hashes, title))

        print(f"[TOC] Dossier TOC: {len(final_headings)} headings after dedupe.")

        # Assign hierarchical IDs
        return self._assign_ids_and_format(final_headings)

    # ------------------------------------------------------------------
    # Legacy method: build_toc_markdown (topic-only, kept for compatibility)
    # ------------------------------------------------------------------

    def build_toc_markdown(self, toc_sections: List[str]) -> str:
        """Legacy: builds a topic-only TOC. Use build_dossier_toc() instead."""
        if not toc_sections:
            raise ValueError("[TOC] No headings extracted; cannot build TOC.")

        headings_block = "\n".join(f"- {h}" for h in toc_sections)

        final_prompt = (
            "You are organizing a non-fiction book's Table of Contents.\n"
            "Given the following extracted headings, rewrite them into a clean "
            "Markdown table of contents structure.\n\n"
            "Rules:\n"
            "- Use Markdown headings only (#, ##, ###).\n"
            "- Do NOT change the wording of any heading text, only group and order.\n"
            "- Do NOT add bullet lists or explanatory paragraphs.\n"
            "- No commentary, only Markdown headings.\n\n"
            "Extracted headings:\n\n"
            f"{headings_block}"
        )

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": final_prompt},
            ],
            temperature=0.2,
            max_tokens=2048,
        )

        toc_md_raw = (response.choices[0].message.content or "").strip()
        if not toc_md_raw:
            raise ValueError("[TOC] DeepSeek returned empty TOC markdown.")

        final_headings: List[Tuple[str, str]] = []
        seen_norms: Set[str] = set()
        seen_embs: List[np.ndarray] = []

        for line in toc_md_raw.splitlines():
            line = line.rstrip()
            if not line:
                continue
            m = re.match(r"^(#{1,6})\s+(.*\S.*)$", line)
            if not m:
                continue
            hashes, title = m.groups()
            title = title.strip()
            heading_text_for_emb = f"{hashes} {title}"
            norm = self._normalize_heading_text(title)
            if norm in seen_norms:
                continue
            emb = self.embedding_store.get_embedding(heading_text_for_emb)
            if emb is None:
                continue
            if any(
                self._cosine_similarity(emb, prev) >= self.similarity_threshold
                for prev in seen_embs
            ):
                continue
            seen_norms.add(norm)
            seen_embs.append(emb)
            final_headings.append((hashes, title))

        print(f"[TOC] After organization+dedupe: {len(final_headings)} headings.")
        return self._assign_ids_and_format(final_headings)

    # ------------------------------------------------------------------
    # Step 3: Write toc/full.md
    # ------------------------------------------------------------------

    def write_toc_files(self, toc_markdown: str, toc_full_path: str = "toc/full.md") -> None:
        toc_path = Path(toc_full_path)
        toc_path.parent.mkdir(parents=True, exist_ok=True)
        toc_path.write_text(toc_markdown, encoding="utf-8")
        print(f"[TOC] Wrote TOC to {toc_path}")

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def generate_from_text(self, full_text: str, toc_full_path: str = "toc/full.md") -> None:
        """Legacy entry point: generates topic-only TOC from merged text."""
        toc_sections = self.extract_headings(full_text)
        toc_markdown = self.build_toc_markdown(toc_sections)
        self.write_toc_files(toc_markdown, toc_full_path)

    def generate_dossier_from_sources(
        self,
        full_text: str,
        source_profiles: Optional[List[SourceProfile]] = None,
        toc_full_path: str = "toc/full.md",
    ) -> None:
        """
        Dossier entry point: generates an investigative dossier TOC using
        source profiles to inform section structure.
        """
        topic_headings = self.extract_headings(full_text)
        toc_markdown = self.build_dossier_toc(topic_headings, source_profiles)
        self.write_toc_files(toc_markdown, toc_full_path)

    # ------------------------------------------------------------------
    # TOC completeness pre-check
    # ------------------------------------------------------------------

    def check_toc_completeness(
        self,
        full_text: str,
        headings: List[Tuple[str, str]],
        threshold: float = 0.35,
    ) -> Dict:
        """
        Verify that every paragraph in the original text has at least one
        plausible heading in the TOC (by embedding similarity).
        """
        paragraphs = split_paragraphs(full_text)
        if not paragraphs or not headings:
            return {
                "total_paragraphs": len(paragraphs),
                "covered": len(paragraphs),
                "uncovered": [],
                "coverage_ratio": 1.0,
            }

        heading_embs: List[Tuple[str, str, Optional[np.ndarray]]] = []
        for hid, title in headings:
            emb = self.embedding_store.get_embedding(title)
            heading_embs.append((hid, title, emb))

        covered = 0
        uncovered: List[Tuple[str, str, float]] = []

        for para in paragraphs:
            para_emb = self.embedding_store.get_embedding(para)
            if para_emb is None:
                uncovered.append((para, "", 0.0))
                continue
            best_id = ""
            best_sim = -1.0
            for hid, title, h_emb in heading_embs:
                if h_emb is None:
                    continue
                sim = float(self._cosine_similarity(para_emb, h_emb))
                if sim > best_sim:
                    best_sim = sim
                    best_id = hid
            if best_sim >= threshold:
                covered += 1
            else:
                uncovered.append((para, best_id, best_sim))

        ratio = covered / len(paragraphs) if paragraphs else 1.0

        if uncovered:
            print(
                f"[TOC] Completeness pre-check: {covered}/{len(paragraphs)} paragraphs "
                f"have a plausible heading (threshold={threshold:.2f})."
            )
            print(
                f"[TOC] ⚠️  {len(uncovered)} paragraph(s) may have no structural home "
                f"in the current TOC."
            )
        else:
            print(f"[TOC] ✅ Completeness pre-check: all {len(paragraphs)} paragraphs have a plausible heading.")

        return {
            "total_paragraphs": len(paragraphs),
            "covered": covered,
            "uncovered": uncovered,
            "coverage_ratio": ratio,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _assign_ids_and_format(self, final_headings: List[Tuple[str, str]]) -> str:
        """Assign hierarchical IDs [H1], [H1.1], etc. and build final markdown."""
        toc_with_ids_lines: List[str] = []
        h1_idx = 0
        h2_idx = 0
        h3_idx = 0

        for hashes, title in final_headings:
            level = len(hashes)
            if level == 1:
                h1_idx += 1
                h2_idx = 0
                h3_idx = 0
                hid = f"H{h1_idx}"
            elif level == 2:
                if h1_idx == 0:
                    h1_idx = 1
                h2_idx += 1
                h3_idx = 0
                hid = f"H{h1_idx}.{h2_idx}"
            elif level == 3:
                if h1_idx == 0:
                    h1_idx = 1
                if h2_idx == 0:
                    h2_idx = 1
                h3_idx += 1
                hid = f"H{h1_idx}.{h2_idx}.{h3_idx}"
            else:
                if h1_idx == 0:
                    h1_idx = 1
                if h2_idx == 0:
                    h2_idx = 1
                h3_idx += 1
                hid = f"H{h1_idx}.{h2_idx}.{h3_idx}"

            line_with_id = f"{hashes} [{hid}] {title}"
            toc_with_ids_lines.append(line_with_id)

        return "\n".join(toc_with_ids_lines).strip() + "\n"

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(a.dot(b) / denom)

    @staticmethod
    def _normalize_heading_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"\[[^\]]+\]", " ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        stopwords = {
            "the", "a", "an", "of", "about", "this", "that", "in", "on",
            "for", "to", "and", "with", "from", "into", "introduction",
            "chapter", "section", "part",
        }
        tokens = [t for t in tokens if t not in stopwords]
        return " ".join(tokens)
