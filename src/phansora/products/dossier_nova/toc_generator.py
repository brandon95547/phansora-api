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
from phansora.shared.ai.deepseek import chat_model


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
                model=chat_model("DOSSIER_MODEL"),
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
    # LLM plumbing for the organize calls
    # ------------------------------------------------------------------

    def _organize_max_tokens(self, heading_count: int) -> int:
        """Token budget for a call that must emit the whole TOC in one answer.

        The old value was a hardcoded 2048 for every dossier regardless of size, but this
        answer scales with the source material: one line per extracted heading plus the
        dossier scaffold. ~24 tokens/line is a generous allowance for "### Some Heading
        Text". Floor of 4096 so a small dossier still has room to think.
        """
        cap = int(getattr(self.config, "toc_organize_max_tokens", 16384) or 16384)
        want = 1024 + max(0, heading_count) * 24
        return max(4096, min(want, cap))

    @staticmethod
    def _describe_response(response) -> str:
        """Everything worth knowing about a response that came back unusable.

        The previous failure mode was `raise ValueError("empty dossier TOC")` and nothing
        else, which cannot distinguish "budget exhausted by reasoning" from "model refused"
        from "answer was whitespace" — so the reported error told you only that it broke.
        """
        bits = []
        try:
            choice = response.choices[0]
            bits.append(f"finish_reason={getattr(choice, 'finish_reason', None)}")
            msg = getattr(choice, "message", None)
            content = getattr(msg, "content", None)
            bits.append(f"content_len={len(content or '')}")
            # deepseek-v4-flash reasons on every tier; chain-of-thought arrives in its own
            # field. Non-empty reasoning with empty content is the signature of a budget
            # spent thinking before any answer token was emitted.
            reasoning = getattr(msg, "reasoning_content", None)
            if reasoning:
                bits.append(f"reasoning_len={len(reasoning)}")
            refusal = getattr(msg, "refusal", None)
            if refusal:
                bits.append(f"refusal={refusal!r}")
        except Exception as exc:  # noqa: BLE001 - diagnostics must never mask the real error
            bits.append(f"choice_unreadable={exc!r}")
        usage = getattr(response, "usage", None)
        if usage is not None:
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                val = getattr(usage, field, None)
                if val is not None:
                    bits.append(f"{field}={val}")
            details = getattr(usage, "completion_tokens_details", None)
            reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None
            if reasoning_tokens is not None:
                bits.append(f"reasoning_tokens={reasoning_tokens}")
        return ", ".join(bits)

    def _organize_call(self, system: str, prompt: str, heading_count: int, label: str) -> str:
        """One organize call, with a retry at double the budget and loud diagnostics.

        Returns "" rather than raising: the caller has the extracted headings and the fixed
        scaffold in hand, so it can still assemble a TOC without the model. Killing a
        pipeline that has already paid for chunk extraction is the worse outcome.
        """
        budget = self._organize_max_tokens(heading_count)
        for attempt in (1, 2):
            response = self.client.chat.completions.create(
                model=chat_model("DOSSIER_MODEL"),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=budget,
            )
            content = (response.choices[0].message.content or "").strip()
            if content:
                return content
            detail = self._describe_response(response)
            print(
                f"[TOC] {label}: empty answer on attempt {attempt} "
                f"(max_tokens={budget}, headings={heading_count}, prompt_chars={len(prompt)}) "
                f"-> {detail}"
            )
            if attempt == 1:
                budget = min(budget * 2, int(getattr(self.config, "toc_organize_max_tokens", 16384) or 16384))
                print(f"[TOC] {label}: retrying with max_tokens={budget}.")
        return ""

    def _place_headings_by_similarity(
        self,
        topic_headings: List[str],
        source_profiles: Optional[List[SourceProfile]] = None,
    ) -> List[Tuple[str, str]]:
        """Assemble the dossier TOC without the model.

        Used when the organize call comes back empty. This is not a stub: DOSSIER_SECTIONS
        carries a description per section and an embedding store is already wired up for
        dedupe, so each heading can be assigned to its nearest section by cosine similarity.
        Headings whose embedding is unavailable land under the configured catch-all rather
        than being dropped.
        """
        section_embs: List[Tuple[str, Optional[np.ndarray]]] = []
        for name, desc in DOSSIER_SECTIONS:
            section_embs.append((name, self.embedding_store.get_embedding(f"{name}. {desc}")))

        catchall = getattr(self.config, "catchall_heading", "Miscellaneous") or "Miscellaneous"
        buckets: Dict[str, List[str]] = {name: [] for name, _ in DOSSIER_SECTIONS}
        overflow: List[str] = []

        for heading in topic_headings:
            title = re.sub(r"^#{1,6}\s+", "", heading).strip()
            if not title:
                continue
            emb = self.embedding_store.get_embedding(title)
            best_name, best_score = None, -1.0
            if emb is not None:
                for name, s_emb in section_embs:
                    if s_emb is None:
                        continue
                    score = self._cosine_similarity(emb, s_emb)
                    if score > best_score:
                        best_name, best_score = name, score
            if best_name is None:
                overflow.append(title)
            else:
                buckets[best_name].append(title)

        out: List[Tuple[str, str]] = []
        for name, _desc in DOSSIER_SECTIONS:
            out.append(("#", name))
            if name == "Source Perspectives":
                for p in source_profiles or []:
                    out.append(("##", p.source_label))
            for title in buckets[name]:
                out.append(("##", title))
        if overflow:
            out.append(("#", catchall))
            for title in overflow:
                out.append(("##", title))

        placed = sum(len(v) for v in buckets.values()) + len(overflow)
        print(
            f"[TOC] Deterministic placement: {placed} heading(s) assigned by embedding "
            f"similarity across {len(DOSSIER_SECTIONS)} sections"
            + (f", {len(overflow)} under '{catchall}'" if overflow else "")
            + "."
        )
        return out

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

        toc_md_raw = self._organize_call(
            "You are a helpful assistant building a dossier TOC.",
            organize_prompt,
            len(topic_headings),
            "dossier organize",
        )
        if not toc_md_raw:
            # Was: raise ValueError("[TOC] DeepSeek returned empty dossier TOC."), which
            # threw away a pipeline that had already paid for chunk extraction. Every
            # ingredient for a valid TOC is in hand here — the fixed DOSSIER_SECTIONS
            # scaffold and the deduped headings — so build it locally instead.
            print("[TOC] Organize call produced nothing; assembling the TOC locally.")
            return self._assign_ids_and_format(
                self._place_headings_by_similarity(topic_headings, source_profiles)
            )

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

        # A non-empty answer that parsed to nothing usable (prose, bullets, fenced code —
        # anything with no "#" lines) is the same outcome as an empty one, so it takes the
        # same route rather than returning a TOC with zero headings downstream.
        if not final_headings:
            print(
                f"[TOC] Organize answer had no usable headings "
                f"({len(toc_md_raw)} chars returned); assembling the TOC locally."
            )
            return self._assign_ids_and_format(
                self._place_headings_by_similarity(topic_headings, source_profiles)
            )

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

        # Same scaling + diagnostics as the dossier path: this answer is also "the whole
        # TOC in one response", so a fixed 2048 under-budgets it for exactly the same reason.
        toc_md_raw = self._organize_call(
            "You are a helpful assistant.",
            final_prompt,
            len(toc_sections),
            "legacy organize",
        )
        if not toc_md_raw:
            raise ValueError(
                "[TOC] DeepSeek returned empty TOC markdown after a retry at double the "
                "token budget. See the [TOC] diagnostics above for finish_reason and usage."
            )

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
