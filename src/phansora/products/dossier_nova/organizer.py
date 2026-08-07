"""
organizer.py — Dossier Edition

Coordinates:
- Building dossier-aware LLM prompts using the TOC (with stable heading IDs)
- Sending source-labeled chunks to DeepSeek for dossier synthesis
- Parsing structured JSON output into heading-based sections WITH source attribution
- Using EmbeddingStore + TocManager to dedupe and insert blocks
- Enforcing source balance so no single source dominates

Key changes from the merger-style organizer:
- Prompt instructs the LLM to SYNTHESIZE, not just classify verbatim
- Each section carries a source_label for attribution tracking
- Claim-level deduplication prevents paraphrased duplicates
- Source balance scoring rejects output that overrepresents one source
- Evidence is separated from interpretation in the output

Main public API:

    - build_dossier_prompt_template(...)
    - DossierOrganizer(...).organize_chunks(chunks)
    - DossierOrganizer(...).insert_sections(sections)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from .toc_manager import TocManager
from .embeddings import EmbeddingStore
from .source_profiler import SourceProfile
from .provenance import ground_passage
from phansora.shared.ai.deepseek import chat_model


# Subheading labels that are internal placeholders, not real dossier structure.
# Stored in normalized form (see _normalize_label).
_PLACEHOLDER_SUBHEADINGS = {
    "auto placed review recommended",
}


def _normalize_label(text: str) -> str:
    """
    Normalize a heading/title/short body for equality comparison: lowercase,
    strip, and reduce to alphanumeric tokens. So "The Idea", "The idea.", and
    "the  idea" all compare equal, and minor punctuation/quote differences in
    otherwise-identical paragraphs collapse together.
    """
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def build_prompt_template(_: Any = None) -> str:
    """
    Build the dossier synthesis prompt template.

    The template contains {toc_outline}, {chunk}, and {source_context}
    placeholders filled in later.
    """
    return (
        "You are a dossier archivist. You are NOT a writer, an analyst, or a summarizer. "
        "Your only job is to decide WHERE each passage of the source chunk belongs in the "
        "dossier, and to quote that passage back EXACTLY as it appears.\n\n"
        "DOSSIER TABLE OF CONTENTS (IDs + headings):\n"
        "{toc_outline}\n"
        "END OF TOC\n\n"
        "{source_context}\n"
        "SOURCE CHUNK:\n"
        "{chunk}\n"
        "END OF CHUNK\n\n"
        "TASK:\n"
        "1. Read every passage in the source chunk.\n"
        "2. For each passage, decide which dossier heading ID it belongs under.\n"
        "3. COPY the passage into \"text\" character for character, exactly as written "
        "in the chunk above.\n\n"
        "THE COPYING RULE — this is the whole job:\n"
        "Every \"text\" value MUST be findable in the chunk above by an exact text "
        "search. If someone searched the chunk for your \"text\", they must find it.\n"
        "You MUST NOT:\n"
        "  - Summarize, paraphrase, condense, or shorten\n"
        "  - Rewrite, reword, or 'improve' the phrasing\n"
        "  - Fix grammar, spelling, punctuation, or capitalization\n"
        "  - Add a single word that is not already in the chunk\n"
        "  - Merge two separate passages into one entry\n"
        "  - Add transitions, introductions, or closing sentences\n"
        "  - Add your own commentary, framing, characterization, or conclusions\n"
        "Copy whole sentences. Never start or stop mid-sentence.\n"
        "If a passage is long, quote all of it, or quote the specific whole sentences "
        "you are placing — never a rewritten shorter version of it.\n\n"
        "PLACEMENT:\n"
        "4. If a passage contains factual claims AND interpretation, quote them as two "
        "separate entries under the appropriate headings (e.g., facts under 'Key "
        "Evidence', interpretation under 'Interpretive Frameworks'). Split them; do not "
        "rewrite either half.\n"
        "5. Do NOT leave any meaningful source content unassigned.\n"
        "6. Do NOT place the same passage under multiple headings.\n"
        "7. source_label identifies which document the passage came from.\n\n"
        "OUTPUT FORMAT (VERY IMPORTANT):\n"
        "Return ONLY valid JSON with this exact structure:\n\n"
        "{{\n"
        "  \"sections\": [\n"
        "    {{\n"
        "      \"heading_id\": \"H2.1\",\n"
        "      \"subheading\": \"Optional short label, may be your own words\",\n"
        "      \"source_label\": \"filename.pdf\",\n"
        "      \"content_type\": \"evidence|analysis|interpretation|policy|narrative\",\n"
        "      \"text\": \"An exact, unaltered quotation from the chunk above\"\n"
        "    }}\n"
        "  ]\n"
        "}}\n\n"
        "Rules:\n"
        "- NEVER return {{\"sections\": []}}. Every chunk has content that belongs "
        "  somewhere.\n"
        "- source_label MUST match the source filename provided in the chunk header.\n"
        "- content_type labels the passage. It describes what the SOURCE is doing, and "
        "  never licenses you to write anything yourself:\n"
        "  - evidence: factual claims, data, documented events\n"
        "  - analysis: conclusions the source itself draws from evidence\n"
        "  - interpretation: scholarly/theoretical perspective the source offers\n"
        "  - policy: advocacy position or recommendation the source makes\n"
        "  - narrative: story, account, personal testimony\n"
        "- \"subheading\" is the ONLY field where your own wording is allowed, and it is "
        "  a label, not a comment — no evaluation, no conclusions.\n"
        "- Do NOT include any explanation outside the JSON.\n"
        "- Do NOT wrap the JSON in Markdown code fences.\n"
    )


# Legacy alias
build_dossier_prompt_template = build_prompt_template


class ChunkOrganizer:
    """
    Dossier-aware organizer that:
    - Sends source-labeled chunks to DeepSeek for dossier synthesis
    - Tracks source attribution per section
    - Enforces source balance and claim-level dedup
    - Inserts blocks under dossier headings with source annotations
    """

    def __init__(
        self,
        client: Any,
        toc: TocManager,
        embedding_store: EmbeddingStore,
        prompt_template: str,
        conservative_mode: bool = False,
        catchall_heading: str = "Miscellaneous",
        content_similarity_threshold: float = 0.95,
        source_profiles: Optional[List[SourceProfile]] = None,
        max_source_share: float = 0.40,
        claim_dedup_threshold: float = 0.90,
        source_only: bool = True,
    ) -> None:
        self.client = client
        self.toc = toc
        self.embedding_store = embedding_store
        self.prompt_template = prompt_template
        self.conservative_mode = conservative_mode
        # When true, nothing reaches the dossier that is not a literal span of a source.
        # See provenance.py — this is the switch between "the model wrote the dossier"
        # and "the model decided where the sources go".
        self.source_only = source_only
        self.catchall_heading = catchall_heading
        self.content_similarity_threshold = content_similarity_threshold
        self.source_profiles = source_profiles or []
        self.max_source_share = max_source_share
        self.claim_dedup_threshold = claim_dedup_threshold

        # Build a lookup of source profiles by label
        self._profile_by_label: Dict[str, SourceProfile] = {}
        for p in self.source_profiles:
            self._profile_by_label[p.source_label] = p

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def organize_chunks(self, chunks: List[str]) -> List[Dict[str, Any]]:
        """
        Organize source-labeled chunks into dossier sections.

        Each chunk may contain a source header line like:
            === SOURCE: filename.pdf ===

        Returns a flat list of section dicts with source_label attribution.
        """
        all_sections: List[Dict[str, Any]] = []
        unmapped_chunks: List[str] = []
        # Quotes the model produced that are not in the source at all, and quotes that
        # were found but had drifted (a smart quote, a reflowed line, a dropped word).
        # Both are reported: a rising `ungrounded` count is the signal that the model has
        # started writing again rather than copying.
        ungrounded = 0
        repaired = 0

        # Filter empty chunks
        valid_chunks: list[tuple[int, str]] = []
        for idx, chunk in enumerate(chunks, start=1):
            chunk = (chunk or "").strip()
            if chunk:
                valid_chunks.append((idx, chunk))

        # Parallel LLM calls
        def _classify_chunk(item: tuple[int, str]) -> tuple[int, str, str]:
            idx, chunk = item
            print(f"[ORG] Processing chunk {idx}/{len(chunks)} ({len(chunk)} chars)...")
            prompt = self._build_prompt(chunk)
            raw_content = self._call_llm(prompt)
            return idx, chunk, raw_content

        with ThreadPoolExecutor(max_workers=8) as executor:
            llm_results = list(executor.map(_classify_chunk, valid_chunks))

        for idx, chunk, raw_content in llm_results:
            if not raw_content:
                print(f"  -> Chunk {idx}: Empty LLM response, tracking for second pass.")
                unmapped_chunks.append(chunk)
                continue

            json_str = self._extract_json(raw_content)
            if not json_str:
                print(f"  -> Chunk {idx}: No JSON found, tracking for second pass.")
                unmapped_chunks.append(chunk)
                continue

            try:
                payload = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"  -> Chunk {idx}: JSON error: {e}, tracking for second pass.")
                unmapped_chunks.append(chunk)
                continue

            sections = payload.get("sections") or []
            if not isinstance(sections, list):
                print(f"  -> Chunk {idx}: 'sections' not a list, tracking for second pass.")
                unmapped_chunks.append(chunk)
                continue

            chunk_had_mappings = False
            # Detect source label from chunk header
            default_source = self._detect_source_label(chunk)

            for sec in sections:
                if not isinstance(sec, dict):
                    continue
                heading_id = (sec.get("heading_id") or "").strip()
                text = (sec.get("text") or "").strip()
                subheading = (sec.get("subheading") or "").strip()
                source_label = (sec.get("source_label") or default_source or "").strip()
                content_type = (sec.get("content_type") or "evidence").strip()

                if not heading_id or not text:
                    continue

                # The prompt asks for an exact quotation; this is what makes it true.
                # `text` is replaced by the span it was found at in the chunk, so the
                # dossier carries the SOURCE's characters and not the model's rendering
                # of them — and anything that cannot be found is dropped, because text
                # with no origin in the source is exactly what must never reach the page.
                if self.source_only:
                    grounded = ground_passage(text, chunk)
                    if grounded is None:
                        ungrounded += 1
                        continue
                    if _normalize_label(grounded) != _normalize_label(text):
                        repaired += 1
                    text = grounded

                chunk_had_mappings = True
                all_sections.append({
                    "heading_id": heading_id,
                    "text": text,
                    "subheading": subheading or None,
                    "source_label": source_label,
                    "content_type": content_type,
                })

            if not chunk_had_mappings:
                unmapped_chunks.append(chunk)

        print(f"[ORG] Pass 1: {len(all_sections)} sections; {len(unmapped_chunks)} unmapped chunks.")
        if self.source_only:
            print(
                f"[ORG] Verbatim check: {repaired} quote(s) snapped back to the source, "
                f"{ungrounded} dropped as not found in any source."
            )
            # A chunk whose every quote was dropped still has its content: it falls through
            # to pass 2, which places the raw chunk by embedding. Nothing is lost by being
            # strict here — it is only placed less precisely.

        # Pass 2: embedding-based fallback for unmapped content
        if unmapped_chunks:
            print(f"[ORG] Pass 2: Processing {len(unmapped_chunks)} unmapped chunks...")
            unmapped_sections = self._organize_unmapped_chunks(unmapped_chunks)
            all_sections.extend(unmapped_sections)
            print(f"[ORG] Pass 2 added {len(unmapped_sections)} sections.")

        # --- Dossier quality enforcement ---
        # Claim-level dedup across all sections
        if not self.conservative_mode:
            before = len(all_sections)
            all_sections = self._deduplicate_claims(all_sections)
            deduped = before - len(all_sections)
            if deduped > 0:
                print(f"[ORG] Claim-level dedup removed {deduped} duplicate sections.")

        # Source balance check
        self._log_source_balance(all_sections)

        return all_sections

    def _organize_unmapped_chunks(self, unmapped_chunks: List[str]) -> List[Dict[str, Any]]:
        """Assign unmapped content to nearest heading by embedding similarity."""
        sections: List[Dict[str, Any]] = []
        headings = self.toc.get_heading_titles()

        for chunk in unmapped_chunks:
            chunk = (chunk or "").strip()
            if not chunk:
                continue
            source_label = self._detect_source_label(chunk)
            chunk_emb = self.embedding_store.get_embedding(chunk)
            if chunk_emb is None:
                sections.append({
                    "heading_id": "CATCHALL",
                    "text": chunk,
                    "subheading": None,
                    "source_label": source_label or "",
                    "content_type": "evidence",
                })
                continue

            nearest_id = self.embedding_store.find_nearest_heading(chunk_emb, headings)
            if nearest_id:
                sections.append({
                    "heading_id": nearest_id,
                    "text": chunk,
                    "subheading": "Auto-placed (review recommended)",
                    "source_label": source_label or "",
                    "content_type": "evidence",
                })
            else:
                sections.append({
                    "heading_id": "CATCHALL",
                    "text": chunk,
                    "subheading": None,
                    "source_label": source_label or "",
                    "content_type": "evidence",
                })

        return sections

    def insert_sections(self, sections: List[Dict[str, Any]]) -> None:
        """
        Insert organized sections into toc/full.md with source attribution
        annotations, optionally deduplicating with FAISS.
        """
        conservative = self.conservative_mode
        catchall = self.catchall_heading
        inserted_count = 0
        skipped_count = 0
        total_chars = 0

        # Verbatim-duplicate guard — always on, even in conservative mode.
        # Conservative mode keeps paraphrases, but exact/near-exact repeats are
        # noise, not content, and make the dossier look unprofessional.
        seen_texts: set[str] = set()
        # Last subheading actually emitted under each heading, so we collapse
        # runs of blocks that share the same subheading into a single heading.
        last_subheading_by_heading: Dict[str, str] = {}

        for sec in sections:
            heading_id = sec["heading_id"]
            text = sec["text"]
            subheading = sec.get("subheading")
            source_label = sec.get("source_label", "")
            content_type = sec.get("content_type", "")

            # Title of the dossier heading this block lives under, used to drop
            # subheadings/bodies that merely echo it.
            if heading_id == "CATCHALL":
                parent_title = catchall
            else:
                parent_title = self.toc.get_title_for_id(heading_id) or ""
            parent_norm = _normalize_label(parent_title)

            text_norm = _normalize_label(text)

            # Drop verbatim / near-verbatim duplicate blocks.
            if text_norm and text_norm in seen_texts:
                skipped_count += 1
                continue

            # Drop stray blocks whose body is just an echo of a heading title
            # (e.g. a one-line "Laws and standards" under the "Laws and
            # standards" heading) — these carry no real content.
            if text_norm and text_norm == parent_norm:
                skipped_count += 1
                continue

            if text_norm:
                seen_texts.add(text_norm)

            # Decide whether the subheading is worth rendering as a `####`.
            sub_norm = _normalize_label(subheading or "")
            render_subheading = bool(sub_norm)
            if render_subheading and sub_norm in _PLACEHOLDER_SUBHEADINGS:
                render_subheading = False
            if render_subheading and sub_norm == parent_norm:
                # Subheading just repeats the parent section title.
                render_subheading = False
            if render_subheading and sub_norm == text_norm:
                # Subheading is identical to the body it labels.
                render_subheading = False
            if render_subheading and sub_norm == last_subheading_by_heading.get(heading_id):
                # Same subheading as the previous block under this heading —
                # group them under the one heading already emitted.
                render_subheading = False

            # Build the markdown block with source attribution
            block_lines: List[str] = []

            if render_subheading:
                block_lines.append(f"#### {subheading}\n\n")

            # Add source attribution as a subtle annotation
            if source_label:
                content_tag = f" · {content_type}" if content_type else ""
                block_lines.append(f"*[Source: {source_label}{content_tag}]*\n\n")

            block_lines.append(text.strip() + "\n\n")
            # Lead with a blank line so the block is cleanly separated from the
            # preceding heading or block (TOC heading lines have no trailing
            # blank line of their own).
            block = "\n" + "".join(block_lines)
            block_chars = len(block)

            def _commit() -> None:
                # Record bookkeeping only when the block is actually inserted,
                # so a later-skipped block doesn't suppress the next heading.
                nonlocal total_chars, inserted_count
                total_chars += block_chars
                inserted_count += 1
                if render_subheading:
                    last_subheading_by_heading[heading_id] = sub_norm

            if heading_id == "CATCHALL":
                self.toc.insert_block_under_heading_id(catchall, block)
                _commit()
                continue

            if conservative:
                self.toc.insert_block_under_heading_id(heading_id, block)
                _commit()
            else:
                emb = self.embedding_store.get_embedding(block)
                if emb is None:
                    print(f"-> Skipping block for {heading_id} due to embedding error.")
                    skipped_count += 1
                    continue

                if self.embedding_store.is_duplicate(emb, threshold=self.content_similarity_threshold):
                    print(f"-> Skipping block for {heading_id} (FAISS duplicate).")
                    skipped_count += 1
                    continue

                self.toc.insert_block_under_heading_id(heading_id, block)
                self.embedding_store.add_block(block, emb)
                _commit()

        self.toc.save()

        mode_info = "(conservative: no dedup)" if conservative else "(dedup enabled)"
        print(f"[ORG] Finished inserting {inserted_count} sections into toc/full.md {mode_info}.")
        if skipped_count > 0:
            print(f"[ORG] Skipped {skipped_count} duplicate blocks.")
        print(f"[ORG] Total content inserted: {total_chars} characters.")

    # ------------------------------------------------------------------
    # Dossier quality enforcement
    # ------------------------------------------------------------------

    def _deduplicate_claims(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove sections whose text is semantically too similar to an already-
        accepted section (claim-level dedup).  Uses a tighter threshold than
        the FAISS block-level dedup to catch paraphrased duplicates.
        """
        accepted: List[Dict[str, Any]] = []
        accepted_embs: List[Any] = []

        for sec in sections:
            text = sec.get("text", "").strip()
            if not text:
                continue

            emb = self.embedding_store.get_embedding(text)
            if emb is None:
                accepted.append(sec)
                continue

            is_dup = False
            for prev_emb in accepted_embs:
                sim = float(emb.dot(prev_emb))
                if sim >= self.claim_dedup_threshold:
                    is_dup = True
                    break

            if not is_dup:
                accepted.append(sec)
                accepted_embs.append(emb)

        return accepted

    def _log_source_balance(self, sections: List[Dict[str, Any]]) -> None:
        """Log how many sections each source contributes; warn if imbalanced."""
        if not sections:
            return

        counter: Counter = Counter()
        char_counter: Counter = Counter()
        for sec in sections:
            label = sec.get("source_label", "unknown") or "unknown"
            counter[label] += 1
            char_counter[label] += len(sec.get("text", ""))

        total_chars = sum(char_counter.values()) or 1
        print("[ORG] Source balance:")
        for label, count in counter.most_common():
            share = char_counter[label] / total_chars
            flag = " ⚠️ OVERWEIGHT" if share > self.max_source_share else ""
            print(f"  {label}: {count} sections, {share:.1%} of content{flag}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_source_label(self, chunk: str) -> str:
        """Extract source label from chunk header like '=== SOURCE: file.pdf ==='."""
        for line in chunk.splitlines()[:3]:
            line = line.strip()
            if line.startswith("=== SOURCE:") and line.endswith("==="):
                return line.replace("=== SOURCE:", "").replace("===", "").strip()
        return ""

    def _build_prompt(self, chunk: str) -> str:
        """Inject TOC outline, source context, and chunk into the prompt template."""
        toc_outline = self.toc.render_outline_for_prompt()

        # Build source context from profiles if available
        source_label = self._detect_source_label(chunk)
        source_context = ""
        if source_label and source_label in self._profile_by_label:
            p = self._profile_by_label[source_label]
            source_context = (
                f"SOURCE CONTEXT:\n"
                f"This chunk comes from '{p.source_label}' which is a {p.source_type} "
                f"document with a {p.rhetorical_role} rhetorical role.\n"
                f"Central thesis: {p.central_argument[:300]}\n"
                f"Treat this source accordingly — preserve its unique framing.\n\n"
            )
        elif source_label:
            source_context = (
                f"SOURCE CONTEXT:\n"
                f"This chunk comes from '{source_label}'. Preserve its framing.\n\n"
            )

        return self.prompt_template.format(
            toc_outline=toc_outline,
            chunk=chunk,
            source_context=source_context,
        )

    def _call_llm(self, prompt: str) -> str:
        """Make a chat.completions call to DeepSeek."""
        response = self.client.chat.completions.create(
            model=chat_model("DOSSIER_MODEL"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a dossier archivist. You place source passages under the "
                        "correct headings and quote them exactly. You never summarize, "
                        "paraphrase, rewrite, or add commentary of your own. Every word you "
                        "put in a \"text\" field is copied verbatim from the source you were "
                        "given, and must be findable there by an exact search."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        message = response.choices[0].message
        return (getattr(message, "content", None) or "").strip()

    @staticmethod
    def _extract_json(raw: str) -> str | None:
        """Try to extract a JSON object from the LLM output."""
        if not raw:
            return None
        raw_strip = raw.strip()
        if raw_strip.startswith("{") and raw_strip.endswith("}"):
            return raw_strip
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = raw[start : end + 1].strip()
        return candidate or None
