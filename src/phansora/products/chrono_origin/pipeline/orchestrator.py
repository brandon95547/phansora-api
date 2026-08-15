"""End-to-end trace pipeline: decompose -> search -> extract -> recurse -> synthesize."""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from ..config import get_settings


ProgressCallback = Callable[[int, str], None]


def _noop_progress(_percent: int, _stage: str) -> None:  # pragma: no cover
    return None
from ..models import Citation, EvidenceDossier, OriginResult, TimelineEvent, TraceRequest, TraceResponse
from ..services.cache import get_cached, normalize_title, save_cached
from phansora.shared.ai.research import GroundedAnswer, build_research_client
from .prompts import (
    DECOMPOSE_PROMPT,
    EXPAND_EXTRACT_PROMPT,
    EXPAND_SEARCH_PROMPT,
    EXTRACT_PROMPT,
    RECURSE_PROMPT,
    SEARCH_PROMPT,
    SOURCE_HIERARCHY,
    SYNTHESIZE_PROMPT,
)

logger = logging.getLogger(__name__)

_VALID_TIERS = {"primary", "repository", "academic", "press", "low_authority", "unknown"}
_VALID_EVIDENCE_TYPES = {
    "primary_document",
    "archaeological",
    "contemporary_record",
    "near_contemporary_account",
    "later_historical_account",
    "scholarly_inference",
    "tradition",
    "disputed",
    "absent",
}
_VALID_CONFIDENCE_LABELS = {"high", "moderate", "low", "speculative"}

# A source URL is a weak signal, but a reliable one at the bottom of the hierarchy:
# these hosts are discovery aids, never the evidentiary foundation of a trace.
_LOW_AUTHORITY_HOSTS = (
    "wikipedia.org", "wikimedia.org", "fandom.com", "reddit.com", "quora.com",
    "medium.com", "substack.com", "blogspot.com", "wordpress.com", "tumblr.com",
    "youtube.com", "youtu.be", "twitter.com", "x.com", "facebook.com", "tiktok.com",
    "pinterest.com", "stackexchange.com", "answers.com",
)
_REPOSITORY_HOSTS = (
    ".museum", "britishmuseum.org", "bl.uk", "loc.gov", "archives.gov", "nationalarchives",
    "bnf.fr", "vatlib.it", "metmuseum.org", "getty.edu", "smithsonian", "digitallibrary",
    "archive.org", "perseus.tufts.edu", "hathitrust.org",
)
_ACADEMIC_HOSTS = (
    ".edu", ".ac.uk", "jstor.org", "doi.org", "cambridge.org", "oup.com", "academia.edu",
    "springer.com", "tandfonline.com", "sciencedirect.com", "brill.com", "degruyter.com",
    "jhu.edu", "persee.fr", "openedition.org",
)


def _default_tier(url: str) -> str:
    """Fall back to a host-based tier when the model does not label a citation."""
    u = (url or "").lower()
    if any(h in u for h in _LOW_AUTHORITY_HOSTS):
        return "low_authority"
    if any(h in u for h in _REPOSITORY_HOSTS):
        return "repository"
    if any(h in u for h in _ACADEMIC_HOSTS):
        return "academic"
    return "unknown"


def _coerce_dossier(raw: Any, *, fallback_claim: str, confidence: float) -> Optional[EvidenceDossier]:
    """Build an EvidenceDossier from model output, tolerating missing/invalid fields.

    Anything the model left out becomes an explicit "None identified" rather than a
    silently absent field — an unanswered evidence question is itself information.
    """
    if not isinstance(raw, dict):
        return None

    def text(key: str, default: str) -> str:
        val = raw.get(key)
        if isinstance(val, (int, float)):
            val = str(val)
        if not isinstance(val, str) or not val.strip():
            return default
        return val.strip()

    ev_type = raw.get("evidence_type")
    if ev_type not in _VALID_EVIDENCE_TYPES:
        ev_type = "scholarly_inference"

    label = raw.get("confidence_label")
    if label not in _VALID_CONFIDENCE_LABELS:
        # Derive it from the numeric confidence so the two never disagree.
        label = (
            "high" if confidence >= 0.75
            else "moderate" if confidence >= 0.5
            else "low" if confidence >= 0.3
            else "speculative"
        )

    return EvidenceDossier(
        claim=text("claim", fallback_claim or "—"),
        earliest_supporting_source=text("earliest_supporting_source", "None identified"),
        estimated_source_date=text("estimated_source_date", "Unknown"),
        earliest_surviving_copy=text("earliest_surviving_copy", "None identified"),
        contemporary_evidence=text("contemporary_evidence", "None identified"),
        independent_corroboration=text("independent_corroboration", "None identified"),
        contradictory_evidence=text("contradictory_evidence", "None identified"),
        evidence_type=ev_type,
        confidence_label=label,
        why=text("why", ""),
        missing_piece=text("missing_piece", ""),
    )


def _earliest_year(mentions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    dated = [m for m in mentions if isinstance(m.get("year"), int)]
    if not dated:
        return mentions[0] if mentions else None
    return min(dated, key=lambda m: m["year"])


def _format_citations_block(citations: List[Dict[str, str]]) -> str:
    if not citations:
        return "(none)"
    lines = []
    for i, c in enumerate(citations, 1):
        lines.append(f"[{i}] {c.get('title') or c.get('url')} -> {c.get('url')}")
    return "\n".join(lines)


def _format_mentions_block(mentions: List[Dict[str, Any]]) -> str:
    if not mentions:
        return "(none)"
    lines = []
    for m in mentions:
        year = m.get("year")
        era = m.get("era_label")
        when = f"{year}" if isinstance(year, int) else (era or "unknown")
        line = (
            f"- when={when} | precision={m.get('precision', 'unknown')} | "
            f"source={m.get('source_title', '?')} | claim={m.get('claim', '')} | "
            f"cites={m.get('citations', [])} | tier={m.get('source_tier', 'unknown')}"
        )
        # Carry the evidence signals the extract stage picked up into synthesis, so
        # the dossier is built from what was actually read rather than re-guessed.
        if m.get("surviving_copy"):
            line += f" | earliest_surviving_copy={m['surviving_copy']}"
        if m.get("chain"):
            line += f" | REPEATS={m['chain']}"
        lines.append(line)
    return "\n".join(lines)


class TraceOrchestrator:
    def __init__(self, client: Optional[object] = None) -> None:
        # Provider chosen by CHRONO_LLM_PROVIDER (deepseek by default).
        self.client = client or build_research_client()
        self.settings = get_settings()

    # ------------------------------------------------------------------ public
    def run(
        self,
        req: TraceRequest,
        on_progress: Optional[ProgressCallback] = None,
    ) -> TraceResponse:
        progress = on_progress or _noop_progress
        started = time.time()
        normalized = normalize_title(req.title)

        progress(2, "Checking cache")
        cached = get_cached(normalized)
        if cached:
            logger.info("Cache hit for %s", normalized)
            progress(100, "Loaded from cache")
            return TraceResponse(**cached)

        max_depth = req.max_depth or self.settings.chrono_max_depth
        max_sources = req.max_sources_per_stage or self.settings.chrono_max_sources_per_stage
        max_queries = self.settings.chrono_max_queries_per_stage

        all_mentions: List[Dict[str, Any]] = []
        all_citations: Dict[str, Dict[str, str]] = {}
        queries_run: List[str] = []
        iterations = 0

        # Stage 1 - Decompose
        progress(8, "Decomposing query")
        plan = self._decompose(req, max_queries=max_queries)
        current_queries: List[str] = plan.get("queries", [])[:max_queries]

        prev_earliest_year: Optional[int] = None
        stagnant_rounds = 0

        # Reserve 10% for cache/decompose, 80% for the recursive loop, 10% for synthesis.
        loop_start_pct = 10
        loop_end_pct = 90
        loop_span = max(1, loop_end_pct - loop_start_pct)

        for depth in range(max_depth):
            iterations += 1
            depth_pct = loop_start_pct + int(loop_span * (depth / max(1, max_depth)))
            logger.info("Trace depth %d with %d queries", depth, len(current_queries))
            if not current_queries:
                break

            # Stage 2 - Search (one grounded call per query)
            queries_to_run = [q for q in current_queries[:max_queries] if q not in queries_run]
            notes_chunks: List[str] = []
            for i, q in enumerate(queries_to_run):
                queries_run.append(q)
                sub_pct = depth_pct + int((loop_span / max(1, max_depth)) * ((i + 1) / max(1, len(queries_to_run) + 2)))
                progress(min(sub_pct, loop_end_pct - 1), f"Round {depth + 1}/{max_depth}: searching ({i + 1}/{len(queries_to_run)})")
                answer = self._search_one(req, q)
                if answer.text:
                    notes_chunks.append(f"### Query: {q}\n{answer.text}")
                for c in answer.citations[:max_sources]:
                    url = c.get("url") or ""
                    if url and url not in all_citations:
                        all_citations[url] = c

            if not notes_chunks:
                break

            # Stage 3 - Extract dated mentions
            progress(min(depth_pct + int(loop_span / max_depth * 0.8), loop_end_pct - 1),
                     f"Round {depth + 1}/{max_depth}: extracting dated mentions")
            new_mentions = self._extract(
                title=req.title,
                notes="\n\n".join(notes_chunks),
                citations=list(all_citations.values()),
            )
            all_mentions.extend(new_mentions)

            # Stage 4 - Decide whether to recurse for older sources
            earliest = _earliest_year(all_mentions)
            earliest_year = earliest.get("year") if earliest else None

            if (
                isinstance(earliest_year, int)
                and isinstance(prev_earliest_year, int)
                and earliest_year >= prev_earliest_year
            ):
                stagnant_rounds += 1
            else:
                stagnant_rounds = 0
            prev_earliest_year = earliest_year if isinstance(earliest_year, int) else prev_earliest_year

            if stagnant_rounds >= 2:
                logger.info("No older candidates after 2 rounds; stopping.")
                break
            if depth == max_depth - 1:
                break
            if earliest is None:
                break

            progress(min(depth_pct + int(loop_span / max_depth * 0.95), loop_end_pct - 1),
                     f"Round {depth + 1}/{max_depth}: hunting for older sources")
            current_queries = self._recurse(
                title=req.title,
                earliest=earliest,
                prior_queries=queries_run,
                max_queries=max_queries,
            )

        # Guard: a trace with no grounded evidence can only synthesize an "unknown"
        # origin — a useless, misleading dossier. Fail loudly instead so the job is
        # marked failed (job_manager) and the caller refunds the credit (Node sync),
        # rather than silently returning an empty result. The usual cause is a broken
        # or unconfigured web-search provider returning zero results.
        if not all_mentions:
            if not all_citations:
                raise RuntimeError(
                    "Web search returned no results for this subject. The search "
                    "provider is likely unconfigured or unavailable — no origin could "
                    "be traced."
                )
            raise RuntimeError(
                "No datable events could be extracted from the sources found for this "
                "subject, so no origin could be traced."
            )

        # Stage 5 - Synthesize
        progress(loop_end_pct, "Synthesizing timeline")
        final = self._synthesize(
            title=req.title,
            mentions=all_mentions,
            citations=list(all_citations.values()),
        )

        progress(97, "Building response")
        response = self._build_response(
            req=req,
            normalized=normalized,
            final=final,
            citations=list(all_citations.values()),
            queries_run=queries_run,
            iterations=iterations,
            duration=time.time() - started,
        )

        try:
            save_cached(normalized, response.model_dump())
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to write cache: %s", exc)

        progress(100, "Done")
        return response

    # ----------------------------------------------------------------- stages
    def _decompose(self, req: TraceRequest, *, max_queries: int) -> Dict[str, Any]:
        prompt = DECOMPOSE_PROMPT.format(
            title=req.title,
            context=req.context or "(none)",
            max_queries=max_queries,
            source_hierarchy=SOURCE_HIERARCHY,
        )
        return self.client.reason_json(prompt, use_reasoning_model=False)

    def _search_one(self, req: TraceRequest, query: str) -> GroundedAnswer:
        ctx = f"(context: {req.context})" if req.context else ""
        prompt = SEARCH_PROMPT.format(
            title=req.title,
            context_clause=ctx,
            query=query,
            source_hierarchy=SOURCE_HIERARCHY,
        )
        try:
            return self.client.grounded_search(prompt)
        except Exception as exc:
            logger.warning("Grounded search failed for %r: %s", query, exc)
            return GroundedAnswer(text="", citations=[], queries=[])

    def _extract(
        self,
        *,
        title: str,
        notes: str,
        citations: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        prompt = EXTRACT_PROMPT.format(
            title=title,
            notes=notes,
            citations_block=_format_citations_block(citations),
        )
        try:
            # Use the light (non-reasoning) path: extracting dated mentions from the
            # provided search notes is mechanical, not a reasoning task. Critically, on
            # GPT-5 Nano the reasoning path burns its whole output-token budget on
            # reasoning for large note prompts and returns EMPTY json ({} -> no
            # mentions), which collapses the whole trace to "unknown". Low effort leaves
            # room for the JSON output. (Verified: 15k-char prompt -> 0 mentions with
            # reasoning, 7 with the light path.)
            data = self.client.reason_json(prompt, use_reasoning_model=False)
            return data.get("mentions", []) or []
        except Exception as exc:
            logger.warning("Extraction failed: %s", exc)
            return []

    def _recurse(
        self,
        *,
        title: str,
        earliest: Dict[str, Any],
        prior_queries: List[str],
        max_queries: int,
    ) -> List[str]:
        prompt = RECURSE_PROMPT.format(
            title=title,
            year=earliest.get("year"),
            era_label=earliest.get("era_label"),
            source_title=earliest.get("source_title", "?"),
            claim=earliest.get("claim", ""),
            max_queries=max_queries,
            prior_queries="\n".join(f"- {q}" for q in prior_queries[-20:]),
        )
        try:
            data = self.client.reason_json(prompt, use_reasoning_model=False)
            return [q for q in (data.get("queries") or []) if q][:max_queries]
        except Exception as exc:
            logger.warning("Recurse query gen failed: %s", exc)
            return []

    def _synthesize(
        self,
        *,
        title: str,
        mentions: List[Dict[str, Any]],
        citations: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        prompt = SYNTHESIZE_PROMPT.format(
            title=title,
            mentions_block=_format_mentions_block(mentions),
            citations_block=_format_citations_block(citations),
            source_hierarchy=SOURCE_HIERARCHY,
        )
        return self.client.reason_json(prompt, use_reasoning_model=True)

    # --------------------------------------------------------------- response
    def _build_response(
        self,
        *,
        req: TraceRequest,
        normalized: str,
        final: Dict[str, Any],
        citations: List[Dict[str, str]],
        queries_run: List[str],
        iterations: int,
        duration: float,
    ) -> TraceResponse:
        url_lookup = {c["url"]: c for c in citations if c.get("url")}
        # Tier labels the synthesis stage assigned, falling back to a host heuristic.
        declared_tiers = final.get("source_tiers")
        declared_tiers = declared_tiers if isinstance(declared_tiers, dict) else {}

        def tier_for(url: str) -> str:
            t = declared_tiers.get(url)
            return t if t in _VALID_TIERS else _default_tier(url)

        def to_citations(urls: List[str]) -> List[Citation]:
            out: List[Citation] = []
            for u in urls or []:
                meta = url_lookup.get(u, {"url": u})
                url = meta.get("url", u)
                out.append(Citation(url=url, title=meta.get("title"), tier=tier_for(url)))
            return out

        origin_data = final.get("origin") or {}
        origin_conf = float(origin_data.get("confidence", 0.5) or 0.5)
        origin = OriginResult(
            year=origin_data.get("year"),
            era_label=origin_data.get("era_label"),
            precision=origin_data.get("precision", "unknown"),
            source_title=origin_data.get("source_title", "Unknown"),
            summary=origin_data.get("summary", ""),
            citations=to_citations(origin_data.get("citations", [])),
            confidence=origin_conf,
            evidence=_coerce_dossier(
                origin_data.get("evidence"),
                fallback_claim=origin_data.get("summary", ""),
                confidence=origin_conf,
            ),
        )

        timeline: List[TimelineEvent] = []
        for ev in final.get("timeline") or []:
            conf = float(ev.get("confidence", 0.5) or 0.5)
            timeline.append(
                TimelineEvent(
                    year=ev.get("year"),
                    era_label=ev.get("era_label"),
                    precision=ev.get("precision", "unknown"),
                    source_title=ev.get("source_title", "Unknown"),
                    claim=ev.get("claim", ""),
                    citations=to_citations(ev.get("citations", [])),
                    confidence=conf,
                    evidence=_coerce_dossier(
                        ev.get("evidence"), fallback_claim=ev.get("claim", ""), confidence=conf
                    ),
                )
            )

        # chronological sort, oldest first; null years go last
        timeline.sort(key=lambda e: (e.year is None, e.year if e.year is not None else 0))

        all_citations = [
            Citation(url=c["url"], title=c.get("title"), tier=tier_for(c["url"]))
            for c in citations
            if c.get("url")
        ]

        return TraceResponse(
            title=req.title,
            normalized_title=normalized,
            origin=origin,
            timeline=timeline,
            reasoning=final.get("reasoning", ""),
            confidence=float(final.get("confidence", 0.5) or 0.5),
            queries_run=queries_run,
            citations=all_citations,
            iterations=iterations,
            duration_seconds=round(duration, 2),
        )

    # ------------------------------------------------------------------ expand
    def expand(self, req: "ExpandRequest") -> "ExpandResponse":  # type: ignore[name-defined]
        """Expand a single timeline item into finer-grained, chronologically ordered sub-events."""
        # Local import to avoid a circular import at module load time.
        from ..models import ExpandRequest, ExpandResponse  # noqa: F401

        started = time.time()

        when = (
            f"{req.parent_year} ({'BCE' if req.parent_year < 0 else 'CE'})"
            if isinstance(req.parent_year, int)
            else (req.parent_era_label or "unknown")
        )
        context_clause = f" (context: {req.context})" if req.context else ""

        # Stage 1 - grounded search around the anchor.
        search_prompt = EXPAND_SEARCH_PROMPT.format(
            story_title=req.story_title,
            context_clause=context_clause,
            when=when,
            parent_source_title=req.parent_source_title,
            parent_claim=req.parent_claim or "(no prior claim recorded)",
            source_hierarchy=SOURCE_HIERARCHY,
        )
        try:
            answer = self.client.grounded_search(search_prompt)
        except Exception as exc:
            logger.warning("Expand grounded search failed: %s", exc)
            answer = GroundedAnswer(text="", citations=[], queries=[])

        citations: List[Dict[str, str]] = []
        seen_urls: set[str] = set()
        for c in answer.citations:
            url = c.get("url") or ""
            if url and url not in seen_urls:
                seen_urls.add(url)
                citations.append(c)

        if not answer.text:
            return ExpandResponse(
                parent_source_title=req.parent_source_title,
                parent_year=req.parent_year,
                parent_era_label=req.parent_era_label,
                events=[],
                queries_run=list(answer.queries or []),
                citations=[
                    Citation(url=c["url"], title=c.get("title"), tier=_default_tier(c["url"]))
                    for c in citations
                    if c.get("url")
                ],
                duration_seconds=round(time.time() - started, 2),
            )

        # Stage 2 - extract structured sub-events.
        extract_prompt = EXPAND_EXTRACT_PROMPT.format(
            story_title=req.story_title,
            when=when,
            parent_source_title=req.parent_source_title,
            parent_claim=req.parent_claim or "(no prior claim recorded)",
            notes=answer.text,
            citations_block=_format_citations_block(citations),
            max_events=req.max_events,
        )
        try:
            # Light path — same reason as _extract: mechanical event extraction, and
            # the reasoning path returns empty JSON on large note prompts with GPT-5 Nano.
            data = self.client.reason_json(extract_prompt, use_reasoning_model=False)
            raw_events = data.get("events") or []
        except Exception as exc:
            logger.warning("Expand extract failed: %s", exc)
            raw_events = []

        url_lookup = {c["url"]: c for c in citations if c.get("url")}

        def to_citations(urls: List[str]) -> List[Citation]:
            out: List[Citation] = []
            for u in urls or []:
                meta = url_lookup.get(u, {"url": u})
                url = meta.get("url", u)
                out.append(Citation(url=url, title=meta.get("title"), tier=_default_tier(url)))
            return out

        events: List[TimelineEvent] = []
        for ev in raw_events[: req.max_events]:
            try:
                conf = float(ev.get("confidence", 0.5) or 0.5)
                events.append(
                    TimelineEvent(
                        year=ev.get("year"),
                        era_label=ev.get("era_label"),
                        precision=ev.get("precision", "unknown"),
                        source_title=ev.get("source_title", "Unknown"),
                        claim=ev.get("claim", ""),
                        citations=to_citations(ev.get("citations", [])),
                        confidence=conf,
                        evidence=_coerce_dossier(
                            ev.get("evidence"), fallback_claim=ev.get("claim", ""), confidence=conf
                        ),
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Skipping malformed expand event: %s", exc)

        events.sort(key=lambda e: (e.year is None, e.year if e.year is not None else 0))

        return ExpandResponse(
            parent_source_title=req.parent_source_title,
            parent_year=req.parent_year,
            parent_era_label=req.parent_era_label,
            events=events,
            queries_run=list(answer.queries or []),
            citations=[
                Citation(url=c["url"], title=c.get("title"), tier=_default_tier(c["url"]))
                for c in citations
                if c.get("url")
            ],
            duration_seconds=round(time.time() - started, 2),
        )
