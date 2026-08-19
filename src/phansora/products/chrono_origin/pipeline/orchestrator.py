"""End-to-end trace pipeline: decompose -> search -> extract+plan -> read -> synthesize.

Two principles govern the shape of this file.

TOKENS BUY JUDGEMENT, NOT REPETITION. Anything deterministic — deduplicating
mentions, deciding which sources are one source repeated, turning an evidence
type into a claim class — happens in ``evidence.py`` for free. The model is asked
only for what it alone can do: read sources, weigh them, and say what is missing.
The loop stops as soon as a round adds nothing new rather than running to a fixed
depth, because a round that found nothing will keep finding nothing.

EVIDENCE IS READ, NOT RECALLED. Search returns titles and URLs; on the OpenAI
path it does not even return snippets. A provenance claim built on that is the
model's memory wearing a citation. So before synthesis the pipeline opens the few
highest-tier pages and reads them, and every claim records whether it rests on
text we actually saw.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config import get_settings


ProgressCallback = Callable[[int, str], None]


def _noop_progress(_percent: int, _stage: str) -> None:  # pragma: no cover
    return None
from ..models import (
    Citation,
    Conclusion,
    Connection,
    ConnectionEvidence,
    EvidenceChain,
    EvidenceDossier,
    OriginResult,
    TimelineEvent,
    TokenUsage,
    TraceRequest,
    TraceResponse,
)
from ..services.cache import get_cached, normalize_title, request_key, save_cached
from phansora.shared.ai import usage
from phansora.shared.ai.research import GroundedAnswer, build_research_client
from . import evidence as ev
from . import source_policy as sp
from .prompts import (
    DECOMPOSE_PROMPT,
    EXPAND_EXTRACT_PROMPT,
    EXPAND_SEARCH_PROMPT,
    CHASE_SEARCH_PROMPT,
    EXTRACT_DOCTRINE,
    EXTRACT_PROMPT,
    SEARCH_DOCTRINE,
    SEARCH_PROMPT,
    SOURCE_HIERARCHY,
    SYNTHESIZE_PROMPT,
)
from .reader import PageRead, format_reads_block, mine_references, read_best

logger = logging.getLogger(__name__)

_VALID_TIERS = sp.VALID_TIERS
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

# The host lists, the tier fallback and the absence vocabulary now live in
# source_policy, where they can be tested without an API key and where the rules
# that read them back sit beside them. Aliased here so the call sites below read
# unchanged.
_is_absent = sp.is_absent
_default_tier = sp.default_tier


_VALID_CONFIDENCE = {"high", "moderate", "low", "speculative"}

_EVIDENCE_KINDS = {
    "text", "manuscript", "scroll", "letter", "inscription",
    "document", "record", "artifact", "archaeological_find",
}
_VALID_ATTRIBUTION = {"established", "attributed", "disputed", "anonymous", "not_applicable"}


def is_evidence_kind(value: Any) -> bool:
    """Is this step a surviving object, i.e. allowed in the chain at all?

    The chain rule is structural, not advisory. The prompt asks for evidence only, but
    a model under pressure to tell a coherent story will still reach for the connective
    tissue between documents — an expectation, a movement, a development. Anything whose
    kind is not one of the nine surviving-object kinds is not evidence, and the caller
    routes it into "conclusions" rather than letting it stand as a link in the chain.

    Deliberately NOT a coercion. The previous version mapped anything unrecognised onto
    a valid type, which meant an interpretation arrived with a respectable label and
    became indistinguishable from an artefact.
    """
    return value in _EVIDENCE_KINDS


def _attribution(value: Any) -> str:
    return value if value in _VALID_ATTRIBUTION else "not_applicable"


def _build_conclusions(
    raw: Any,
    demoted: List[Dict[str, Any]],
    *,
    valid_ids: set,
) -> List[Conclusion]:
    """The readings of the evidence, stated after it.

    Two sources feed this. The model's own "conclusions", and any step it offered that
    was not a surviving object — those are demoted here rather than deleted, because a
    silent drop would hide a claim the model actually made, and the whole point of this
    product is that a reader can see what rests on what.

    A demoted step keeps its own words as the statement and declares that nothing in the
    chain supports it, which is the honest reading: it was offered as evidence, and it
    was not evidence.
    """
    out: List[Conclusion] = []

    for item in (raw or []):
        if not isinstance(item, dict):
            continue
        statement = str(item.get("statement") or "").strip()
        if not statement:
            continue
        # Only ids that actually exist in the chain — a conclusion citing a step that was
        # never emitted reads as supported when it is not.
        rests = [str(r) for r in (item.get("rests_on") or []) if str(r) in valid_ids]
        label = item.get("confidence_label")
        out.append(
            Conclusion(
                statement=statement,
                rests_on=rests,
                confidence_label=label if label in _VALID_CONFIDENCE else "moderate",
                reasoning=str(item.get("reasoning") or "").strip(),
                dissent=str(item.get("dissent") or "").strip() or "None identified",
            )
        )

    for entry in demoted:
        statement = str(entry.get("claim") or entry.get("source_title") or "").strip()
        if not statement:
            continue
        kind = str(entry.get("node_type") or "unspecified")
        out.append(
            Conclusion(
                statement=statement,
                rests_on=[],
                confidence_label="speculative",
                reasoning=(
                    f"Offered as a step in the chain (as \"{kind}\"), but it is not a surviving "
                    "object that can be examined, so it is recorded here as a reading of the "
                    "evidence rather than as evidence."
                ),
                dissent="None identified",
            )
        )

    return out


def _coerce_dossier(
    raw: Any,
    *,
    fallback_claim: str,
    confidence: float,
    read_urls: Optional[set] = None,
    citations: Optional[List[str]] = None,
    citation_ranks: Optional[Dict[str, int]] = None,
) -> Optional[EvidenceDossier]:
    """Build an EvidenceDossier from model output, tolerating missing/invalid fields.

    Anything the model left out becomes an explicit "None identified" rather than a
    silently absent field — an unanswered evidence question is itself information.

    Several fields are settled here rather than asked for, because they are facts
    about the run and not judgements: ``claim_class`` is derived from the evidence
    type so it cannot be talked up, ``disputed`` follows from contradictory
    evidence actually being present, ``verified_from_source`` records whether this
    pipeline opened any of the cited pages, and ``verification`` records whether
    anything under the claim could serve as evidence at all.

    ``citation_ranks`` is what makes the difference between stating a policy and
    holding to one. The evidence type is capped at what the strongest citation can
    actually support, so a claim standing on a wiki cannot describe itself as a
    primary document — and because claim_class derives from evidence type, that cap
    reaches the marker on the board without the renderer knowing anything about it.
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

    contradictory = text("contradictory_evidence", "None identified")
    dispute = text("scholarly_dispute", "None identified")
    read = read_urls or set()
    used = [u for u in (citations or []) if u in read]

    cites = [u for u in (citations or []) if u]
    ranks = citation_ranks or {}
    best = sp.best_rank(ranks.get(u, sp.WORST_RANK) for u in cites)
    ev_type = sp.cap_evidence_type(ev_type, best)
    earliest = text("earliest_supporting_source", "None identified")
    verification = sp.verification_for(
        best_evidence_rank=best,
        has_citations=bool(cites),
        read_from_source=bool(used),
        earliest_supporting_source=earliest,
    )
    # An honest absence is a finding with its own marker on the board, so it has
    # to travel through evidence_type — the one field claim_class is derived from.
    if verification == "unknown":
        ev_type = "absent"

    return EvidenceDossier(
        claim=text("claim", fallback_claim or "—"),
        earliest_supporting_source=earliest,
        estimated_source_date=text("estimated_source_date", "Unknown"),
        earliest_surviving_copy=text("earliest_surviving_copy", "None identified"),
        contemporary_evidence=text("contemporary_evidence", "None identified"),
        independent_corroboration=text("independent_corroboration", "None identified"),
        contradictory_evidence=contradictory,
        provenance=text("provenance", "None identified"),
        scholarly_dispute=dispute,
        evidence_type=ev_type,
        claim_class=ev.claim_class_for(ev_type, raw.get("claim_class")),
        confidence_label=label,
        why=text("why", ""),
        missing_piece=text("missing_piece", ""),
        disputed=(ev_type == "disputed") or not _is_absent(contradictory) or not _is_absent(dispute),
        verification=verification,
        verified_from_source=bool(used),
        sources_read=used,
    )


def _coerce_connection_evidence(
    raw: Any,
    *,
    read_urls: set,
    citations: List[str],
    citation_ranks: Optional[Dict[str, int]] = None,
) -> ConnectionEvidence:
    """Grade a connection on the same terms as a claim, defaulting to honest ignorance."""
    raw = raw if isinstance(raw, dict) else {}

    def text(key: str, default: str) -> str:
        val = raw.get(key)
        if isinstance(val, (int, float)):
            val = str(val)
        if not isinstance(val, str) or not val.strip():
            return default
        return val.strip()

    try:
        conf = float(raw.get("confidence", 0.5) or 0.5)
    except (TypeError, ValueError):
        conf = 0.5
    conf = min(1.0, max(0.0, conf))

    ev_type = raw.get("evidence_type")
    if ev_type not in _VALID_EVIDENCE_TYPES:
        ev_type = "scholarly_inference"

    label = raw.get("confidence_label")
    if label not in _VALID_CONFIDENCE_LABELS:
        label = (
            "high" if conf >= 0.75
            else "moderate" if conf >= 0.5
            else "low" if conf >= 0.3
            else "speculative"
        )

    contradictory = text("contradictory_evidence", "None identified")
    dispute = text("scholarly_dispute", "None identified")

    cites = [u for u in (citations or []) if u]
    ranks = citation_ranks or {}
    best = sp.best_rank(ranks.get(u, sp.WORST_RANK) for u in cites)
    ev_type = sp.cap_evidence_type(ev_type, best)
    supporting = text("supporting_evidence", "None identified")
    verification = sp.verification_for(
        best_evidence_rank=best,
        has_citations=bool(cites),
        read_from_source=bool([u for u in cites if u in (read_urls or set())]),
        earliest_supporting_source=supporting,
    )
    if verification == "unknown":
        ev_type = "absent"

    return ConnectionEvidence(
        mechanism=text("mechanism", "No mechanism established."),
        supporting_evidence=supporting,
        contradictory_evidence=contradictory,
        independent_corroboration=text("independent_corroboration", "None identified"),
        scholarly_dispute=dispute,
        claim_class=ev.claim_class_for(ev_type, raw.get("claim_class")),
        evidence_type=ev_type,
        verification=verification,
        confidence=conf,
        confidence_label=label,
        why=text("why", ""),
        missing_piece=text("missing_piece", ""),
        disputed=(ev_type == "disputed") or not _is_absent(contradictory) or not _is_absent(dispute),
    )


# The strands a trace can be built from. Kept here rather than in the prompt
# alone so the loop can measure coverage instead of taking the model's word for it.
_STRANDS = {
    "precursor_evidence", "earliest_texts", "manuscripts", "external_sources",
    "documents_records", "inscriptions_artifacts", "archaeology",
}
# A mention reports its strand covered by the KIND of surviving thing it is. The
# strands are categories of evidence to go looking for, so the mapping is just
# "which hunt would have turned this up".
_NODE_TYPE_STRAND = {
    "text": "earliest_texts",
    "letter": "earliest_texts",
    "manuscript": "manuscripts",
    "scroll": "manuscripts",
    "inscription": "inscriptions_artifacts",
    "artifact": "inscriptions_artifacts",
    "document": "documents_records",
    "record": "documents_records",
    "archaeological_find": "archaeology",
}


# A search for each strand, written here rather than asked for. The extract
# prompt has always been told to cover the open strands first, and it has always
# had to do that inside one list of six queries also carrying "push further
# back", "chase this lead", "find independent corroboration" and "find the
# scholarship that disputes this". Recency won every time, which is how a trace
# for Jesus Christ ran twenty-four searches without one of them asking what
# calendar the dates were in, who Paul was, or when the gospels were written.
#
# So the open strands get their slots taken off the top and the model plans with
# what is left. These are deliberately generic: a strand is a KIND of question,
# and the grounded-search step is what turns "what texts exist and when were
# they written" into a subject's actual bibliography.
_STRAND_QUERIES = {
    "precursor_evidence": (
        "what texts, manuscripts, inscriptions and excavated objects survive from BEFORE "
        "{title}, in the same culture and language, with their dates and where they are held"
    ),
    "earliest_texts": (
        "the earliest surviving texts about or by {title}, ordered earliest first, with "
        "estimated composition dates and the scholarly basis for each date"
    ),
    "manuscripts": (
        "earliest surviving manuscripts, scrolls and fragments relating to {title}: shelfmark, "
        "holding repository, palaeographic date, and how far they postdate composition"
    ),
    "external_sources": (
        "surviving texts from outside the tradition of {title} that mention it, their own "
        "composition dates, and how those texts themselves physically survive"
    ),
    "documents_records": (
        "surviving documents and administrative records — decrees, censuses, court, tax or "
        "official registers — from the period and place of {title}, and which archive holds them"
    ),
    "inscriptions_artifacts": (
        "surviving inscriptions, coins, seals, ostraca and inscribed objects relating to "
        "{title}, with their find context, date and holding institution"
    ),
    "archaeology": (
        "excavated sites, structures and assemblages relevant to {title}, as reported in "
        "excavation reports, with dates and the report that publishes each find"
    ),
}


def _strand_queries(title: str, open_strands: List[str], already_run: List[str], *, limit: int) -> List[str]:
    """One search per still-open strand, up to ``limit``, skipping repeats."""
    if limit <= 0:
        return []
    seen = {q.strip().lower() for q in already_run}
    out: List[str] = []
    for strand in open_strands:
        template = _STRAND_QUERIES.get(strand)
        if not template:
            continue
        query = template.format(title=title)
        if query.lower() in seen:
            continue
        seen.add(query.lower())
        out.append(query)
        if len(out) >= limit:
            break
    return out


def _as_strands(raw: Any) -> List[str]:
    """The planned strands, tolerating both ['name'] and [{'strand': name}] shapes."""
    out: List[str] = []
    for item in raw or []:
        name = item.get("strand") if isinstance(item, dict) else item
        name = str(name or "").strip().lower()
        if name in _STRANDS and name not in out:
            out.append(name)
    return out


def _strands_covered(mentions: List[Dict[str, Any]]) -> set:
    """Which strands this batch of mentions actually produced evidence for."""
    covered: set = set()
    for m in mentions or []:
        if not isinstance(m, dict):
            continue
        node_type = str(m.get("node_type") or "").strip().lower()
        strand = _NODE_TYPE_STRAND.get(node_type, node_type)
        if strand in _STRANDS:
            covered.add(strand)
    return covered


def _open_strands(planned: List[str], covered: set) -> List[str]:
    return [s for s in planned if s not in covered]


def _earliest_year(mentions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    dated = [m for m in mentions if isinstance(m.get("year"), int)]
    if not dated:
        return mentions[0] if mentions else None
    return min(dated, key=lambda m: m["year"])


def _describe_earliest(earliest: Optional[Dict[str, Any]]) -> str:
    if not earliest:
        return "(nothing established yet)"
    year = earliest.get("year")
    when = f"{year}" if isinstance(year, int) else (earliest.get("era_label") or "unknown")
    return f"{when} — {earliest.get('source_title', '?')}: {earliest.get('claim', '')}"[:300]


def _format_citations_block(citations: List[Dict[str, str]]) -> str:
    if not citations:
        return "(none)"
    lines = []
    for i, c in enumerate(citations, 1):
        lines.append(f"[{i}] {c.get('title') or c.get('url')} -> {c.get('url')}")
    return "\n".join(lines)


def _format_mentions_block(mentions: List[Dict[str, Any]]) -> str:
    """Everything the research rounds established about an item, for synthesis.

    The extract stage types every mention — this is a text being composed, this
    is a surviving copy of one, this is the calendar the dates were converted
    into — and the loop measures its own coverage against those types. None of
    it used to reach synthesis: this block sent the date, the title, the claim
    and the tier, and dropped the type, the span and the provenance signals on
    the floor.

    The result was a trace that believed it had researched a subject's texts and
    then published a report with no text in it. Coverage was measured on the
    mentions and breadth was decided by a stage that had never been told what
    kind of thing any of them were, so the loop would exit satisfied while the
    report it produced was missing whole strands. Everything the extractor
    established travels now.
    """
    if not mentions:
        return "(none)"
    lines = []
    for m in mentions:
        year = m.get("year")
        era = m.get("era_label")
        when = f"{year}" if isinstance(year, int) else (era or "unknown")
        end = m.get("year_end")
        if isinstance(end, int) and end != year:
            when = f"{when}..{end}"
        line = (
            f"- when={when} | precision={m.get('precision', 'unknown')} | "
            f"type={m.get('node_type') or 'event'} | "
            f"source={m.get('source_title', '?')} | claim={m.get('claim', '')} | "
            f"cites={m.get('citations', [])} | tier={m.get('source_tier', 'unknown')}"
        )
        # Carry the evidence signals the extract stage picked up into synthesis, so
        # the dossier is built from what was actually read rather than re-guessed.
        if m.get("published"):
            line += f" | source_published={m['published']}"
        if m.get("surviving_copy"):
            line += f" | earliest_surviving_copy={m['surviving_copy']}"
        if m.get("discovery_only"):
            line += " | LEAD_ONLY=this appears only on a tier 4-5 page"
        if m.get("cites"):
            line += f" | that_page_cites={m['cites']}"
        if m.get("chain"):
            line += f" | REPEATS={m['chain']}"
        lines.append(line)
    return "\n".join(lines)


def _format_strands_block(planned: List[str], covered: set) -> str:
    """What the research plan asked for, and which parts of it found something.

    Handed to synthesis so it can tell the two kinds of silence apart. A strand
    the rounds covered and the report omits is a report that threw away research
    the user paid for; a strand the rounds could not cover is a finding, and
    belongs in the open questions rather than being quietly dropped.
    """
    if not planned:
        return "(no strand plan recorded)"
    lines = []
    for strand in planned:
        state = "RESEARCHED — must appear as at least one entry" if strand in covered else (
            "NOT COVERED — say so in the reasoning; do not invent entries for it"
        )
        lines.append(f"- {strand}: {state}")
    extra = sorted(covered - set(planned))
    for strand in extra:
        lines.append(f"- {strand}: RESEARCHED (unplanned) — must appear as at least one entry")
    return "\n".join(lines)


def _as_queries(raw: Any) -> List[str]:
    """A model's query list, however it chose to shape it.

    The decompose prompt asks for queries "allocated across the hierarchy", and the model
    answers that instruction two ways: sometimes a list of strings, sometimes a list of
    {"tier": n, "query": "..."} objects. TraceResponse.queries_run is List[str], so the
    object form failed Pydantic validation at the very END of a trace — after every search
    and every model call had been paid for. Runs died at 97%, "Building response", with a
    string_type error naming a dict.

    Both shapes are legitimate readings of the prompt, so this accepts both rather than
    trying to make the model more obedient. Anything with no usable query text is dropped:
    a blank search is a wasted round, not a query.
    """
    out: List[str] = []
    for item in raw or []:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            # "query" is what the prompt names it; the others are what models reach for.
            text = str(item.get("query") or item.get("q") or item.get("text") or "").strip()
        else:
            continue
        if text:
            out.append(text)
    return out


class TraceOrchestrator:
    def __init__(self, client: Optional[object] = None) -> None:
        # Provider chosen by CHRONO_LLM_PROVIDER.
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
        usage.start()
        normalized = normalize_title(req.title)
        # The WHOLE request, not just the title: context, depth, sources and language all
        # change the answer, and keying on the title alone meant the first run of a title
        # answered every later variation of it.
        key = request_key(
            req.title,
            context=req.context,
            max_depth=req.max_depth,
            max_sources_per_stage=req.max_sources_per_stage,
            language=req.language,
        )

        progress(2, "Checking cache")
        cached = get_cached(req.title, key)
        if cached:
            logger.info("Cache hit for %s", normalized)
            progress(100, "Loaded from cache")
            return TraceResponse(**cached)

        max_depth = req.max_depth or self.settings.chrono_max_depth
        min_depth = max(1, min(self.settings.chrono_min_depth, max_depth))
        max_sources = req.max_sources_per_stage or self.settings.chrono_max_sources_per_stage
        max_queries = self.settings.chrono_max_queries_per_stage

        all_mentions: List[Dict[str, Any]] = []
        all_citations: Dict[str, Dict[str, str]] = {}
        queries_run: List[str] = []
        gaps: List[str] = []
        iterations = 0

        # Stage 1 - Decompose
        progress(8, "Planning the research")
        usage.stage("decompose")
        plan = self._decompose(req, max_queries=max_queries)
        current_queries: List[str] = _as_queries(plan.get("queries"))[:max_queries]

        # The strands this subject needs covered. The loop used to run until it
        # stopped finding anything OLDER, which is why traces came back as a thin
        # chain of dates: once the earliest text was found there was nothing left
        # to look for, and manuscripts, outside attestation, language and later
        # institutional development were never researched at all. A trace is
        # finished when its strands are covered, not when the origin stops moving.
        planned_strands = _as_strands(plan.get("strands"))
        covered_strands: set[str] = set()

        prev_earliest_year: Optional[int] = None
        stagnant_rounds = 0

        # Reserve 10% for cache/decompose, 70% for the loop, 20% for reading+synthesis.
        loop_start_pct, loop_end_pct = 10, 80
        loop_span = max(1, loop_end_pct - loop_start_pct)

        for depth in range(max_depth):
            iterations += 1
            depth_pct = loop_start_pct + int(loop_span * (depth / max(1, max_depth)))
            logger.info("Trace depth %d with %d queries", depth, len(current_queries))
            if not current_queries:
                break

            # Stage 2 - Search. Concurrent: these calls do not depend on each
            # other, and running five of them in series is the single largest
            # chunk of a trace's wall clock. Costs nothing in tokens.
            queries_to_run = [q for q in current_queries[:max_queries] if q not in queries_run]
            if not queries_to_run:
                break
            queries_run.extend(queries_to_run)
            progress(
                min(depth_pct + 2, loop_end_pct - 1),
                f"Round {depth + 1}: searching {len(queries_to_run)} angles",
            )
            answers = self._search_many(req, queries_to_run)

            notes_chunks: List[str] = []
            round_urls: List[str] = []
            for q, answer in zip(queries_to_run, answers):
                if answer.text:
                    notes_chunks.append(f"### Query: {q}\n{answer.text}")
                for c in answer.citations[:max_sources]:
                    url = c.get("url") or ""
                    if not url:
                        continue
                    if url not in all_citations:
                        all_citations[url] = c
                    round_urls.append(url)

            if not notes_chunks:
                break

            # Stage 3 - Extract dated mentions AND plan the next round in one call.
            progress(
                min(depth_pct + int(loop_span / max_depth * 0.8), loop_end_pct - 1),
                f"Round {depth + 1}: extracting dated mentions",
            )
            usage.stage("extract")
            extracted = self._extract(
                title=req.title,
                notes="\n\n".join(notes_chunks),
                # Only this round's citations. Passing every URL gathered so far
                # grew the prompt on every round AND invited the model to cite
                # sources that appear nowhere in the notes it was handed.
                citations=[all_citations[u] for u in dict.fromkeys(round_urls)],
                earliest_known=_describe_earliest(_earliest_year(all_mentions)),
                prior_queries=queries_run,
                max_queries=max_queries,
                open_strands=_open_strands(planned_strands, covered_strands),
            )
            new_mentions = extracted.get("mentions", [])
            for g in extracted.get("gaps", []) or []:
                if isinstance(g, str) and g.strip() and g not in gaps:
                    gaps.append(g.strip())

            fresh = ev.new_mention_count(all_mentions, new_mentions)
            all_mentions = ev.dedupe_mentions(all_mentions + new_mentions)
            newly_covered = _strands_covered(new_mentions) - covered_strands
            covered_strands |= newly_covered

            # ---- Stopping rules. Every extra round is a full set of searches, so
            # the loop must justify continuing rather than run to the ceiling.
            if depth + 1 >= min_depth and fresh == 0:
                logger.info("Round %d added no new mentions; stopping.", depth + 1)
                break

            # A round is productive if it pushed the origin back OR covered a
            # strand that was still open. Judging only on recency is what made
            # the loop quit with half the subject unresearched.
            earliest = _earliest_year(all_mentions)
            earliest_year = earliest.get("year") if earliest else None
            went_older = not (
                isinstance(earliest_year, int)
                and isinstance(prev_earliest_year, int)
                and earliest_year >= prev_earliest_year
            )
            open_now = _open_strands(planned_strands, covered_strands)
            if went_older or newly_covered:
                stagnant_rounds = 0
            else:
                stagnant_rounds += 1
            prev_earliest_year = earliest_year if isinstance(earliest_year, int) else prev_earliest_year

            # Stagnation ends a trace only once there is nothing left to look
            # for. A round can fail to move the origin and fail to close a
            # strand and still be one round short of the strand it was working
            # on, and quitting there is how a report comes back with no texts,
            # no calendar and no language chain while its own plan still listed
            # all three as open. Open strands buy the loop two more rounds; they
            # cannot buy it an unbounded number, because a strand this subject
            # has no evidence for will stay open forever.
            if stagnant_rounds >= 2 and not open_now:
                logger.info("Two rounds with no older evidence and no strand left open; stopping.")
                break
            if stagnant_rounds >= 4:
                logger.info(
                    "Four stagnant rounds; stopping with strands still open: %s", open_now
                )
                break
            if not open_now and not went_older and depth + 1 >= min_depth:
                logger.info("All planned strands covered and the origin has settled; stopping.")
                break
            if depth == max_depth - 1 or earliest is None:
                break

            # The open strands take their slots off the top; the model plans the
            # rest. Reserving at most half the round leaves it room to chase the
            # leads and contradictions it just found, which is work no template
            # can write in advance.
            planned = [
                q for q in _as_queries(extracted.get("next_queries")) if q.strip()
            ]
            reserved = _strand_queries(
                req.title, open_now, queries_run, limit=max(1, max_queries // 2)
            )
            current_queries = list(dict.fromkeys(reserved + planned))[:max_queries]

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

        # Stage 3.5 - Chase. Every claim still resting on a lead gets one attempt
        # to find what it actually rests on, which is the half of the doctrine
        # that was never implemented: the prompts have always said to follow a
        # wiki's references backward, and nothing in the pipeline could.
        mention_tiers = self._mention_tiers(all_mentions)

        def pre_tier(url: str) -> str:
            return sp.resolve_tier(mention_tiers.get(url), url)

        if self.settings.chrono_chase_enabled:
            progress(78, "Chasing citations backward")
            usage.stage("chase")
            try:
                chased = self._chase(
                    req,
                    mentions=all_mentions,
                    citations=list(all_citations.values()),
                    tier_for=pre_tier,
                    queries_run=queries_run,
                    max_sources=max_sources,
                )
                for url, meta in chased.get("citations", {}).items():
                    all_citations.setdefault(url, meta)
                new_mentions = chased.get("mentions", [])
                if new_mentions:
                    all_mentions = ev.dedupe_mentions(all_mentions + new_mentions)
                    mention_tiers = self._mention_tiers(all_mentions)
            except Exception as exc:  # noqa: BLE001 - a failed chase must not cost a trace
                logger.warning("Citation chase failed: %s", exc)

        # Stage 4 - Read the strongest sources properly, before judging them.
        citation_list = list(all_citations.values())

        progress(82, "Reading source pages")
        want_reads = self.settings.chrono_read_sources
        # Rank more candidates than we intend to read: the strongest sources are
        # also the ones most likely to refuse a server-side fetch, and a blocked
        # page should cost us the next-best source, not the read itself.
        to_read = ev.select_for_reading(
            citation_list, all_mentions, pre_tier, limit=want_reads * 2
        )
        reads = read_best(to_read, want=want_reads, max_chars=self.settings.chrono_read_chars)
        read_ok = [r for r in reads if r.ok]
        logger.info("Read %d source pages (from %d candidates)", len(read_ok), len(to_read))

        # Stage 5 - Synthesize
        progress(88, "Synthesizing timeline")
        usage.stage("synthesize")
        final = self._synthesize(
            title=req.title,
            mentions=all_mentions,
            citations=citation_list,
            reads=reads,
            # What the research set out to cover, and what it actually found.
            # Synthesis used to be handed a flat list of items with no idea that
            # any of them belonged to a strand, so a strand researched across
            # four rounds could be collapsed into one line or dropped, and
            # nothing downstream could tell the difference between "the evidence
            # was not there" and "the report did not mention it".
            planned_strands=planned_strands,
            covered_strands=_strands_covered(all_mentions),
        )

        progress(97, "Building response")
        response = self._build_response(
            req=req,
            normalized=normalized,
            final=final,
            citations=citation_list,
            mentions=all_mentions,
            reads=reads,
            queries_run=queries_run,
            open_questions=gaps,
            iterations=iterations,
            duration=time.time() - started,
        )

        # An empty trace is a FAILED trace, and must not be stored or cached as a
        # success. A synthesize step that returned nothing usable still produces a
        # well-formed TraceResponse — every field simply takes its default — so the
        # result looked complete: origin with no year, no summary, no citations, and an
        # empty timeline, marked done, with the user's credit spent. Raising here sends
        # it down the failure path that already exists, which refunds.
        if not response.timeline and response.origin.year is None and not response.origin.summary:
            raise RuntimeError(
                "The trace produced no timeline and no origin. This usually means the "
                "model's answer could not be read; the credit has been refunded."
            )

        try:
            save_cached(req.title, key, response.model_dump())
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to write cache: %s", exc)

        progress(100, "Done")
        return response

    # ----------------------------------------------------------------- stages
    @staticmethod
    def _mention_tiers(mentions: List[Dict[str, Any]]) -> Dict[str, str]:
        """Tier per URL as claimed at extract time, for use before synthesis runs."""
        out: Dict[str, str] = {}
        for m in mentions:
            tier = m.get("source_tier")
            if tier not in _VALID_TIERS:
                continue
            for url in m.get("citations") or []:
                # Keep the strongest claim made about a URL across rounds.
                if url and (
                    url not in out
                    or ev.TIER_ORDER.index(tier) < ev.TIER_ORDER.index(out[url])
                ):
                    out[url] = tier
        return out

    def _decompose(self, req: TraceRequest, *, max_queries: int) -> Dict[str, Any]:
        prompt = DECOMPOSE_PROMPT.format(
            title=req.title,
            context=req.context or "(none)",
            max_queries=max_queries,
            source_hierarchy=SOURCE_HIERARCHY,
        )
        return self.client.reason_json(prompt, use_reasoning_model=False)

    def _search_many(self, req: TraceRequest, queries: List[str]) -> List[GroundedAnswer]:
        """Run this round's searches concurrently, in the order asked.

        Each worker meters itself and hands the total back: the usage counter is
        thread-local so concurrent traces cannot contaminate each other, which
        also means a worker's calls land in its own meter until absorbed here.
        """
        def one(query: str) -> Tuple[GroundedAnswer, Dict[str, Any]]:
            usage.start()
            usage.stage("search")
            try:
                answer = self._search_one(req, query)
            except Exception as exc:  # noqa: BLE001 - a dead query must not kill the round
                logger.warning("Grounded search failed for %r: %s", query, exc)
                answer = GroundedAnswer(text="", citations=[], queries=[])
            return answer, usage.snapshot()

        if len(queries) == 1:
            usage.stage("search")
            return [self._search_one(req, queries[0])]

        with ThreadPoolExecutor(max_workers=min(5, len(queries))) as pool:
            results = list(pool.map(one, queries))
        for _, snap in results:
            usage.absorb(snap)
        return [answer for answer, _ in results]

    def _search_prompts(self, prompts: List[str]) -> List[GroundedAnswer]:
        """Run already-formatted search prompts concurrently.

        _search_many builds SEARCH_PROMPT around a bare query; the chase writes
        its own prompt, so it needs the concurrency without the template.
        """
        def one(prompt: str) -> Tuple[GroundedAnswer, Dict[str, Any]]:
            usage.start()
            usage.stage("chase")
            try:
                answer = self.client.grounded_search(prompt)
            except Exception as exc:  # noqa: BLE001 - a dead chase is just a dead lead
                logger.warning("Chase search failed: %s", exc)
                answer = GroundedAnswer(text="", citations=[], queries=[])
            return answer, usage.snapshot()

        if not prompts:
            return []
        if len(prompts) == 1:
            usage.stage("chase")
            try:
                return [self.client.grounded_search(prompts[0])]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Chase search failed: %s", exc)
                return [GroundedAnswer(text="", citations=[], queries=[])]

        with ThreadPoolExecutor(max_workers=min(3, len(prompts))) as pool:
            results = list(pool.map(one, prompts))
        for _, snap in results:
            usage.absorb(snap)
        return [answer for answer, _ in results]

    def _search_one(self, req: TraceRequest, query: str) -> GroundedAnswer:
        ctx = f"(context: {req.context})" if req.context else ""
        prompt = SEARCH_PROMPT.format(
            title=req.title,
            context_clause=ctx,
            query=query,
            search_doctrine=SEARCH_DOCTRINE,
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
        earliest_known: str,
        prior_queries: List[str],
        max_queries: int,
        pages_block: str = "",
        open_strands: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        prompt = EXTRACT_PROMPT.format(
            title=title,
            notes=notes,
            pages_block=pages_block,
            citations_block=_format_citations_block(citations),
            earliest_known=earliest_known,
            max_queries=max_queries,
            extract_doctrine=EXTRACT_DOCTRINE,
            open_strands=", ".join(open_strands or []) or "(none — all planned strands covered)",
            prior_queries="\n".join(f"- {q}" for q in prior_queries[-20:]) or "- (none yet)",
        )
        try:
            # Use the light (non-reasoning) path: extracting dated mentions from the
            # provided material is mechanical, not a reasoning task. Critically, on a
            # reasoning model the heavy path burns its whole output-token budget on
            # reasoning for large note prompts and returns EMPTY json ({} -> no
            # mentions), which collapses the whole trace to "unknown". Low effort leaves
            # room for the JSON output. (Verified: 15k-char prompt -> 0 mentions with
            # reasoning, 7 with the light path.)
            data = self.client.reason_json(prompt, use_reasoning_model=False)
        except Exception as exc:
            logger.warning("Extraction failed: %s", exc)
            return {"mentions": [], "next_queries": [], "gaps": []}
        return {
            "mentions": data.get("mentions") or [],
            "next_queries": data.get("next_queries") or [],
            "gaps": data.get("gaps") or [],
        }

    def _chase(
        self,
        req: TraceRequest,
        *,
        mentions: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        tier_for: Callable[[str], str],
        queries_run: List[str],
        max_sources: int,
    ) -> Dict[str, Any]:
        """Find what the weakly-sourced claims actually rest on.

        Two passes, cheapest first, and the cheap one usually does the work.

        Mining costs no tokens at all: open the lead pages the trace is leaning
        on, take their reference lists, and hand the URLs to the ranker. A wiki's
        footnotes are mostly JSTOR, DOIs, archives and university presses, so the
        page that could never be evidence becomes the thing it was always
        supposed to be — a way of finding evidence.

        Only then, and only for claims still standing on nothing arguable, spend
        a search naming the underlying work. Returns whatever it found; finding
        nothing is a valid outcome and costs the trace nothing but the attempt.
        """
        found_citations: Dict[str, Dict[str, str]] = {}

        # --- free pass: harvest the leads' own reference lists
        lead_pages = ev.select_for_reference_mining(
            citations, tier_for, limit=self.settings.chrono_chase_mine_pages
        )
        harvested = mine_references(lead_pages) if lead_pages else []
        for ref in harvested:
            url = ref.get("url") or ""
            if url and url not in found_citations:
                found_citations[url] = {"url": url, "title": ref.get("text") or ""}
        if harvested:
            logger.info(
                "Mined %d references from %d lead pages", len(harvested), len(lead_pages)
            )

        # --- paid pass: ask what the weakest claims are actually built on
        targets = ev.chase_targets(
            mentions, tier_for, limit=self.settings.chrono_chase_max_targets
        )
        if not targets:
            logger.info("No claims resting on leads; chase search skipped.")
            return {"citations": found_citations, "mentions": []}

        refs_block = "; ".join(
            f"{r.get('text') or ''} <{r.get('url')}>" for r in harvested[:12]
        ) or "(none harvested)"

        prompts: List[str] = []
        for m in targets[: self.settings.chrono_chase_max_queries]:
            weak = next((u for u in (m.get("citations") or []) if u), "(unknown)")
            prompts.append(
                CHASE_SEARCH_PROMPT.format(
                    title=req.title,
                    claim=str(m.get("claim") or m.get("source_title") or "")[:300],
                    weak_source=weak,
                    cites=str(m.get("cites") or "(not stated)")[:200],
                    references=refs_block[:1200],
                    search_doctrine=SEARCH_DOCTRINE,
                )
            )

        answers = self._search_prompts(prompts)
        notes: List[str] = []
        round_urls: List[str] = []
        for answer in answers:
            if answer.text:
                notes.append(answer.text)
            for c in answer.citations[:max_sources]:
                url = c.get("url") or ""
                if url and url not in found_citations:
                    found_citations[url] = c
                if url:
                    round_urls.append(url)

        if not notes:
            return {"citations": found_citations, "mentions": []}

        usage.stage("extract")
        extracted = self._extract(
            title=req.title,
            notes="\n\n".join(notes),
            citations=[found_citations[u] for u in dict.fromkeys(round_urls) if u in found_citations],
            earliest_known=_describe_earliest(_earliest_year(mentions)),
            prior_queries=queries_run,
            max_queries=0,  # the chase is the last word; it does not plan another round
            open_strands=[],
        )
        return {"citations": found_citations, "mentions": extracted.get("mentions", [])}

    def _synthesize(
        self,
        *,
        title: str,
        mentions: List[Dict[str, Any]],
        citations: List[Dict[str, str]],
        reads: List[PageRead],
        planned_strands: Optional[List[str]] = None,
        covered_strands: Optional[set] = None,
    ) -> Dict[str, Any]:
        pages_block = ""
        if reads:
            pages_block = (
                "\nSOURCE PAGES actually read (prefer these over any summary of them):\n"
                f"{format_reads_block(reads)}\n"
            )
        prompt = SYNTHESIZE_PROMPT.format(
            title=title,
            mentions_block=_format_mentions_block(mentions),
            citations_block=_format_citations_block(citations),
            pages_block=pages_block,
            strands_block=_format_strands_block(planned_strands or [], covered_strands or set()),
            source_hierarchy=SOURCE_HIERARCHY,
            max_connections=self.settings.chrono_max_connections,
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
        mentions: List[Dict[str, Any]],
        reads: List[PageRead],
        queries_run: List[str],
        open_questions: List[str],
        iterations: int,
        duration: float,
    ) -> TraceResponse:
        url_lookup = {c["url"]: c for c in citations if c.get("url")}
        read_urls = {r.url for r in reads if r.ok}

        # Tier labels the synthesis stage assigned, falling back to what extract
        # claimed, then to a host heuristic.
        declared_tiers = final.get("source_tiers")
        declared_tiers = declared_tiers if isinstance(declared_tiers, dict) else {}
        mention_tiers = self._mention_tiers(mentions)

        def tier_for(url: str) -> str:
            claimed = declared_tiers.get(url)
            if claimed not in _VALID_TIERS:
                claimed = mention_tiers.get(url)
            # resolve_tier, not the raw claim: a wiki labelled "primary" by the
            # model is still a wiki, and letting the label win is how one got
            # read as a source page and cited as evidence.
            return sp.resolve_tier(claimed, url)

        # Which sources are really one source repeated. Computed from the sources
        # themselves rather than asked for, so "independent corroboration" has an
        # arithmetic meaning instead of a rhetorical one.
        chain_of, chain_groups = ev.build_chains(citations, mentions, tier_for)

        # Publication dates, cheapest source first: whatever the synthesis stage
        # stated, else a year sitting in the URL. Needed because a tier is a
        # relation between a source and a claim, not a property of a domain — the
        # same newspaper is primary evidence for the week it was printed and a
        # secondary account of everything else.
        declared_dates = final.get("source_dates")
        declared_dates = declared_dates if isinstance(declared_dates, dict) else {}

        def published_for(url: str) -> Tuple[Optional[str], Optional[int]]:
            stated = declared_dates.get(url)
            if isinstance(stated, dict):  # tolerate {"tier":…, "published":…}
                stated = stated.get("published")
            year = sp.parse_year(stated) if stated else None
            if year is None:
                year = sp.url_year(url)
            return (str(stated) if stated else (str(year) if year else None)), year

        def rank_urls(urls: List[str], *, claim_year: Any, precision: Any) -> Dict[str, int]:
            """Rank each citation UNDER THIS CLAIM. The same URL can differ per node."""
            ranks: Dict[str, int] = {}
            for u in urls or []:
                _, pub_year = published_for(u)
                ranks[u] = sp.rank_for(
                    tier_for(u),
                    published_year=pub_year,
                    claim_year=claim_year if isinstance(claim_year, int) else None,
                    claim_precision=precision,
                    url=u,
                )
            return ranks

        def to_citations(urls: List[str], ranks: Optional[Dict[str, int]] = None) -> List[Citation]:
            out: List[Citation] = []
            ranks = ranks or {}
            for u in urls or []:
                meta = url_lookup.get(u, {"url": u})
                url = meta.get("url", u)
                rank = ranks.get(u, sp.EVIDENCE_RANK.get(tier_for(url), sp.WORST_RANK))
                published, _ = published_for(url)
                out.append(
                    Citation(
                        url=url,
                        title=meta.get("title"),
                        tier=tier_for(url),
                        tier_rank=rank,
                        role=sp.role_for(rank),
                        published=published,
                        chain=chain_of.get(url),
                    )
                )
            return out

        origin_data = final.get("origin") or {}
        origin_conf = float(origin_data.get("confidence", 0.5) or 0.5)
        origin_cites = origin_data.get("citations", []) or []
        origin_ranks = rank_urls(
            origin_cites,
            claim_year=origin_data.get("year"),
            precision=origin_data.get("precision", "unknown"),
        )
        origin = OriginResult(
            id="origin",
            year=origin_data.get("year"),
            era_label=origin_data.get("era_label"),
            precision=origin_data.get("precision", "unknown"),
            year_end=origin_data.get("year_end"),
            # The origin is the earliest surviving piece of evidence. An origin the model
            # typed as something non-surviving is defaulted to "text" rather than dropped,
            # because a trace with no origin has nothing to hang the chain from; the gate
            # below is what keeps the chain itself clean.
            node_type=(origin_data.get("node_type") if is_evidence_kind(origin_data.get("node_type")) else "text"),
            attribution=_attribution(origin_data.get("attribution")),
            source_title=origin_data.get("source_title", "Unknown"),
            summary=origin_data.get("summary", ""),
            citations=to_citations(origin_cites, origin_ranks),
            confidence=origin_conf,
            evidence=_coerce_dossier(
                origin_data.get("evidence"),
                fallback_claim=origin_data.get("summary", ""),
                confidence=origin_conf,
                read_urls=read_urls,
                citations=origin_cites,
                citation_ranks=origin_ranks,
            ),
        )

        timeline: List[TimelineEvent] = []
        # Steps the model offered that are not surviving objects. They are not thrown
        # away — they become conclusions, stated after the chain, which is where a
        # reading of the evidence belongs. Dropping them would hide that the model
        # believes something the evidence does not show.
        demoted: List[Dict[str, Any]] = []
        used_ids = {"origin"}
        for i, entry in enumerate(final.get("timeline") or []):
            if not is_evidence_kind(entry.get("node_type")):
                demoted.append(entry)
                continue
            conf = float(entry.get("confidence", 0.5) or 0.5)
            # Ids come from the model so connections can reference them, but must
            # be unique and present — a duplicate id would silently reroute edges.
            raw_id = str(entry.get("id") or "").strip()
            event_id = raw_id if raw_id and raw_id not in used_ids else f"t{i + 1}"
            while event_id in used_ids:
                event_id = f"{event_id}_{i + 1}"
            used_ids.add(event_id)
            entry_cites = entry.get("citations", []) or []
            entry_ranks = rank_urls(
                entry_cites,
                claim_year=entry.get("year"),
                precision=entry.get("precision", "unknown"),
            )
            timeline.append(
                TimelineEvent(
                    id=event_id,
                    year=entry.get("year"),
                    era_label=entry.get("era_label"),
                    precision=entry.get("precision", "unknown"),
                    year_end=entry.get("year_end"),
                    node_type=entry.get("node_type"),
                    attribution=_attribution(entry.get("attribution")),
                    source_title=entry.get("source_title", "Unknown"),
                    claim=entry.get("claim", ""),
                    citations=to_citations(entry_cites, entry_ranks),
                    confidence=conf,
                    evidence=_coerce_dossier(
                        entry.get("evidence"),
                        fallback_claim=entry.get("claim", ""),
                        confidence=conf,
                        read_urls=read_urls,
                        citations=entry_cites,
                        citation_ranks=entry_ranks,
                    ),
                )
            )

        # Chronological sort, oldest first; null years go last. Ids were assigned
        # before this so they survive the reordering and connections stay valid.
        timeline.sort(key=lambda e: (e.year is None, e.year if e.year is not None else 0))

        conclusions = _build_conclusions(final.get("conclusions"), demoted, valid_ids=used_ids)

        connections = self._build_connections(
            raw=final.get("connections"),
            valid_ids=used_ids,
            to_citations=to_citations,
            read_urls=read_urls,
            rank_urls=rank_urls,
            node_types={e.id: e.node_type for e in timeline} | {"origin": origin.node_type},
        )

        # The trace-level source list. Every citation attached to a NODE carries
        # its rank and its role, and this list — the one the UI renders as
        # "sources" — carried neither, so a reader looking at it saw a hundred
        # and thirty-seven undifferentiated links with the leads and the
        # manuscript repositories side by side. A trace whose whole argument is
        # that the tier matters cannot present its own bibliography as a flat
        # list. Ranked without a claim to be contemporary with, since a source
        # list is not an argument about any one moment.
        all_citations = []
        for c in citations:
            url = c.get("url")
            if not url:
                continue
            rank = sp.rank_for(tier_for(url), url=url)
            published, _ = published_for(url)
            all_citations.append(
                Citation(
                    url=url,
                    title=c.get("title"),
                    tier=tier_for(url),
                    tier_rank=rank,
                    role=sp.role_for(rank),
                    published=published,
                    chain=chain_of.get(url),
                )
            )

        snap = usage.snapshot()
        return TraceResponse(
            title=req.title,
            normalized_title=normalized,
            origin=origin,
            timeline=timeline,
            connections=connections,
            reasoning=final.get("reasoning", ""),
            confidence=float(final.get("confidence", 0.5) or 0.5),
            queries_run=queries_run,
            citations=all_citations,
            chains=[EvidenceChain(**g) for g in chain_groups],
            independent_chain_count=len(chain_groups),
            sources_read=sorted(read_urls),
            open_questions=open_questions,
            conclusions=conclusions,
            usage=TokenUsage(
                input_tokens=snap["input_tokens"],
                output_tokens=snap["output_tokens"],
                total_tokens=snap["total_tokens"],
                llm_calls=snap["llm_calls"],
                pages_read=len(read_urls),
                by_stage=snap["by_stage"],
            ),
            iterations=iterations,
            duration_seconds=round(duration, 2),
        )

    def _build_connections(
        self,
        *,
        raw: Any,
        valid_ids: set,
        to_citations: Callable[..., List[Citation]],
        read_urls: set,
        rank_urls: Optional[Callable[..., Dict[str, int]]] = None,
        node_types: Optional[Dict[str, str]] = None,
    ) -> List[Connection]:
        """Validate, then grade, the edges the model proposed."""
        cleaned = ev.validate_connections(
            raw,
            valid_ids,
            max_connections=self.settings.chrono_max_connections,
            node_types=node_types,
        )
        out: List[Connection] = []
        for item in cleaned:
            cites = item.get("citations", []) or []
            # An edge is a claim about a relationship, not about a moment, so its
            # citations are ranked without a date to be contemporary with.
            ranks = rank_urls(cites, claim_year=None, precision=None) if rank_urls else {}
            try:
                out.append(
                    Connection(
                        from_id=item["from_id"],
                        to_id=item["to_id"],
                        relation=item["relation"],
                        evidence=_coerce_connection_evidence(
                            item.get("evidence"),
                            read_urls=read_urls,
                            citations=cites,
                            citation_ranks=ranks,
                        ),
                        citations=to_citations(cites, ranks),
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Skipping malformed connection: %s", exc)
        return out

    # ------------------------------------------------------------------ expand
    def expand(self, req: "ExpandRequest") -> "ExpandResponse":  # type: ignore[name-defined]
        """Expand a single timeline item into finer-grained, chronologically ordered sub-events."""
        # Local import to avoid a circular import at module load time.
        from ..models import ExpandRequest, ExpandResponse  # noqa: F401

        started = time.time()
        usage.start()

        when = (
            f"{req.parent_year} ({'BCE' if req.parent_year < 0 else 'CE'})"
            if isinstance(req.parent_year, int)
            else (req.parent_era_label or "unknown")
        )
        context_clause = f" (context: {req.context})" if req.context else ""
        parent_id = (req.parent_id or "parent").strip() or "parent"

        # Stage 1 - grounded search around the anchor.
        usage.stage("search")
        search_prompt = EXPAND_SEARCH_PROMPT.format(
            story_title=req.story_title,
            context_clause=context_clause,
            when=when,
            parent_source_title=req.parent_source_title,
            parent_claim=req.parent_claim or "(no prior claim recorded)",
            search_doctrine=SEARCH_DOCTRINE,
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
                connections=[],
                queries_run=list(answer.queries or []),
                citations=[
                    Citation(url=c["url"], title=c.get("title"), tier=_default_tier(c["url"]))
                    for c in citations
                    if c.get("url")
                ],
                duration_seconds=round(time.time() - started, 2),
            )

        # Stage 2 - read the best source behind this anchor, same reasoning as the
        # main trace: an expansion that cannot cite read text is asserting, not tracing.
        want_reads = self.settings.chrono_expand_read_sources
        to_read = ev.select_for_reading(
            citations, [], lambda u: _default_tier(u), limit=want_reads * 2
        )
        reads = read_best(to_read, want=want_reads, max_chars=self.settings.chrono_read_chars)
        read_urls = {r.url for r in reads if r.ok}
        pages_block = ""
        if reads:
            pages_block = (
                "\nSOURCE PAGES actually read (prefer these over any summary of them):\n"
                f"{format_reads_block(reads)}\n"
            )

        # Stage 3 - extract structured sub-events plus their links to the anchor.
        usage.stage("extract")
        extract_prompt = EXPAND_EXTRACT_PROMPT.format(
            story_title=req.story_title,
            when=when,
            parent_id=parent_id,
            parent_source_title=req.parent_source_title,
            parent_claim=req.parent_claim or "(no prior claim recorded)",
            notes=answer.text,
            pages_block=pages_block,
            citations_block=_format_citations_block(citations),
            extract_doctrine=EXTRACT_DOCTRINE,
            max_events=req.max_events,
        )
        try:
            # Light path — same reason as _extract: mechanical event extraction, and
            # the reasoning path returns empty JSON on large note prompts.
            data = self.client.reason_json(extract_prompt, use_reasoning_model=False)
            raw_events = data.get("events") or []
            raw_connections = data.get("connections") or []
        except Exception as exc:
            logger.warning("Expand extract failed: %s", exc)
            raw_events, raw_connections = [], []

        url_lookup = {c["url"]: c for c in citations if c.get("url")}
        chain_of, _ = ev.build_chains(citations, [], lambda u: _default_tier(u))

        def to_citations(urls: List[str]) -> List[Citation]:
            out: List[Citation] = []
            for u in urls or []:
                meta = url_lookup.get(u, {"url": u})
                url = meta.get("url", u)
                out.append(
                    Citation(
                        url=url,
                        title=meta.get("title"),
                        tier=_default_tier(url),
                        chain=chain_of.get(url),
                    )
                )
            return out

        events: List[TimelineEvent] = []
        used_ids = {parent_id}
        for i, entry in enumerate(raw_events[: req.max_events]):
            try:
                conf = float(entry.get("confidence", 0.5) or 0.5)
                raw_id = str(entry.get("id") or "").strip()
                event_id = raw_id if raw_id and raw_id not in used_ids else f"e{i + 1}"
                while event_id in used_ids:
                    event_id = f"{event_id}_{i + 1}"
                used_ids.add(event_id)
                entry_cites = entry.get("citations", []) or []
                events.append(
                    TimelineEvent(
                        id=event_id,
                        year=entry.get("year"),
                        era_label=entry.get("era_label"),
                        precision=entry.get("precision", "unknown"),
                        source_title=entry.get("source_title", "Unknown"),
                        claim=entry.get("claim", ""),
                        citations=to_citations(entry_cites),
                        confidence=conf,
                        evidence=_coerce_dossier(
                            entry.get("evidence"),
                            fallback_claim=entry.get("claim", ""),
                            confidence=conf,
                            read_urls=read_urls,
                            citations=entry_cites,
                        ),
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Skipping malformed expand event: %s", exc)

        events.sort(key=lambda e: (e.year is None, e.year if e.year is not None else 0))

        connections = self._build_connections(
            raw=raw_connections,
            valid_ids=used_ids,
            to_citations=to_citations,
            read_urls=read_urls,
        )

        snap = usage.snapshot()
        return ExpandResponse(
            parent_source_title=req.parent_source_title,
            parent_year=req.parent_year,
            parent_era_label=req.parent_era_label,
            events=events,
            connections=connections,
            queries_run=list(answer.queries or []),
            citations=[
                Citation(
                    url=c["url"],
                    title=c.get("title"),
                    tier=_default_tier(c["url"]),
                    chain=chain_of.get(c["url"]),
                )
                for c in citations
                if c.get("url")
            ],
            sources_read=sorted(read_urls),
            usage=TokenUsage(
                input_tokens=snap["input_tokens"],
                output_tokens=snap["output_tokens"],
                total_tokens=snap["total_tokens"],
                llm_calls=snap["llm_calls"],
                pages_read=len(read_urls),
                by_stage=snap["by_stage"],
            ),
            duration_seconds=round(time.time() - started, 2),
        )
