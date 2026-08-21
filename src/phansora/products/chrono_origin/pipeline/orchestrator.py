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
import re
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
from .dated_list import parse_dated_list
from .prompts import (
    RESEARCH_PROMPT,
    EXPAND_EXTRACT_PROMPT,
    EXPAND_DOCTRINE,
    EXPAND_SEARCH_PROMPT,
    expand_mode,
    format_existing_block,
    SEARCH_DOCTRINE,
    SYNTHESIZE_PROMPT,
)
from .reader import PageRead, format_reads_block, read_best

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


def _as_year(value: Any) -> Optional[int]:
    """A year from whatever the model emitted, or None if there genuinely is not one.

    Strict ``isinstance(value, int)`` was rejecting dates that are perfectly good:
    a JSON string ``"-400"``, or ``-400.0`` where a range midpoint had been divided.
    TimelineEvent coerces both, but the gate below runs on the raw dict and never got
    that far — so a real, dated step was demoted into `conclusions` and told it was
    "not a surviving object", which is neither true nor the reason.
    """
    if isinstance(value, bool):  # bool is an int subclass; a flag is not a year
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+", value.strip())
        if m:
            return int(m.group(0))
    return None


def _sort_key(year: Optional[int], year_end: Optional[int]):
    """Order by when a thing STARTS, falling back to when it ends.

    A corpus composed across centuries is asked for as a span, and prompts.py tells
    the model to give exactly that for the works that open a chain. Keying on `year`
    alone sent any step carrying only `year_end` to the BOTTOM of the timeline — the
    oldest material in the trace, sorted last.
    """
    start = year if year is not None else year_end
    return (start is None, start if start is not None else 0)


def is_node_kind(value: Any) -> bool:
    """Is this one of the labels TimelineEvent.node_type will accept?

    A label for display, not a test an item has to pass. It matters only because
    node_type is a pydantic Literal: an unrecognised string raises, and one odd label
    would take a whole trace down rather than costing one entry its icon.
    """
    return value in _EVIDENCE_KINDS or value == "event"


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


def _warn_if_copy_without_work(origin: Any, timeline: List[Any]) -> None:
    """A chain that starts at a copy has dropped the thing being copied.

    prompts.py states the rule: if the chain contains a manuscript, a scroll or a
    fragment, the work it carries is itself a step, dated by COMPOSITION and placed
    earlier. Nothing enforced it, and the failure is invisible — the trace looks
    complete, every step is real, and the reader has no way to see that the oldest
    half is missing. A trace of Jesus opened at the Dead Sea Scrolls: the copies were
    there, the scriptures they are copies OF were not.

    Warns rather than mutates. The earlier step has to come from the research; this
    cannot invent one, and inventing one is precisely what the product must not do.
    """
    head = origin if origin is not None else (timeline[0] if timeline else None)
    if head is None:
        return
    kind = getattr(head, "node_type", None)
    if kind not in ("manuscript", "scroll"):
        return
    start = _as_year(getattr(head, "year", None))
    if start is None:
        start = _as_year(getattr(head, "year_end", None))
    if start is None:
        return
    for e in timeline:
        other = _as_year(getattr(e, "year", None))
        if other is None:
            other = _as_year(getattr(e, "year_end", None))
        if other is not None and other < start:
            return
    logger.warning(
        "Chain starts at a copy: %r is a %s and nothing in the chain predates it. "
        "The work it carries should be an earlier step, dated by composition.",
        getattr(head, "source_title", "?"), kind,
    )


def _build_conclusions(raw: Any, *, valid_ids: set) -> List[Conclusion]:
    """The readings of the evidence, stated after it.

    Only the model's own conclusions now. Steps used to be demoted in here — anything
    that was not a "surviving object", and anything undated — which was the code half
    of a judgement the synthesis prompt was also making in prose. Both are gone; what
    research finds goes on the timeline.
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



def _format_citations_block(citations: List[Dict[str, str]]) -> str:
    """Every source gathered, in full.

    This used to stop at 60 and say how many it had swallowed. The ceiling was written
    for a pipeline that ran three rounds of six searches and could arrive here with
    ~144 URLs; there are no rounds and no fan-out now, and one research call returns a
    bounded 15-33. Synthesis is told to cite from this list, so anything missing from
    it is a source the trace cannot attribute a claim to.
    """
    if not citations:
        return "(none)"
    return "\n".join(
        f"[{i}] {c.get('title') or c.get('url')} -> {c.get('url')}"
        for i, c in enumerate(citations, 1)
    )




def build_research_prompt(title: str, context: Optional[str]) -> str:
    """The exact text handed to the research call.

    RESEARCH_PROMPT is tuned by hand against the live model, so this assembles AROUND it
    rather than editing it. `{context_clause}` was dropped from the template during one
    of those retunes, and str.format() ignores a keyword the template does not use — so
    no error was raised anywhere and the context box on the dashboard simply stopped
    reaching the model. A trace of "Mercury" then researched whichever Mercury the model
    felt like, while the context still partitioned the cache key, so the two readings got
    two cache entries of the same wrong answer.

    Appending restores the field without editing prose somebody tuned on purpose. With no
    context given the prompt is byte-identical to the template; the check against `prompt`
    keeps this from doubling up if `{context_clause}` is ever put back.
    """
    prompt = RESEARCH_PROMPT.format(
        title=title,
        context_clause=f" ({context})" if context else "",
    )
    if context and context not in prompt:
        prompt = (
            f"{prompt.rstrip()}\n\n"
            f'For this trace, "{title}" means: {context}. Research that subject and no other.'
        )
    return prompt


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
            language=req.language,
        )

        progress(2, "Checking cache")
        cached = get_cached(req.title, key)
        if cached:
            logger.info("Cache hit for %s", normalized)
            progress(100, "Loaded from cache")
            return TraceResponse(**cached)


        all_citations: Dict[str, Dict[str, str]] = {}
        queries_run: List[str] = []
        iterations = 0

        # Stage 1 - Gather. ONE grounded call; the model runs its own searches.
        #
        # There is no planner call and no round loop. Both existed to make the
        # searching adaptive, and adaptivity cost far more than it bought: measured on
        # a live trace, a round's six searches finished in 14 seconds and the call
        # that turned them into structured JSON took 104 — every round, overrunning
        # its output budget and regenerating from scratch. Synthesis then overran the
        # doubled budget and the trace failed outright at 20 minutes. Generating JSON
        # is the expensive act, so it happens exactly once.
        #
        # There is no seven-way fan-out either. That was written for a provider with
        # no search of its own: the queries had to be guessed in advance and fired
        # blind, one per evidence category, to guarantee coverage. It also decided
        # where every chain STARTED, and decided it wrongly — six of the seven asked
        # what survives ABOUT the subject and exactly one asked what its evidence
        # DESCENDS FROM, so the half the chain begins with was outvoted six to one in
        # every corpus. A trace of Jesus opened at the Dead Sea Scrolls and lost the
        # four centuries of scripture the Scrolls are copies OF.
        #
        # The model now chooses and runs its own queries, so there are no slots left
        # to allocate badly. RESEARCH_PROMPT asks for descent first and evidence about
        # the subject second, in one answer.
        progress(15, "Searching")
        research_prompt = build_research_prompt(req.title, req.context)
        answers = self._search_prompts([research_prompt])
        # What the model actually searched for, reported back so the user sees the
        # real queries rather than a template we wrote.
        for answer in answers:
            queries_run.extend(answer.queries or [])
        iterations = 1

        # Citations if the model grounded, kept at the TOP LEVEL only. Nothing on this
        # path reads them — a node is a title and a date — but throwing away sources a
        # search already paid for would turn "add sources back" into a research problem
        # instead of a display one.
        for answer in answers:
            for c in (answer.citations or []):
                url = c.get("url") if isinstance(c, dict) else None
                if url and url not in all_citations:
                    all_citations[url] = c

        text = "\n\n".join(a.text.strip() for a in answers if (a.text or "").strip())
        if not text.strip():
            raise RuntimeError(
                "The research call came back empty, so there was no list to read. The "
                "credit has been refunded. Please try again in a moment."
            )

        # Stage 2 - Read the list. NOT a model call.
        #
        # The research prompt asks for `Title - Date`, one per line, oldest first, and
        # that is already the timeline. Sending it to a second model to be turned into
        # JSON was the single most expensive and least reliable step in the trace: it
        # overran its 32000-token budget four times in three days, each time failing a
        # trace whose research had already succeeded, and being a model it could also
        # shorten a list it found long or drop an item it found odd. A parser does none
        # of that. See dated_list.
        progress(80, "Reading the list")
        items = parse_dated_list(text)
        if len(items) < 2:
            raise RuntimeError(
                "The research answer did not contain a readable list of dated items, so "
                "there is no timeline to show. The credit has been refunded. Please try "
                "again in a moment."
            )

        progress(97, "Building response")
        response = self._build_list_response(
            req=req,
            normalized=normalized,
            items=items,
            citations=list(all_citations.values()),
            queries_run=queries_run,
            iterations=iterations,
            duration=time.time() - started,
        )

        # An empty trace is a FAILED trace, and must not be stored or cached as a
        # success. A synthesize step that returned nothing usable still produces a
        # well-formed TraceResponse — every field simply takes its default — so the
        # result looked complete: origin with no year, no summary, no citations, and an
        # empty timeline, marked done, with the user's credit spent. Raising here sends
        # it down the failure path that already exists, which refunds.
        #
        # Gated on the timeline ALONE. The three-way `and` this replaces required an
        # empty timeline AND a null year AND an empty summary — but a model with nothing
        # to report does not leave the summary empty, it writes "No research material was
        # provided", which is prose, which is truthy. So a trace with zero events was
        # cached and charged as a success. The timeline IS the product: if it is empty
        # the trace failed, whatever the origin block manages to say about it.
        if not response.timeline:
            raise RuntimeError(
                "The trace produced an empty timeline. This usually means the model's "
                "answer could not be read; the credit has been refunded."
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

    def _search_prompts(self, prompts: List[str]) -> List[GroundedAnswer]:
        """Run already-formatted search prompts concurrently.

        The gather and expand stages build their own prompt; the chase writes
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

    # ------------------------------------------------------- PARKED: the synthesis path
    # Nothing below reaches _synthesize, _build_response, _mention_tiers or
    # _build_connections any more. A trace is one model call and a parser; these turned
    # a corpus into claims, dossiers, tiers and evaluated connections, and they are kept
    # because that is the direction the product is going back in, one piece at a time —
    # descriptions first, then sources.
    #
    # They are NOT dead-but-harmless: re-enabling any of them means paying for the
    # second model call again, with the token ceiling that failed four traces in three
    # days. Bring one back deliberately, with the budget it needs, not by wiring it
    # into run() because it happens to be here.
    def _synthesize(
        self,
        *,
        title: str,
        corpus: str,
        citations: List[Dict[str, str]],
        reads: List[PageRead],
    ) -> Dict[str, Any]:
        pages_block = ""
        if reads:
            pages_block = (
                "\nSOURCE PAGES actually read (prefer these over any summary of them):\n"
                f"{format_reads_block(reads)}\n"
            )
        prompt = SYNTHESIZE_PROMPT.format(
            title=title,
            mentions_block=corpus,
            citations_block=_format_citations_block(citations),
            pages_block=pages_block,
            max_connections=self.settings.chrono_max_connections,
        )
        return self.client.reason_json(prompt, use_reasoning_model=True)

    # --------------------------------------------------------------- response
    def _build_list_response(
        self,
        *,
        req: TraceRequest,
        normalized: str,
        items: List[Any],
        citations: List[Dict[str, str]],
        queries_run: List[str],
        iterations: int,
        duration: float,
    ) -> TraceResponse:
        """A timeline of titles and dates, and nothing else.

        Every other field keeps its default rather than being filled with something
        invented. An empty `claim` says we have no description of this item, which is
        true; a generated one would say we do. The fields are still there, so adding
        descriptions or sources later is a matter of filling them in — see _synthesize
        and _build_response, which stay for exactly that and are not on this path.
        """
        events = [
            TimelineEvent(
                id=f"t{i + 1}",
                year=item.year,
                year_end=item.year_end,
                era_label=item.era_label,
                precision=item.precision,
                node_type="event",
                attribution="not_applicable",
                source_title=item.title,
                claim="",
                citations=[],
                confidence=0.5,
                evidence=None,
            )
            for i, item in enumerate(items)
        ]
        # The model is asked for oldest-first and gives it, but a list that came back a
        # little out of order should be shown in order rather than shown wrong. Ids were
        # assigned before this, so they survive the sort.
        events.sort(key=lambda e: _sort_key(e.year, e.year_end))

        # The oldest item IS the origin — that is what the trace set out to find. It
        # moves out of the timeline rather than being copied, so the board does not show
        # the same item twice.
        first = events[0]
        origin = OriginResult(
            id="origin",
            year=first.year,
            year_end=first.year_end,
            era_label=first.era_label,
            precision=first.precision,
            node_type=first.node_type,
            attribution=first.attribution,
            source_title=first.source_title,
            summary="",
            citations=[],
            confidence=first.confidence,
        )

        return TraceResponse(
            title=req.title,
            normalized_title=normalized,
            origin=origin,
            timeline=events[1:],
            connections=[],
            reasoning="",
            confidence=0.5,
            queries_run=queries_run,
            citations=[Citation(**c) for c in citations if c.get("url")],
            usage=TokenUsage(**usage.snapshot(), pages_read=0),
            iterations=iterations,
            duration_seconds=duration,
        )

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
            node_type=(origin_data.get("node_type") if is_node_kind(origin_data.get("node_type")) else "event"),
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
        # Nothing is demoted. Two gates used to stand here — one rejecting anything that
        # was not one of nine "surviving object" kinds, one rejecting anything undated —
        # and between them they enforced, in code, the same judgement the synthesis
        # prompt was making in prose. Two stages adjudicating with differently worded
        # rules meant they disagreed, and the code always won: research named the Hebrew
        # scriptures in six runs out of six and no trace ever showed them, because a work
        # is not an object you can name a shelfmark for.
        #
        # The judgement happens once now, in the research prompt, where the searching is.
        # What comes back goes on the board.
        used_ids = {"origin"}
        for i, entry in enumerate(final.get("timeline") or []):
            year, year_end = _as_year(entry.get("year")), _as_year(entry.get("year_end"))
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
                    year=year,
                    era_label=entry.get("era_label"),
                    precision=entry.get("precision", "unknown"),
                    year_end=year_end,
                    # A label, not a test. An unknown one becomes "event" rather than
                    # raising — node_type is a Literal, so an odd label would otherwise
                    # take the entire trace down with it.
                    node_type=(entry.get("node_type") if is_node_kind(entry.get("node_type")) else "event"),
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
        timeline.sort(key=lambda e: _sort_key(e.year, e.year_end))

        # If the model's origin has no date and the chain does, the two swap places —
        # ids included, so "origin" keeps naming the first step and connections still
        # resolve. The model picked the right object last time and simply left the date
        # off it, which is enough to make a trace look like it starts centuries late.
        if origin.year is None and origin.year_end is None and timeline:
            first = timeline.pop(0)
            displaced = TimelineEvent(**{**origin.model_dump(), "id": first.id, "claim": origin.summary})
            origin = OriginResult(**{**first.model_dump(), "id": "origin", "summary": first.claim})
            timeline.append(displaced)
            timeline.sort(key=lambda e: _sort_key(e.year, e.year_end))

        _warn_if_copy_without_work(origin, timeline)

        conclusions = _build_conclusions(
            final.get("conclusions"), valid_ids=used_ids
        )


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
        # The axis this expansion was asked for. Unaimed, an expansion mostly returns
        # the anchor's own neighbours — which are already on the board.
        mode = expand_mode(req.mode)
        existing_block = format_existing_block(req.existing)

        search_prompt = EXPAND_SEARCH_PROMPT.format(
            story_title=req.story_title,
            context_clause=context_clause,
            when=when,
            parent_source_title=req.parent_source_title,
            parent_claim=req.parent_claim or "(no prior claim recorded)",
            search_doctrine=SEARCH_DOCTRINE,
            mode_search=mode["search"],
            mode_query=mode["query"],
            existing_block=existing_block,
        )
        try:
            answer = self.client.grounded_search(search_prompt)
        except Exception as exc:
            logger.warning("Expand grounded search failed: %s", exc)
            answer = GroundedAnswer(text="", citations=[], queries=[])

        # No citations at all means the search itself came back empty — throttled,
        # blocked, or down — and everything after this is extraction over nothing.
        # Reported rather than inferred: a silent empty corpus is indistinguishable
        # from a subject with no evidence, and that ambiguity cost hours.
        search_unavailable = not (answer.citations or (answer.text or "").strip())
        if search_unavailable:
            logger.warning(
                "Expand: the search returned no results for %r; nothing to extract from.",
                req.parent_source_title,
            )

        citations: List[Dict[str, str]] = []
        seen_urls: set[str] = set()
        for c in answer.citations:
            url = c.get("url") or ""
            if url and url not in seen_urls:
                seen_urls.add(url)
                citations.append(c)

        if not answer.text:
            return ExpandResponse(
                search_unavailable=search_unavailable,
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
            expand_doctrine=EXPAND_DOCTRINE,
            max_events=req.max_events,
            mode_label=mode["label"],
            mode_extract=mode["extract"],
            existing_block=existing_block,
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
                # The same two gates the trace itself applies. Expanding had neither, so
                # a node could sprout children the chain rule would have refused — an
                # undated press headline came back as a step under the Dead Sea Scrolls.
                # A branch is part of the timeline; it is held to the timeline's rules.
                # A branch may be an event; the chain may not. Expanding explains a
                # node, and "in 1947 shepherds found jars in a cave" is the true answer
                # to how the scrolls were discovered — refusing it because a shepherd is
                # not an artefact is what made three of the six modes unanswerable.
                kind = entry.get("node_type")
                if not (is_evidence_kind(kind) or kind == "event"):
                    logger.info("Expand: dropped unusable kind %r for %r", kind, entry.get("source_title"))
                    continue
                if not isinstance(entry.get("year"), int) and not isinstance(entry.get("year_end"), int):
                    logger.info("Expand: dropped undated %r", entry.get("source_title"))
                    continue
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
