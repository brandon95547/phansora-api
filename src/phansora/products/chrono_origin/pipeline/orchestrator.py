"""The trace pipeline: research -> synthesize -> build response.

Two principles govern the shape of this file.

TOKENS BUY JUDGEMENT, NOT REPETITION. Anything deterministic — deduplicating
mentions, deciding which sources are one source repeated, turning an evidence
type into a claim class — happens in ``evidence.py`` for free. The model is asked
only for what it alone can do: search, read what it finds, and weigh it.

ONE PLACE DECIDES. ``RESEARCH_PROMPT`` holds every rule about what qualifies as a
step in a chain of descent. Synthesis is a FORMATTER: it maps what the research
returned onto the timeline schema and does not re-judge it. There used to be a
second set of rules downstream, worded differently, and the two disagreed — the
downstream one silently won, which is why work-level entries (the Hebrew
scriptures, say) never reached a timeline that its own research had named.

A trace is one grounded call. The provider's model searches for itself and the
citations come back attached to the answer, so there is no separate search,
decompose or read stage here. ``expand()`` is the exception and still reads pages
directly: it works from an item already on the board rather than from a search.
"""
from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple, get_args

from ..config import get_settings
from ..models import (
    Citation,
    DetailField,
    Connection,
    ConnectionEvidence,
    EvidenceDossier,
    ExpandRequest,
    ExpandResponse,
    OriginResult,
    RelationType,
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
from .dated_list import SIGNIFICANCE_LABEL, parse_dated_list
from .prompts import (
    RESEARCH_PROMPT,
    expand_extract,
    EXPAND_PROMPT,
    expand_body,
    expand_mode,
    part_of_line,
)
from phansora.shared.ai.json_repair import parse_json_loose, repair_truncated_json

logger = logging.getLogger(__name__)


ProgressCallback = Callable[[int, str], None]


def _noop_progress(_percent: int, _stage: str) -> None:  # pragma: no cover
    return None


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


_EVIDENCE_KINDS = {
    "text", "manuscript", "scroll", "letter", "inscription",
    "document", "record", "artifact", "archaeological_find",
}


def _claim_of(item: Any) -> str:
    """The item's significance, which is the one detail that is a statement about it.

    Everything else the research returns — where it is from, what it is made of, who
    made it — is a property, and properties belong in the table. Empty when the research
    did not say: a node with no claim is telling the truth about what came back.
    """
    for label, value in item.details:
        if label == SIGNIFICANCE_LABEL:
            return value
    return ""


def _details_of(item: Any) -> List[DetailField]:
    """Everything else, in the order the research gave it."""
    return [
        DetailField(label=label, value=value)
        for label, value in item.details
        if label != SIGNIFICANCE_LABEL
    ]


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


# ------------------------------------------------------------------- expansions
# An expansion answers in six flat fields — name, year, group, relation, shared, url —
# and the board wants a graded TimelineEvent with an edge hanging off the anchor. Every
# difference between the two lives here, not in the prompt: the prompt is the part a
# person tunes by hand, and it should not also have to carry a schema.
#
# `relation` is the model's own word for the strength of the link, and the vocabulary
# differs by axis — "direct_source" is the wrong word for a discovery record, "records"
# is the wrong word for a parallel. Each maps to the edge that asserts no MORE than the
# word does: a probable influence is not a demonstrated descent, so it draws as a
# retelling and not an arrow, and a disputed parallel asserts nothing but the
# resemblance. Anything unrecognised lands on no_established_link, the claim that costs
# the reader least if it is wrong.
# A word the board already knows is taken at face value. The first live answer came back
# saying "provides_context" — a real RelationType, and the honest one for what it had
# found — and a table keyed only on the prompt's own vocabulary turned it into
# "no_established_link". A prompt is a request, not a schema: an answer that lands in the
# target vocabulary by itself must not be downgraded for arriving early.
_BOARD_RELATIONS = frozenset(get_args(RelationType))

_RELATION_WORDS: Dict[str, str] = {
    "direct_source": "derives_from",
    "probable_influence": "retells",
    "possible_influence": "no_established_link",
    "independent_parallel": "retells",
    "disputed_parallel": "no_established_link",
    "records": "attests",
    "contemporaneous": "contemporaneous",
    "context": "provides_context",
}

# How sure the link is, from the same word. A flat 0.5 made confidence meaningless on
# branches — every card on the board said the same thing about itself.
_RELATION_CONFIDENCE: Dict[str, float] = {
    "direct_source": 0.8,
    "records": 0.75,
    "contemporaneous": 0.65,
    "probable_influence": 0.6,
    "independent_parallel": 0.5,
    "context": 0.5,
    "possible_influence": 0.45,
    "disputed_parallel": 0.35,
}


def _text_of_keys(entry: Dict[str, Any], *keys: str) -> str:
    """The first of these keys the model actually filled.

    Several keys per value because a prompt is a request, not a schema: a grounded call
    cannot be pinned to a response format, so an answer that says "source_title" where
    the template said "name" is a normal event rather than a failure.
    """
    for key in keys:
        val = entry.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            val = str(val)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


_YEAR_TEXT = re.compile(
    r"^\s*(?:c(?:irca|a)?\.?\s*)?(?:ad\s+)?(-?\d{1,5})\s*(bce|bc|b\.c\.(?:e\.)?|ce|ad)?\s*$",
    re.IGNORECASE,
)


def _truthy(value) -> bool:
    """A model's boolean, however it chose to spell it.

    A shaping call returns JSON, so this is usually a real bool — but the same field has
    come back as "true", "True" and 1 often enough that reading only `is True` would drop
    the flag silently, and a collection read as a single subject is the exact failure this
    field exists to fix.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    return str(value or "").strip().lower() in {"true", "yes", "1"}


def _year_of(value: Any) -> Optional[int]:
    """A signed year out of whatever the model wrote, or None.

    The gate below discards anything undated, which is right — a branch with no date has
    nowhere to sit — but it used to accept nothing except a JSON int, and that is not
    what a model asked for "the earliest defensible attestation date" reliably returns.
    "-2350", "2350 BCE" and 2350.0 are all dated answers, and all three were being thrown
    away one at a time in the log while the user counted the cards that did not arrive.

    source_policy.parse_year is not this function: it reads publication years off pages
    and cannot read a negative one at all.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if float(value).is_integer() else None
    if not isinstance(value, str):
        return None
    found = _YEAR_TEXT.match(value)
    if not found:
        return None
    year = int(found.group(1))
    era = (found.group(2) or "").lower()
    if era.startswith("b"):
        # "2350 BCE" and "-2350 BCE" mean the same thing; nobody writes the second to
        # mean 2350 CE.
        return -abs(year)
    return year


def adapt_expand_events(
    raw_events: Any,
    *,
    parent_id: str,
    max_events: int,
    to_citations,
) -> Tuple[List[TimelineEvent], List[Connection]]:
    """Flat answer in, (events, connections) out.

    Two gates, the same two the trace applies: an entry with no name is nothing, and an
    undated one has nowhere to sit on a timeline — dropped here rather than parked at
    year zero, which is both a real-looking date and not a year that exists.
    """
    events: List[TimelineEvent] = []
    connections: List[Connection] = []
    used_ids = {parent_id}
    entries = raw_events if isinstance(raw_events, list) else []
    cap = max(1, int(max_events or 1))
    if len(entries) > cap:
        # Loudly. This is the one loss with no trace on the board at all: the cards that
        # do arrive look like the whole answer.
        logger.warning(
            "Expand: the model returned %d items and the cap is %d — %d are being thrown away.",
            len(entries), cap, len(entries) - cap,
        )
    dropped: List[str] = []

    for i, entry in enumerate(entries[:cap]):
        if not isinstance(entry, dict):
            dropped.append(f"not an object: {entry!r:.80}")
            continue
        title = _text_of_keys(entry, "name", "source_title", "title")
        if not title:
            # The one thing still refused: a card with no title is not a result, and a
            # board full of "Untitled" is worse than a board missing one row. Named here
            # so it is not a silent loss.
            dropped.append(f"no name: {entry!r:.120}")
            continue

        year = _year_of(entry.get("year"))
        year_end = _year_of(entry.get("year_end"))
        era = _text_of_keys(entry, "era_label", "era", "period")
        # Undated used to be dropped here, on the rule that a step with no date has
        # nowhere to sit. True of the chain, and NOT true of a branch: branches are
        # stacked beside the node they hang from, and their position on screen owes
        # nothing to their year. So an undated find is shown, last, saying so — which is
        # a great deal more useful than a call that quietly returns four of nine.
        undated = year is None and year_end is None

        word = _text_of_keys(entry, "relation", "classification", "link").lower()
        relation = (
            word if word in _BOARD_RELATIONS
            else _RELATION_WORDS.get(word, "no_established_link")
        )
        confidence = _RELATION_CONFIDENCE.get(word, 0.5)
        shared = _text_of_keys(entry, "shared", "claim", "feature", "summary")
        disputed = word == "disputed_parallel"

        raw_urls = [_text_of_keys(entry, "url", "source_url")] + [
            u for u in (entry.get("citations") or []) if isinstance(u, str)
        ]
        urls: List[str] = []
        for u in raw_urls:
            u = (u or "").strip()
            if u and u not in urls:
                urls.append(u)

        event_id = f"e{i + 1}"
        while event_id in used_ids:
            event_id += "_"
        used_ids.add(event_id)

        # The model's own two words, kept verbatim on the card. They carry a distinction
        # the board's nine relations cannot: "disputed parallel" and "independent
        # parallel" draw the same line and are not the same finding.
        details: List[DetailField] = []
        group = _text_of_keys(entry, "group", "category")
        if group:
            details.append(DetailField(label="Category", value=group.replace("_", " ")))
        if word:
            details.append(DetailField(label="Classification", value=word.replace("_", " ")))

        # What each citation is worth under this claim. Without it every URL falls back to
        # WORST_RANK, verification comes out "unknown", and the dossier is forced to
        # "absent" — so a card standing on a museum catalogue graded exactly the same as
        # one standing on a forum post. The flat answer names no earliest supporting
        # source, which is the other half of that calculation, so the URL is all there is
        # to judge by and it has to actually be judged.
        ranks = {u: sp.rank_for(sp.default_tier(u), url=u) for u in urls}

        dispute_note = "Reported as a disputed comparison." if disputed else "None identified"
        dossier = {
            "claim": shared or title,
            "evidence_type": "disputed" if disputed else "scholarly_inference",
            "why": shared,
            "scholarly_dispute": dispute_note,
        }
        raw_kind = entry.get("node_type")

        events.append(
            TimelineEvent(
                id=event_id,
                year=year,
                year_end=year_end,
                era_label=era or ("Undated" if undated else None),
                precision="year" if year is not None else "unknown",
                node_type=raw_kind if is_node_kind(raw_kind) else "event",
                # A card produced by an expansion is itself a node the reader can expand,
                # so it has to carry the same judgement the trace makes. Read leniently:
                # the model returns a JSON bool, but a string "true" from a shaping call
                # that quoted it should not silently mean False.
                is_collection=_truthy(entry.get("is_collection")),
                source_title=title,
                claim=shared,
                citations=to_citations(urls),
                confidence=confidence,
                details=details,
                evidence=_coerce_dossier(
                    dossier,
                    fallback_claim=shared or title,
                    confidence=confidence,
                    citations=urls,
                    citation_ranks=ranks,
                ),
            )
        )
        connections.append(
            Connection(
                from_id=parent_id,
                to_id=event_id,
                relation=relation,
                citations=to_citations(urls),
                evidence=ConnectionEvidence(
                    mechanism=shared or f"{title} is offered as {word.replace('_', ' ') or 'related'}.",
                    scholarly_dispute=dispute_note,
                    evidence_type="disputed" if disputed else "scholarly_inference",
                    confidence=confidence,
                    confidence_label=(
                        "high" if confidence >= 0.75
                        else "moderate" if confidence >= 0.5
                        else "low" if confidence >= 0.3
                        else "speculative"
                    ),
                    why=shared,
                ),
            )
        )

    events.sort(key=lambda e: (e.year is None, e.year if e.year is not None else 0))

    # One record rather than twenty, so an expansion reads as a block in the terminal:
    # what the model found, what reached the board, and what did not.
    lines = [
        f"Expand: {len(entries)} item(s) in, {len(events)} on the board"
        + (f", {len(dropped)} refused" if dropped else "")
    ]
    by_id = {c.to_id: c for c in connections}
    for e in events:
        when = "undated" if e.year is None else str(e.year)
        rel = by_id[e.id].relation if e.id in by_id else "?"
        lines.append(f"    {when:>8}  {e.source_title[:52]:<52}  {rel}")
    lines.extend(f"    REFUSED   {d}" for d in dropped)
    logger.info("\n".join(lines))

    return events, connections


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
    # Substitution, not str.format(). The prompt now contains a JSON example, and every
    # brace in it is a format field as far as format() is concerned — the whole template
    # died on `KeyError: '\n    "title"'` the moment the output shape became an array.
    # Escaping them would work until the next retune pasted an unescaped one back in,
    # and this file is edited by hand far more often than that would survive.
    prompt = RESEARCH_PROMPT.replace("{title}", title).replace(
        "{context_clause}", f" ({context})" if context else ""
    )
    # str.format() ignores a keyword the template does not use, so a template that has
    # lost `{title}` raises nothing and simply researches whatever subject is written
    # into it — every trace, for every user, silently. That has now happened twice: to
    # `{context_clause}`, which quietly stopped the context box reaching the model, and
    # to `{title}` itself, hardcoded to one subject while a prompt was being tuned.
    # The prompt is edited far more often than this code is, so the check lives here.
    if title not in prompt:
        raise RuntimeError(
            f"RESEARCH_PROMPT never mentions {title!r}, so the trace would research "
            "whatever subject the template names instead. Put {title} back in "
            "prompts.py where the subject belongs."
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

        The claim carries whatever metadata the research returned with the item —
        origin, material, authorship, significance — labelled and in the model's own
        words. It is not judged, graded or rewritten on the way through: no second model
        call runs on this path, so what the research said is what the node says.

        Everything else keeps its default rather than being filled with something
        invented. Citations stay empty and evidence stays None until sources are the
        step being added. The synthesis pass that once filled them (claims, dossiers,
        graded connections) is kept at the git tag parked/chrono-synthesis.
        """
        events = [
            TimelineEvent(
                id=f"t{i + 1}",
                year=item.year,
                year_end=item.year_end,
                era_label=item.era_label,
                precision=item.precision,
                node_type="event",
                # The only path a trace takes, so the flag has to be set here: wired
                # only into the old synthesis path, it stayed false on every node.
                is_collection=item.is_collection,
                attribution="not_applicable",
                source_title=item.title,
                claim=_claim_of(item),
                details=_details_of(item),
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
            # The origin says what the timeline's first item said. It is the same node
            # promoted, so it carries the same research: `summary` is the origin's name
            # for `claim`, and the details table is the same table.
            summary=first.claim,
            citations=[],
            confidence=first.confidence,
            details=list(first.details),
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

    # ------------------------------------------------------------------ expand
    def expand(self, req: ExpandRequest) -> ExpandResponse:
        """Expand a single timeline item into finer-grained, chronologically ordered sub-events."""
        started = time.time()
        usage.start()

        parent_id = (req.parent_id or "parent").strip() or "parent"

        usage.stage("search")
        # The axis this expansion was asked for. Unaimed, an expansion mostly returns
        # the anchor's own neighbours — which are already on the board.
        mode = expand_mode(req.mode)

        # One call: it searches and answers in the same breath, so the body below is the
        # whole instruction. There is no summary in the middle to lose half of it, which
        # is how an axis aimed at earlier parallels used to come back with old documents.
        prompt = EXPAND_PROMPT.format(
            parent_source_title=req.parent_source_title,
            mode_query=mode["query"],
            part_of_line=part_of_line(req.parent_source_title, req.part_of),
            context_line=(
                f"\nThe subject is understood in this context: {req.context}\n"
                if req.context else ""
            ),
            mode_body=expand_body(mode, req.parent_source_title, req.parent_is_collection),
        )
        try:
            answer = self.client.grounded_search(prompt)
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

        # Shape the research in a SECOND call, which is the only way the first one gets to
        # search at all.
        #
        # A grounded call cannot be pinned to a response format — the API refuses a forced
        # JSON mime type alongside the search tool — so this used to ask for the shape in
        # the prompt instead and read whatever came back. That request is complied with:
        # measured against the live API, the same prompt searches at 1835 characters and
        # stops searching the moment "Return JSON only" is added, on both model tiers. So
        # every expansion was answered from the model's recall, and the JSON it produced was
        # well-formed and unsourced.
        #
        # reason_json CAN force the mime type, because it is not grounded and has nothing to
        # search. It runs on the chat model rather than the reasoning one: turning prose that
        # is already in the prompt into six flat fields is formatting work, and the research
        # it is shaping has already been paid for.
        #
        # The loose read below stays as the fallback. A forced mime type still leaves a tail
        # cut off at the token cap, and a half-parsed answer is worth more than none.
        # Asked twice before giving up, escalating to the reasoning model.
        #
        # An empty shape throws away research that has already been searched for and paid
        # for, which is the worst outcome available: the user is told the direction found
        # nothing when it found everything. It happened on prod with a grounded answer that
        # listed all nine books correctly — and the same research, resubmitted twenty-six
        # times across both tiers and with and without its citations, came back with nine
        # events every single time. So the cause is not the input and is not reproducible on
        # demand, which makes a retry the honest response: it is one ungrounded call against
        # a model that has already demonstrated it can do this.
        shape = expand_extract(
            req.parent_source_title, answer.text, _format_citations_block(citations),
        )
        data: Any = None
        for attempt, reasoning in enumerate((False, True)):
            try:
                data = self.client.reason_json(shape, use_reasoning_model=reasoning)
            except Exception as exc:  # noqa: BLE001 — fall through to the research itself
                logger.warning(
                    "Expand: shaping the research failed on attempt %d (%s).", attempt + 1, exc,
                )
                data = None
            if isinstance(data, dict) and data.get("events"):
                break
            # Said out loud. This was silent, and a silent empty shape is indistinguishable
            # from a search that found nothing — which is exactly how it was read.
            logger.warning(
                "Expand: the shaping call returned no events on attempt %d over %d characters "
                "of research for %r.",
                attempt + 1, len(answer.text or ""), req.parent_source_title,
            )
            data = None
        for candidate in (
            () if data is not None
            else (answer.text or "", repair_truncated_json(answer.text or "") or "")
        ):
            if not candidate:
                continue
            try:
                parsed = parse_json_loose(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                data = parsed
                break
        if data is None:
            logger.warning(
                "Expand: no readable answer for %r in %d characters.",
                req.parent_source_title, len(answer.text or ""),
            )
            data = {}
        raw_events = data.get("events") or []
        # The model's answer, verbatim, because every count below is a claim about it and
        # the only way to check a claim about an answer is to read the answer.
        logger.info(
            "Expand [%s] %r — the model answered with %d characters:\n%s",
            req.mode, req.parent_source_title, len(answer.text or ""), (answer.text or "").strip(),
        )
        read_urls: set = set()

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

        events, connections = adapt_expand_events(
            raw_events,
            parent_id=parent_id,
            max_events=req.max_events,
            to_citations=to_citations,
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
