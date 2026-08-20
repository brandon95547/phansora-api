"""Pydantic schemas for the Chrono-Origin API."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


DatePrecision = Literal["exact", "year", "decade", "century", "millennium", "era", "unknown"]

# Where a citation sits in the source hierarchy the trace works down. The pipeline
# ranks evidence by this, not by the outlet's brand: a claim is only as strong as
# the citation you can follow backwards from it.
SourceTier = Literal[
    "primary",          # tier 1 — the original / earliest surviving document, inscription, record
    "repository",       # tier 1 — the institution holding or publishing it: archive, museum, library
    "academic",         # tier 2 — peer-reviewed papers, university-press books, critical editions
    "reference_index",  # tier 3 — Crossref/DOI, catalogue records: metadata about a work, not the work
    "institutional",    # tier 4 — university, agency and museum write-ups; follow their citations back
    "press",            # tier 5, or tier 1 when published contemporaneously with the claim
    "low_authority",    # tier 5 — wikis, blogs, forums, social media, video: leads only
    "unknown",          # unclassified: never the evidentiary basis of anything
]

# Whether a citation may serve as the evidentiary basis of a claim, or is only a
# lead to follow. Computed from the tier rank in code, never asserted by the model:
# a source cannot promote itself.
CitationRole = Literal["evidence", "discovery"]

# Did we actually check? Kept separate from ClaimClass, which asks what the
# evidence entitles us to SAY. A claim can be a sound historical inference
# (claim_class) that nobody on this run verified against a source (verification).
Verification = Literal[
    "verified",    # an evidence-tier citation, and its page was actually read
    "unverified",  # evidence exists but was not checked, or rests on metadata alone
    "unknown",     # no evidence located — a finding, not a blank
]

# What KIND of thing supports a claim. Kept separate from confidence: a well-attested
# tradition is still a tradition.
EvidenceType = Literal[
    "primary_document",           # the original record itself
    "archaeological",             # physical / excavated evidence
    "contemporary_record",        # written down by someone alive at the time
    "near_contemporary_account",  # within living memory, not by a witness
    "later_historical_account",   # written generations later
    "scholarly_inference",        # a historian's reasoned conclusion, not a record
    "tradition",                  # transmitted belief / oral tradition
    "disputed",                   # sources actively contradict each other
    "absent",                     # no supporting evidence located
]

ConfidenceLabel = Literal["high", "moderate", "low", "speculative"]

# The reader-facing question "how do we know this?", answered in five words.
#
# EvidenceType above records what KIND of artefact supports a claim; this records
# what that artefact entitles us to say. They are derived, never independently
# guessed (see CLAIM_CLASS_BY_EVIDENCE_TYPE), so the two can never drift apart and
# the classification costs no extra tokens.
ClaimClass = Literal[
    "direct_evidence",       # a document, artefact or contemporary record states it
    "historical_inference",  # reasoned from evidence; no source states it outright
    "interpretation",        # a scholar's or commentator's reading, contested or not
    "tradition",             # transmitted belief with no documentary trail
    "unknown",               # no evidence located either way — a finding, not a blank
]

CLAIM_CLASS_BY_EVIDENCE_TYPE: dict[str, ClaimClass] = {
    "primary_document": "direct_evidence",
    "archaeological": "direct_evidence",
    "contemporary_record": "direct_evidence",
    "near_contemporary_account": "historical_inference",
    "later_historical_account": "historical_inference",
    "scholarly_inference": "interpretation",
    "disputed": "interpretation",
    "tradition": "tradition",
    "absent": "unknown",
}

# How one timeline item is held to relate to another. "no_established_link" is the
# most important member: two events being consecutive is not a connection, and the
# pipeline must be able to say so instead of inventing a causal story to fill the
# gap.
RelationType = Literal[
    "derives_from",         # B's content demonstrably comes from A
    "retells",              # same story retold; no textual dependency demonstrated
    "translates",           # B is a translation, recension or edition of A
    "responds_to",          # B answers, corrects or argues against A
    "contradicts",          # B is incompatible with A
    "contemporaneous",      # same period; no dependency claimed
    "attests",              # B is evidence that A already existed by then
    "provides_context",     # A is background B emerged from; influence NOT claimed
    "no_established_link",  # sequential only — the link is not evidenced
]

# What KIND of item sits on the timeline. The distinction a timeline of "events"
# cannot express, and the one this product exists for: when a text was WRITTEN,
# when the oldest surviving COPY of it was made, what independently ATTESTS it,
# and what is merely the BACKGROUND it emerged from are four different claims
# with four different kinds of evidence. Defaults to "event" so every trace
# written before this taxonomy existed is still valid.
# What kind of SURVIVING THING a step in the chain is.
#
# A trace is a chain of evidence, and every link in it must be something that still
# exists and can be examined. That is the whole constraint: each step answers "what is
# the next surviving piece of evidence?", and nothing that is not a surviving piece of
# evidence is a step. The old vocabulary mixed artefacts ("manuscript_witness") with
# interpretations of them ("context", "reconstructed_date", "institutional_development"),
# which is how traces filled up with things nobody can go and look at — a supposed
# climate of expectation, a movement's mood, a development inferred from silence. Those
# are conclusions, and conclusions now live in their own list, AFTER the evidence that
# is supposed to support them (see Conclusion below).
#
# If you cannot name where the object is, or the edition that publishes it, it is not
# one of these.
EvidenceKind = Literal[
    "text",                # a work surviving through copies — a gospel, a history, a treatise
    "manuscript",          # a specific physical copy: codex, papyrus, parchment leaf
    "scroll",              # a rolled manuscript, kept distinct because its finds are dated as such
    "letter",              # correspondence surviving as a text or an object
    "inscription",         # cut, carved, painted or stamped onto a durable surface
    "document",            # a charter, deed, contract, decree, or administrative record
    "record",              # a register kept as a series: census, court, tax, parish
    "artifact",            # an object carrying evidence: coin, seal, ostracon, tablet
    "archaeological_find", # an excavated site, structure, or assemblage, as reported
]

# Whether a work is really by whom it is credited to. "Attributed" is the honest
# default for most ancient texts and has nowhere to live in a plain title string.
Attribution = Literal[
    "established",     # authorship is evidenced
    "attributed",      # traditionally credited; not established
    "disputed",        # scholars actively disagree
    "anonymous",       # the work does not name its author
    "not_applicable",  # this node is not a work
]


class TraceRequest(BaseModel):
    title: str = Field(..., min_length=2, description="Story / event title to trace.")
    context: Optional[str] = Field(
        default=None,
        description="Optional disambiguating context, e.g. 'biblical figure' or 'New Mexico, 1947'.",
    )
    max_depth: Optional[int] = Field(default=None, ge=1, le=8)
    max_sources_per_stage: Optional[int] = Field(default=None, ge=1, le=20)
    language: str = Field(default="en")


class CacheKeyRequest(BaseModel):
    """Invalidate a cached trace by its title (see /cache/invalidate)."""
    title: str = Field(..., min_length=1)


class Citation(BaseModel):
    title: Optional[str] = None
    url: str
    snippet: Optional[str] = None
    tier: SourceTier = "unknown"
    # The tier as a number, 1 (primary evidence) to 5 (general web). Carried
    # alongside the string because the string is a category and this is an
    # ordering — and because it is computed per CLAIM, not per URL: a 1963
    # newspaper is primary evidence for a 1963 event and general web for a
    # medieval one. Legacy citations have no rank; the client derives one.
    tier_rank: int = Field(default=5, ge=1, le=5)
    # Whether this citation may be the evidentiary basis of the claim, or is a
    # lead to follow backward. Never taken from the model.
    role: CitationRole = "discovery"
    published: Optional[str] = Field(
        default=None,
        description="When this source was published, when stated — decides whether it is contemporary with the claim.",
    )
    # Set when several citations demonstrably descend from one upstream report:
    # they are one source repeated, not independent confirmations.
    chain: Optional[str] = Field(
        default=None,
        description="Identifier of the information chain this source belongs to, when it repeats another source.",
    )


class EvidenceDossier(BaseModel):
    """What actually backs a single claim, stated plainly enough to be argued with.

    Every field is answerable with 'None identified' — an honest gap is a finding,
    not a failure, and is far more useful than an invented citation.
    """

    claim: str = Field(..., description="The claim restated as one testable proposition.")
    earliest_supporting_source: str = Field(default="None identified")
    estimated_source_date: str = Field(
        default="Unknown",
        description="When the source was likely COMPOSED (a range is fine).",
    )
    earliest_surviving_copy: str = Field(
        default="None identified",
        description="The oldest physical copy that still exists, with its date — often far later than composition.",
    )
    contemporary_evidence: str = Field(
        default="None identified",
        description="Evidence created at the time of the event itself.",
    )
    independent_corroboration: str = Field(
        default="None identified",
        description="Support that does NOT descend from the same information chain.",
    )
    contradictory_evidence: str = Field(default="None identified")
    provenance: str = Field(
        default="None identified",
        description="Who holds the object, under what shelfmark, and how it got there.",
    )
    scholarly_dispute: str = Field(
        default="None identified",
        description="Live disagreement AMONG SCHOLARS about this claim — distinct from evidence against it.",
    )
    evidence_type: EvidenceType = "scholarly_inference"
    claim_class: ClaimClass = Field(
        default="interpretation",
        description="What the evidence entitles us to say. Derived from evidence_type, never guessed separately.",
    )
    confidence_label: ConfidenceLabel = "moderate"
    why: str = Field(default="", description="Short plain-language explanation of the assessment.")
    missing_piece: str = Field(
        default="",
        description="The single piece of evidence whose absence most limits this claim.",
    )
    disputed: bool = Field(
        default=False,
        description="Sources actively disagree about this claim (not merely unproven).",
    )
    # Derived in code from the citations' tier ranks and whether a page was read,
    # for the same reason claim_class is: a claim asked to rate its own
    # verification will talk itself up.
    verification: Verification = "unknown"
    # True only when the pipeline actually opened and read a source page for this
    # claim. False means the provenance fields rest on search summaries alone, and
    # the UI says so rather than presenting them with equal weight.
    verified_from_source: bool = Field(default=False)
    sources_read: List[str] = Field(
        default_factory=list,
        description="URLs whose page text was fetched and read when assessing this claim.",
    )


class TimelineEvent(BaseModel):
    # Stable handle so connections can point at this item. Assigned by the
    # pipeline ("t0", "t1", …; the origin is always "origin") and kept stable
    # across the chronological sort, which would otherwise renumber everything.
    id: str = Field(default="")
    year: Optional[int] = Field(
        default=None,
        description="Signed year. Negative = BCE. Null when only an era marker is known.",
    )
    era_label: Optional[str] = Field(
        default=None,
        description="Human-readable era when no year is available, e.g. 'Bronze Age oral tradition'.",
    )
    precision: DatePrecision = "unknown"
    # Set when the item covers a SPAN rather than a moment — a language shifting
    # over centuries, a canon forming. Without it, everything that is a process
    # rather than an event has to be either falsely pinned to one year or dropped.
    year_end: Optional[int] = Field(
        default=None,
        description="Signed end year when this item spans a period rather than happening at a moment.",
    )
    node_type: EvidenceKind = "text"
    attribution: Attribution = "not_applicable"
    source_title: str
    claim: str
    citations: List[Citation] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: Optional[EvidenceDossier] = None


class OriginResult(BaseModel):
    id: str = Field(default="origin")
    year: Optional[int] = None
    era_label: Optional[str] = None
    precision: DatePrecision = "unknown"
    year_end: Optional[int] = None
    node_type: EvidenceKind = "text"
    attribution: Attribution = "not_applicable"
    source_title: str
    summary: str
    citations: List[Citation] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: Optional[EvidenceDossier] = None


class ConnectionEvidence(BaseModel):
    """Why one event is held to lead to another — judged on the same terms as a claim.

    A connection is itself a claim, and the weakest link in most historical
    narratives: "A, then B, therefore A caused B" is exactly the reasoning this
    product exists to make visible. So a connection carries its own mechanism,
    its own supporting and contradicting evidence, its own confidence, and its
    own statement of what is missing.
    """

    mechanism: str = Field(
        default="",
        description="One sentence stating HOW the earlier item leads to the later one.",
    )
    supporting_evidence: str = Field(default="None identified")
    contradictory_evidence: str = Field(
        default="None identified",
        description="Evidence or scholarship against this link, including 'the link is merely assumed'.",
    )
    independent_corroboration: str = Field(
        default="None identified",
        description="Support for the link that does not descend from the same information chain.",
    )
    scholarly_dispute: str = Field(
        default="None identified",
        description="Live disagreement among scholars about whether this link holds.",
    )
    claim_class: ClaimClass = "unknown"
    evidence_type: EvidenceType = "scholarly_inference"
    verification: Verification = "unknown"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence_label: ConfidenceLabel = "moderate"
    why: str = Field(default="", description="Plain-language reading of how strong the link is.")
    missing_piece: str = Field(
        default="",
        description="What evidence would settle whether this link is real.",
    )
    disputed: bool = False


class Connection(BaseModel):
    """A directed, evidence-graded edge between two timeline items."""

    from_id: str = Field(..., description="id of the earlier item.")
    to_id: str = Field(..., description="id of the later item.")
    relation: RelationType = "no_established_link"
    evidence: ConnectionEvidence = Field(default_factory=ConnectionEvidence)
    citations: List[Citation] = Field(default_factory=list)


class EvidenceChain(BaseModel):
    """A set of citations that all descend from one upstream report.

    Ten sites repeating one article are one chain, and counting them as ten
    confirmations is the failure mode this exists to prevent. Chains are computed
    in code from the sources themselves, not asserted by the model.
    """

    id: str
    label: str = Field(default="", description="The upstream source this chain traces back to, when identified.")
    urls: List[str] = Field(default_factory=list)
    tier: SourceTier = "unknown"


class TokenUsage(BaseModel):
    """What this trace actually cost, so the budget is observable rather than assumed."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    pages_read: int = 0
    by_stage: dict = Field(default_factory=dict)


class Conclusion(BaseModel):
    """Something the evidence is taken to show — kept OUT of the chain, and after it.

    The chain answers one question at every step: what is the next surviving piece of
    evidence? Anything that is not a surviving piece of evidence is a reading OF that
    evidence, and mixing the two is what lets an interpretation inherit the authority of
    an artefact. A reader who sees "messianic expectation" sitting between two
    manuscripts has been shown a conclusion dressed as a find.

    So conclusions are stated separately, after the evidence, and each one has to name
    which pieces it rests on. A conclusion resting on nothing in the chain is exactly the
    kind of claim this product exists to expose, and it stays visible rather than being
    quietly dropped.
    """

    statement: str = Field(description="What the evidence is taken to show, as one plain proposition.")
    rests_on: List[str] = Field(
        default_factory=list,
        description="Ids of the chain steps supporting this. Empty means nothing in the chain supports it.",
    )
    confidence_label: ConfidenceLabel = "moderate"
    reasoning: str = Field(
        default="",
        description="How the named evidence gets you to the statement, in plain language.",
    )
    dissent: str = Field(
        default="",
        description="Serious disagreement with this reading, or 'None identified'.",
    )


class TraceResponse(BaseModel):
    title: str
    normalized_title: str
    origin: OriginResult
    timeline: List[TimelineEvent]
    # Evaluated edges between items. Empty on traces generated before connections
    # existed — every consumer must treat absence as "not assessed", never as
    # "no connections".
    connections: List[Connection] = Field(default_factory=list)
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
    queries_run: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    chains: List[EvidenceChain] = Field(default_factory=list)
    independent_chain_count: int = Field(
        default=0,
        description="Distinct information chains found. The honest count of how many separate things agree.",
    )
    sources_read: List[str] = Field(
        default_factory=list,
        description="URLs whose full page text was fetched and read during this trace.",
    )
    open_questions: List[str] = Field(
        default_factory=list,
        description="Evidence the research rounds looked for and did not find. What we do not know.",
    )
    conclusions: List[Conclusion] = Field(
        default_factory=list,
        description="What the evidence is taken to show. Stated after the chain, never inside it.",
    )
    usage: Optional[TokenUsage] = None
    iterations: int = 0
    duration_seconds: float = 0.0


# --------------------------------------------------------------------- expand
# What an expansion is being asked to go and find.
#
# Expanding exists to GROW a timeline with material it does not already contain, and
# an unaimed expansion mostly returns the neighbours the chain already has: ask for
# "related sub-events" around Paul's letters and you get the gospels back, which are
# already the next step along. Each mode points the search at a different axis, so
# six expansions of one node produce six different kinds of finding rather than six
# rewordings of one.
#
# Deliberately subject-agnostic. These have to read the same way for a manuscript, a
# telescope observation, a treaty, an excavation and a patent, because Chrono Origin
# does not know which it is looking at.
ExpandMode = Literal[
    "discovery",     # how this became known to us
    "preservation",  # how it survived or was carried forward
    "verification",  # how what we know about it was established
    "related",       # other evidence directly connected to it
    "earlier",       # evidence before it that fills a gap
    "later",         # what it influenced or contributed to afterwards
]


class ExpandRequest(BaseModel):
    """Request to expand a single timeline entry into finer-grained sub-events."""

    story_title: str = Field(..., min_length=2, description="The overall story being traced.")
    parent_source_title: str = Field(..., min_length=1, description="Source / event name of the timeline item being expanded.")
    parent_year: Optional[int] = Field(default=None, description="Signed year of the parent item (negative = BCE).")
    parent_era_label: Optional[str] = Field(default=None, description="Era label of the parent item when no year is known.")
    parent_claim: Optional[str] = Field(default=None, description="Existing claim text for the parent item.")
    parent_id: str = Field(default="parent", description="id the returned connections should hang off.")
    context: Optional[str] = Field(default=None, description="Optional disambiguating context for the overall story.")
    max_events: int = Field(default=6, ge=1, le=12, description="Max sub-events to return.")
    mode: ExpandMode = Field(
        default="related",
        description="Which axis to expand along. See ExpandMode.",
    )
    # What the board already shows. An expansion that returns the step sitting next to
    # the anchor has cost a call and added nothing, and the user cannot tell the
    # difference from a card that looks new until they read it.
    existing: List[str] = Field(
        default_factory=list,
        description="Titles already on the timeline, so the expansion can avoid repeating them.",
    )
    language: str = Field(default="en")


class ExpandResponse(BaseModel):
    """Sub-events related to a specific timeline item, in chronological order."""

    parent_source_title: str
    parent_year: Optional[int] = None
    parent_era_label: Optional[str] = None
    events: List[TimelineEvent] = Field(default_factory=list)
    # Parent -> sub-event edges, graded exactly like the main trace's connections.
    # An expansion that cannot justify why a sub-event belongs under its anchor is
    # showing the user a false relationship, which is worse than showing none.
    connections: List[Connection] = Field(default_factory=list)
    queries_run: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    sources_read: List[str] = Field(default_factory=list)
    usage: Optional[TokenUsage] = None
    duration_seconds: float = 0.0
