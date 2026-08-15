"""Pydantic schemas for the Chrono-Origin API."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


DatePrecision = Literal["exact", "year", "decade", "century", "millennium", "era", "unknown"]

# Where a citation sits in the source hierarchy the trace works down. The pipeline
# ranks evidence by this, not by the outlet's brand: a claim is only as strong as
# the citation you can follow backwards from it.
SourceTier = Literal[
    "primary",        # original / earliest surviving document, inscription, record, photo, map
    "repository",     # museum, archive, library, manuscript collection, critical edition
    "academic",       # peer-reviewed papers, university-press books, specialist historians
    "press",          # journalism and general publications, mainstream or alternative
    "low_authority",  # wikis, blogs, forums, social media, video — leads only
    "unknown",
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
    evidence_type: EvidenceType = "scholarly_inference"
    confidence_label: ConfidenceLabel = "moderate"
    why: str = Field(default="", description="Short plain-language explanation of the assessment.")
    missing_piece: str = Field(
        default="",
        description="The single piece of evidence whose absence most limits this claim.",
    )


class TimelineEvent(BaseModel):
    year: Optional[int] = Field(
        default=None,
        description="Signed year. Negative = BCE. Null when only an era marker is known.",
    )
    era_label: Optional[str] = Field(
        default=None,
        description="Human-readable era when no year is available, e.g. 'Bronze Age oral tradition'.",
    )
    precision: DatePrecision = "unknown"
    source_title: str
    claim: str
    citations: List[Citation] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: Optional[EvidenceDossier] = None


class OriginResult(BaseModel):
    year: Optional[int] = None
    era_label: Optional[str] = None
    precision: DatePrecision = "unknown"
    source_title: str
    summary: str
    citations: List[Citation] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence: Optional[EvidenceDossier] = None


class TraceResponse(BaseModel):
    title: str
    normalized_title: str
    origin: OriginResult
    timeline: List[TimelineEvent]
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
    queries_run: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    iterations: int = 0
    duration_seconds: float = 0.0


# --------------------------------------------------------------------- expand
class ExpandRequest(BaseModel):
    """Request to expand a single timeline entry into finer-grained sub-events."""

    story_title: str = Field(..., min_length=2, description="The overall story being traced.")
    parent_source_title: str = Field(..., min_length=1, description="Source / event name of the timeline item being expanded.")
    parent_year: Optional[int] = Field(default=None, description="Signed year of the parent item (negative = BCE).")
    parent_era_label: Optional[str] = Field(default=None, description="Era label of the parent item when no year is known.")
    parent_claim: Optional[str] = Field(default=None, description="Existing claim text for the parent item.")
    context: Optional[str] = Field(default=None, description="Optional disambiguating context for the overall story.")
    max_events: int = Field(default=6, ge=1, le=12, description="Max sub-events to return.")
    language: str = Field(default="en")


class ExpandResponse(BaseModel):
    """Sub-events related to a specific timeline item, in chronological order."""

    parent_source_title: str
    parent_year: Optional[int] = None
    parent_era_label: Optional[str] = None
    events: List[TimelineEvent] = Field(default_factory=list)
    queries_run: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    duration_seconds: float = 0.0
