"""Pydantic schemas for the Chrono-Origin API."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


DatePrecision = Literal["exact", "year", "decade", "century", "millennium", "era", "unknown"]


class TraceRequest(BaseModel):
    title: str = Field(..., min_length=2, description="Story / event title to trace.")
    context: Optional[str] = Field(
        default=None,
        description="Optional disambiguating context, e.g. 'biblical figure' or 'New Mexico, 1947'.",
    )
    max_depth: Optional[int] = Field(default=None, ge=1, le=8)
    max_sources_per_stage: Optional[int] = Field(default=None, ge=1, le=20)
    language: str = Field(default="en")


class Citation(BaseModel):
    title: Optional[str] = None
    url: str
    snippet: Optional[str] = None


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


class OriginResult(BaseModel):
    year: Optional[int] = None
    era_label: Optional[str] = None
    precision: DatePrecision = "unknown"
    source_title: str
    summary: str
    citations: List[Citation] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


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
