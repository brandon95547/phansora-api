"""Pydantic request/response models for Narrava Studio."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ── Voices ───────────────────────────────────────────────────────────────────
class VoicePreset(BaseModel):
    id: str
    name: str
    description: str
    language: str = "en"
    gender: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


# ── Script generation ────────────────────────────────────────────────────────
class ScriptGenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=4000)
    # Narration voice/style flavour the writer should target.
    style: str = "documentary"
    tone: Optional[str] = None
    target_duration_sec: Optional[int] = Field(default=None, ge=10, le=3600)


class ScriptSegment(BaseModel):
    id: str
    index: int
    text: str
    start_sec: float
    end_sec: float
    keywords: List[str] = Field(default_factory=list)


class Script(BaseModel):
    title: str
    full_text: str
    segments: List[ScriptSegment] = Field(default_factory=list)
    estimated_duration_sec: float = 0.0
    source: Literal["prompt", "provided"] = "prompt"


class ScriptGenerateResponse(BaseModel):
    script: Script


# ── Segmentation (for a pasted/uploaded script) ──────────────────────────────
class SegmentRequest(BaseModel):
    script_text: str = Field(..., min_length=3)
    title: Optional[str] = None
    words_per_minute: Optional[int] = Field(default=None, ge=60, le=300)


class SegmentResponse(BaseModel):
    script: Script


# ── Timeline ─────────────────────────────────────────────────────────────────
class MediaClip(BaseModel):
    id: str
    segment_id: str
    type: Literal["image", "video"] = "image"
    url: str
    thumbnail_url: Optional[str] = None
    source: str = ""
    license: Optional[str] = None
    license_url: Optional[str] = None
    attribution: Optional[str] = None
    title: Optional[str] = None
    query: str = ""
    start_sec: float = 0.0
    end_sec: float = 0.0


class Transition(BaseModel):
    type: Literal["none", "cut", "fade", "crossfade", "slide"] = "fade"
    duration_sec: float = 0.5


class TimelineItem(BaseModel):
    segment_id: str
    clip: MediaClip
    transition_in: Transition = Field(default_factory=Transition)


class Timeline(BaseModel):
    voice_id: Optional[str] = None
    items: List[TimelineItem] = Field(default_factory=list)
    total_duration_sec: float = 0.0


class TimelineBuildRequest(BaseModel):
    segments: List[ScriptSegment]
    voice_id: Optional[str] = None
    media_types: List[Literal["image", "video"]] = Field(default_factory=lambda: ["image"])
    per_segment: int = Field(default=1, ge=1, le=4)


class TimelineBuildResponse(BaseModel):
    timeline: Timeline


# ── Media search (replace-a-clip) ────────────────────────────────────────────
class MediaSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)
    media_type: Literal["image", "video"] = "image"
    limit: int = Field(default=8, ge=1, le=20)


class MediaSearchResponse(BaseModel):
    clips: List[MediaClip]
