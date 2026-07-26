"""Pydantic request/response models for Narrava Studio."""
from __future__ import annotations

from typing import List, Literal, Optional, Tuple

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
    limit: int = Field(default=48, ge=1, le=60)


class MediaSearchResponse(BaseModel):
    clips: List[MediaClip]


# ── Storyboard (AI first visual pass) ────────────────────────────────────────
# An experienced-documentary-editor pass over the narration: where should the
# VISUAL change, and what should fill each section. Boundaries follow the story
# (a new idea / person / place / event / time period / visual concept), never a
# fixed interval or every sentence.
class StoryboardRequest(BaseModel):
    full_text: str = Field(..., min_length=3, max_length=20000)
    # The narration's real rendered length on the timeline; scene times are laid
    # out across it (the model never guesses seconds).
    total_duration_sec: float = Field(..., gt=0, le=36000)
    max_scenes: int = Field(default=24, ge=1, le=80)
    # Measured (start, end) per word of full_text, from POST /narration/align. With these
    # every boundary sits on a real word; without them the layout models how long the text
    # takes to say, which is close but drifts by a word or two — visible once a scene is
    # only a few seconds long. Must be positionally 1:1 with full_text's words or it is
    # ignored, since a mismatch would put every scene on the wrong word.
    word_times: Optional[List[Tuple[float, float]]] = Field(default=None)


class StoryboardScene(BaseModel):
    id: str
    index: int
    text: str                       # the narration span this visual covers
    start_sec: float
    end_sec: float
    rationale: str = ""             # one line: why the visual changes here
    media_type: Literal["image", "video"] = "image"
    search_terms: List[str] = Field(default_factory=list)
    visual_ideas: List[str] = Field(default_factory=list)


class StoryboardResponse(BaseModel):
    scenes: List[StoryboardScene] = Field(default_factory=list)
