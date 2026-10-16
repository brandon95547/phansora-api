"""Request and response shapes. The wire format is camelCase, like the rest of the dashboard's JSON."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel


class Model(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class IllustrationRequest(Model):
    # What to draw. May be empty when the picture is redrawn from a reference (below): the
    # reference is the subject then, and the style still comes from style_id.
    prompt: str = Field(default="", max_length=4000)
    style_id: str
    # Width over height. The picture keeps the base size's pixel budget at this shape.
    aspect: float = Field(gt=0.2, lt=5)
    purpose: Literal["interior", "cover"] = "interior"
    # The book's pages: pictures for dark pages are drawn on a dark ground.
    theme: Literal["light", "dark"] = "light"
    seed: int | None = None
    # The author's own description of the look, used when style_id is "custom".
    custom_style: str | None = Field(default=None, max_length=400)
    palette: Literal["color", "grayscale"] = "color"
    detail: Literal["simple", "balanced", "rich"] = "balanced"
    # Image-to-image: a picture to redraw instead of drawing from nothing — base64 of a PNG,
    # JPEG or WebP — and how far to move from it. `strength` is the sampler's denoise: near 0
    # keeps the reference as it is, 1 ignores it entirely.
    reference: str | None = Field(default=None, max_length=40_000_000)
    strength: float = Field(default=0.6, ge=0.05, le=1.0)

    @model_validator(mode="after")
    def _something_to_draw(self) -> "IllustrationRequest":
        if not self.prompt.strip() and not self.reference:
            raise ValueError("Describe the picture, or give a reference picture to redraw.")
        return self


class IllustrationResponse(Model):
    # A path under the API's root (e.g. /talari/users/7/images/<hex>.png). A browser that
    # cannot send a header with an <img> adds ?pt=<token>, which the auth gate accepts.
    src: str
    seed: int


class ServiceState(Model):
    available: bool
    detail: str = ""


class Status(Model):
    illustration: ServiceState


# --- writing ----------------------------------------------------------------
# Requests are lenient where the app may leave a field out (it is optional in the UI),
# and bounded where an unbounded field would only cost tokens.


class Lenient(Model):
    """A null is a field left out: it takes the field's default. JavaScript sends null for
    "not set" as readily as it omits the key, and both mean the same thing here. A
    required field sent as null is still refused."""

    @model_validator(mode="before")
    @classmethod
    def _nulls_are_absent(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {key: value for key, value in data.items() if value is not None}
        return data


Mode =Literal["idea", "rewrite", "extract", "documents"]
SourceFormat = Literal["pdf", "epub", "docx", "txt", "md"]

TEXT_MAX = 400_000      # one chapter's text, or a passage being revised
NOTE_MAX = 4_000        # a brief's free-text fields


class ExtractedChapter(Lenient):
    index: int
    title: str
    words: int
    page_start: int | None = None
    page_end: int | None = None
    text: str


class ExtractResponse(Lenient):
    title: str
    format: SourceFormat
    pages: int | None = None
    words: int
    chapters: list[ExtractedChapter]


class SourceChapterText(Lenient):
    index: int
    title: str = ""
    text: str = Field(default="", max_length=TEXT_MAX)


class AnalyzeRequest(Lenient):
    kind: Literal["book", "document"] = "book"
    title: str = ""
    chapters: list[SourceChapterText] = Field(min_length=1, max_length=2000)


class Concept(Lenient):
    name: str
    description: str = ""
    chapter_indexes: list[int] = Field(default_factory=list)


class ChapterSummary(Lenient):
    index: int
    title: str = ""
    summary: str = ""


class AnalyzeResponse(Lenient):
    summary: str
    themes: list[str]
    concepts: list[Concept]
    audience: str
    tone: str
    chapters: list[ChapterSummary]


class Brief(Lenient):
    idea: str = Field(default="", max_length=NOTE_MAX)
    genre: str = ""
    audience: str = ""
    tone: str = ""
    # An idea's length. A book made from sources is sized by them instead: target_words is
    # the length the app estimates its material supports, and when it is set, it wins.
    length: Literal["short", "medium", "long"] = "medium"
    target_words: int = Field(default=0, ge=0, le=1_000_000)
    chapter_count: int | None = Field(default=None, ge=1, le=60)
    approach: str = Field(default="", max_length=NOTE_MAX)
    instructions: str = Field(default="", max_length=NOTE_MAX)


class Rewrite(Lenient):
    depth: Literal["light", "moderate", "deep", "full"] = "moderate"
    options: list[str] = Field(default_factory=list, max_length=20)


class SourceConcept(Lenient):
    name: str
    description: str = ""


class SourceChapterSummary(ChapterSummary):
    # How much text the chapter holds, so a plan can give it the room its material needs.
    words: int = Field(default=0, ge=0)


class BlueprintSource(Lenient):
    id: str
    title: str = ""
    summary: str = ""
    themes: list[str] = Field(default_factory=list)
    concepts: list[SourceConcept] = Field(default_factory=list)
    chapters: list[SourceChapterSummary] = Field(default_factory=list)


class BlueprintRequest(Lenient):
    mode: Mode
    brief: Brief = Field(default_factory=Brief)
    rewrite: Rewrite | None = None
    sources: list[BlueprintSource] = Field(default_factory=list, max_length=50)
    count: int = Field(default=1, ge=1, le=3)


class Section(Lenient):
    title: str
    topics: list[str] = Field(default_factory=list)


class SourceRef(Lenient):
    source_id: str
    chapter_index: int


class BlueprintChapter(Lenient):
    title: str
    summary: str = ""
    sections: list[Section] = Field(default_factory=list)
    estimated_words: int = 0
    source_refs: list[SourceRef] = Field(default_factory=list)


class Blueprint(Lenient):
    title: str
    subtitle: str = ""
    description: str = ""
    audience: str = ""
    tone: str = ""
    chapters: list[BlueprintChapter]


class BlueprintResponse(Lenient):
    proposals: list[Blueprint]


class ChatTurn(Lenient):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=20_000)


class RefineRequest(Lenient):
    blueprint: Blueprint
    message: str = Field(min_length=1, max_length=NOTE_MAX)
    history: list[ChatTurn] = Field(default_factory=list)


class RefineResponse(Lenient):
    reply: str
    blueprint: Blueprint


class BookInfo(Lenient):
    title: str = ""
    subtitle: str = ""
    description: str = ""
    audience: str = ""
    tone: str = ""
    instructions: str = Field(default="", max_length=NOTE_MAX)


class OutlineEntry(Lenient):
    title: str = ""
    summary: str = ""


class ChapterPlan(Lenient):
    title: str
    summary: str = ""
    sections: list[Section] = Field(default_factory=list)
    estimated_words: int = Field(default=1500, ge=0, le=50_000)
    instructions: str = Field(default="", max_length=NOTE_MAX)


class Excerpt(Lenient):
    title: str = ""
    text: str = Field(default="", max_length=TEXT_MAX)


class IllustrationPlan(Lenient):
    enabled: bool = False
    per_chapter: int = Field(default=1, ge=0, le=6)


class ChapterRequest(Lenient):
    book: BookInfo = Field(default_factory=BookInfo)
    outline: list[OutlineEntry] = Field(default_factory=list)
    chapter_index: int = Field(default=0, ge=0)
    chapter: ChapterPlan
    previous_summary: str = Field(default="", max_length=NOTE_MAX)
    source_excerpts: list[Excerpt] = Field(default_factory=list, max_length=50)
    mode: Mode = "idea"
    rewrite: Rewrite | None = None
    illustrations: IllustrationPlan = Field(default_factory=IllustrationPlan)


class Block(Lenient):
    """One piece of a chapter. ``html`` and ``items`` hold inline HTML only: escaped text
    with <em> and <strong>. Fields that do not apply are left out of the JSON."""

    type: Literal["heading", "paragraph", "quote", "list"]
    level: Literal[2, 3] | None = None
    html: str | None = None
    items: list[str] | None = None
    # Present (true) only on a numbered list, whose order is part of what it says.
    ordered: bool | None = None


class PlacedIllustration(Lenient):
    # The picture goes after this block (an index into ``blocks``).
    after_block: int
    prompt: str
    caption: str = ""


class ChapterResponse(Lenient):
    blocks: list[Block]
    summary: str
    illustrations: list[PlacedIllustration]


class ReviseContext(Lenient):
    book_title: str = ""
    chapter_title: str = ""
    before: str = Field(default="", max_length=TEXT_MAX)
    after: str = Field(default="", max_length=TEXT_MAX)


class ReviseRequest(Lenient):
    action: Literal["rewrite", "expand", "shorten", "simplify", "tone", "continue", "flow", "examples", "factcheck", "custom"]
    html: str = Field(default="", max_length=TEXT_MAX)
    instruction: str = Field(default="", max_length=NOTE_MAX)
    tone: str = Field(default="", max_length=500)
    context: ReviseContext = Field(default_factory=ReviseContext)


class ReviseResponse(Lenient):
    html: str
    notes: list[str]


class IllustrationBook(Lenient):
    title: str = ""
    description: str = ""
    tone: str = ""


class IllustrationChapter(Lenient):
    title: str = ""
    text: str = Field(default="", max_length=TEXT_MAX)


class IllustrationPromptsRequest(Lenient):
    book: IllustrationBook = Field(default_factory=IllustrationBook)
    chapter: IllustrationChapter = Field(default_factory=IllustrationChapter)
    style_id: str = ""
    custom_style: str | None = Field(default=None, max_length=400)
    count: int = Field(default=1, ge=1, le=6)
    purpose: Literal["interior", "cover"] = "interior"


class IllustrationIdea(Lenient):
    prompt: str
    caption: str = ""
    # A short verbatim quote from the chapter the picture illustrates; empty for a cover.
    anchor: str = ""


class IllustrationPromptsResponse(Lenient):
    prompts: list[IllustrationIdea]


# --- the ComfyUI lab ------------------------------------------------------------------
# The styleguide's ComfyUI test page (styleguide/comfyui.html in the app) drives this: every
# input a FLUX or Stable Diffusion workflow has, passed through as given, so ComfyUI itself
# can be tried. None of Talari's styles, sizes or fixed settings apply. Field names are
# ComfyUI's own (negative_prompt, sampler_name…); the camelCase spellings work too.

PICTURE_MAX = 40_000_000    # base64 of one picture


class LabRequest(Lenient):
    prompt: str = Field(default="", max_length=4000)
    negative_prompt: str = Field(default="", max_length=4000)
    # A file from the lab's model list; empty for the one Talari draws with.
    model: str = Field(default="", max_length=300)
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    seed: int | None = Field(default=None, ge=0, le=2**64 - 1)
    steps: int = Field(default=24, ge=1, le=80)
    cfg: float = Field(default=1.0, ge=0, le=30)
    # FLUX's prompt strength (FluxGuidance); ignored by a checkpoint model.
    guidance: float = Field(default=3.5, ge=0, le=30)
    denoise: float = Field(default=1.0, ge=0, le=1)
    sampler_name: str = Field(default="euler", max_length=64)
    scheduler: str = Field(default="simple", max_length=64)
    # Base64 PNG, JPEG or WebP. `image` is the source picture the sampler starts from; `mask`
    # (needs `image`) is black and white, white where the picture may change; `style_image`
    # is a separate picture whose look is applied (FLUX Redux), by `style_image_weight`.
    image: str | None = Field(default=None, max_length=PICTURE_MAX)
    mask: str | None = Field(default=None, max_length=PICTURE_MAX)
    style_image: str | None = Field(default=None, max_length=PICTURE_MAX)
    style_image_weight: float = Field(default=1.0, ge=0, le=10)
    style_strength_type: Literal["multiply", "attn_bias"] = "multiply"
    # Files for the style picture: a style model and the vision model it reads pictures
    # with; empty for the first ComfyUI has.
    style_model: str = Field(default="", max_length=300)
    clip_vision: str = Field(default="", max_length=300)

    @model_validator(mode="after")
    def _something_to_draw(self) -> "LabRequest":
        if not self.prompt.strip() and not self.image and not self.style_image:
            raise ValueError("Write a prompt, or give a picture to start from or to take a style from.")
        if self.mask and not self.image:
            raise ValueError("A mask marks part of a picture: give the picture (image) too.")
        return self


class LabResponse(Model):
    src: str
    seed: int
    width: int
    height: int
    # The workflow ComfyUI was given, pictures named by their upload names.
    workflow: dict[str, Any]


class LabOptions(Model):
    """What this ComfyUI has to choose from, and what the lab starts with."""

    models: list[str]
    default_model: str
    samplers: list[str]
    schedulers: list[str]
    style_models: list[str]
    clip_vision: list[str]
    strength_types: list[str]
    defaults: dict[str, Any]
