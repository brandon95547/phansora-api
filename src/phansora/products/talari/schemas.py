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
    # The look as the app now chooses it — an art style and a medium ("auto" lets the art
    # style decide). A known art style wins over style_id, which older books still send.
    art_style: str | None = Field(default=None, max_length=40)
    medium: str | None = Field(default=None, max_length=40)
    # A substyle of the art style (styles.py SUBSTYLES); another art style's is not used.
    substyle: str | None = Field(default=None, max_length=40)
    # Image-to-image: a picture to redraw instead of drawing from nothing — base64 of a PNG,
    # JPEG or WebP — and how far to move from it. `strength` is the sampler's denoise: near 0
    # keeps the reference as it is, 1 ignores it entirely.
    reference: str | None = Field(default=None, max_length=40_000_000)
    strength: float = Field(default=0.6, ge=0.05, le=1.0)
    # Enlarged after drawing (Lanczos, enlarge.py), for a picture that fills a printed page.
    upscale: float = Field(default=1.0, ge=1.0, le=2.0)

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


class ExtractedSection(Lenient):
    """A part of a chapter under one of the document's own headings, as written."""

    index: int
    chapter_index: int
    level: int = 1
    title: str
    page_start: int | None = None
    page_end: int | None = None
    words: int = 0
    text: str


class ExtractedAsset(Lenient):
    """A picture, a drawing or a table found in the document, kept as it is: the picture
    (or, for a table, a picture of it as printed) under ``src``, a table's cells in ``rows``."""

    index: int
    kind: Literal["picture", "drawing", "table"]
    src: str | None = None
    width: int | None = None
    height: int | None = None
    page: int | None = None
    caption: str = ""
    rows: list[list[str]] | None = None
    header: bool = False
    chapter_index: int = 0
    section_index: int | None = None


class ExtractResponse(Lenient):
    title: str
    format: SourceFormat
    pages: int | None = None
    words: int
    chapters: list[ExtractedChapter]
    sections: list[ExtractedSection] = Field(default_factory=list)
    assets: list[ExtractedAsset] = Field(default_factory=list)
    # The upload as it came, kept for its pages to be looked at again (GET …/pages/{n}).
    original: str | None = None
    # A PDF read without looking for its pictures and tables: ask for them with
    # POST …/sources/{original}/assets.
    assets_pending: bool = False


class ClassifyPicture(Lenient):
    id: str = Field(max_length=40)
    src: str = Field(default="", max_length=200)
    kind: Literal["picture", "drawing", "table"] = "picture"
    caption: str = Field(default="", max_length=600)


class ClassifySection(Lenient):
    id: str = Field(max_length=40)
    title: str = Field(default="", max_length=300)
    text: str = Field(default="", max_length=4_000)


class ClassifyRequest(Lenient):
    pictures: list[ClassifyPicture] = Field(default_factory=list, max_length=600)
    sections: list[ClassifySection] = Field(default_factory=list, max_length=3000)


class PictureClass(Lenient):
    id: str
    kind: Literal["image", "diagram", "table", "music"]
    description: str = ""


class SectionClass(Lenient):
    id: str
    kind: Literal["text", "exercise"]
    topics: list[str] = Field(default_factory=list)


class ClassifyResponse(Lenient):
    pictures: list[PictureClass]
    sections: list[SectionClass]


class KeptSource(Lenient):
    """An upload kept as it came, to be read later (…/sources/{original}/read)."""

    original: str
    format: SourceFormat
    bytes: int


class ReadRequest(Lenient):
    """How to read a kept upload: with its pictures, drawings and tables (``assets``), or
    only its text and structure; its title, else the file's own or its name's."""

    assets: bool = True
    title: str = Field(default="", max_length=300)
    filename: str = Field(default="", max_length=300)


class ForgetRequest(Lenient):
    """Files of a user's that the app no longer uses: pictures (drawn, or found in an import)
    by their names under ``images/``, and kept imports by theirs under ``sources/``."""

    images: list[str] = Field(default_factory=list, max_length=20_000)
    sources: list[str] = Field(default_factory=list, max_length=2_000)


class ForgetResponse(Lenient):
    images: int = 0
    sources: int = 0


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
    art_style: str | None = Field(default=None, max_length=40)
    medium: str | None = Field(default=None, max_length=40)
    # A substyle of the art style (styles.py SUBSTYLES); another art style's is not used.
    substyle: str | None = Field(default=None, max_length=40)
    count: int = Field(default=1, ge=1, le=6)
    purpose: Literal["interior", "cover"] = "interior"


class IllustrationIdea(Lenient):
    prompt: str
    caption: str = ""
    # A short verbatim quote from the chapter the picture illustrates; empty for a cover.
    anchor: str = ""


class IllustrationPromptsResponse(Lenient):
    prompts: list[IllustrationIdea]


# --- a picture made by a conversation (image_guide.py) -----------------------------------


class ImageAnswer(Lenient):
    id: str = Field(default="", max_length=60)
    question: str = Field(default="", max_length=300)
    answer: str = Field(default="", max_length=300)
    # "Let Talari decide": the model chooses, and says what in the summary.
    decide: bool = False


# A background design (image_guide.designs): what it looks like, the kinds it combines —
# border, corners, texture, pattern, minimal, scene, emblem, abstract — and colors for its
# preview, as #rrggbb.
class ImageDesign(Lenient):
    title: str = Field(default="", max_length=120)
    description: str = Field(default="", max_length=400)
    kinds: list[str] = Field(default_factory=list, max_length=4)
    palette: list[str] = Field(default_factory=list, max_length=4)


class ImageBrief(Lenient):
    book: IllustrationBook = Field(default_factory=IllustrationBook)
    # A cover's background (front or back), or a picture for a page inside.
    purpose: Literal["interior", "cover", "back-cover"] = "interior"
    # A background fills the page; media is a picture placed on it, in its own shape.
    mode: Literal["background", "media"] = "media"
    # The page's own words, for a picture made for it.
    page_text: str = Field(default="", max_length=8000)
    # The look it will be drawn in, in words (an art style and a medium): already chosen.
    look: str = Field(default="", max_length=200)
    # The page's size and shape, in words ("6 × 9 in, portrait"): already chosen.
    size: str = Field(default="", max_length=80)
    # The background design (or designs, combined) the author chose to begin from.
    designs: list[ImageDesign] = Field(default_factory=list, max_length=3)
    answers: list[ImageAnswer] = Field(default_factory=list, max_length=12)


class ImageQuestion(Lenient):
    id: str
    text: str
    suggestions: list[str] = Field(default_factory=list)


class ImageGuideResponse(Lenient):
    # The next question, or none when the picture needs nothing more.
    question: ImageQuestion | None = None
    # The picture as it stands, in plain words for the author.
    summary: str = ""


class ImageDesignsRequest(ImageBrief):
    # Designs already shown, for "Explore other designs": not suggested again.
    exclude: list[str] = Field(default_factory=list, max_length=40)
    count: int = Field(default=5, ge=4, le=6)


class ImageDesignsResponse(Lenient):
    designs: list[ImageDesign]


class ImageComposeRequest(ImageBrief):
    # A change to a picture already made: its description, and the change in the author's words.
    previous: str = Field(default="", max_length=4000)
    instruction: str = Field(default="", max_length=300)


class ImageComposeResponse(Lenient):
    # For the image model only: never shown to the author.
    prompt: str
    caption: str = ""
    summary: str = ""
    # A placed picture's shape; a background is the page's.
    shape: Literal["portrait", "square", "landscape"] = "square"


# --- the ComfyUI lab ------------------------------------------------------------------
# The styleguide's ComfyUI test page (styleguide/comfyui.html in the app) drives this: every
# input a FLUX or Stable Diffusion workflow has, passed through as given, so ComfyUI itself
# can be tried. None of Talari's styles, sizes or fixed settings apply. Field names are
# ComfyUI's own (negative_prompt, sampler_name…); the camelCase spellings work too.

PICTURE_MAX = 40_000_000    # base64 of one picture


class LabRequest(Lenient):
    # FLUX reads a prompt with two text encoders, which CLIPTextEncodeFlux feeds separately:
    # `clip_l` (CLIP-L: short, about 77 tokens; keywords, style, mood) and `t5xxl` (T5-XXL:
    # long, plain sentences). `prompt` sends one text to both (CLIPTextEncode), and is what a
    # checkpoint model reads; given either of the two, the two win.
    prompt: str = Field(default="", max_length=4000)
    clip_l: str = Field(default="", max_length=4000)
    t5xxl: str = Field(default="", max_length=4000)
    negative_prompt: str = Field(default="", max_length=4000)
    # A file from the lab's model list; empty for the one Talari draws with.
    model: str = Field(default="", max_length=300)
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    seed: int | None = Field(default=None, ge=0, le=2**64 - 1)
    steps: int = Field(default=24, ge=1, le=80)
    cfg: float = Field(default=1.0, ge=0, le=30)
    # FLUX's prompt strength (FluxGuidance, or inside CLIPTextEncodeFlux); ignored by a
    # checkpoint model.
    guidance: float = Field(default=3.5, ge=0, le=30)
    denoise: float = Field(default=1.0, ge=0, le=1)
    sampler_name: str = Field(default="euler", max_length=64)
    scheduler: str = Field(default="simple", max_length=64)
    # Base64 PNG, JPEG or WebP. `image` is the source picture the sampler starts from; `mask`
    # (needs `image`) is black and white, white where the picture may change; `style_image`
    # is a separate picture whose style alone is applied — not its subject — by ByteDance's
    # USO (USOStyleReference). `style_image_weight` is the strength of USO's LoRA, its one
    # dial: 1 is ComfyUI's template, lower applies the style more lightly.
    image: str | None = Field(default=None, max_length=PICTURE_MAX)
    mask: str | None = Field(default=None, max_length=PICTURE_MAX)
    style_image: str | None = Field(default=None, max_length=PICTURE_MAX)
    style_image_weight: float = Field(default=1.0, ge=0, le=2)
    # EasyCache's reuse threshold: skips sampling steps that would barely change the picture —
    # faster, slightly grainier. 0 is off; 0.2 is ComfyUI's default.
    easycache: float = Field(default=0.0, ge=0, le=1)

    @model_validator(mode="after")
    def _something_to_draw(self) -> "LabRequest":
        if not (self.prompt + self.clip_l + self.t5xxl).strip() and not self.image and not self.style_image:
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
    # Whether ComfyUI has what a style picture needs: USO's LoRA and projector, and the vision
    # model it reads the picture with.
    style_ready: bool
    style_missing: list[str]
    defaults: dict[str, Any]
