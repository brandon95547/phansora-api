"""What kind of material each piece of an import is (POST /sources/classify).

The app builds a project's Content Library from what an upload held (sources.py,
material.py) and asks here what each piece is:

  * a picture is an image (a photograph, a painting, an illustration), a diagram (a chart,
    a map, a labelled drawing), a table (a grid of data printed as a picture) or music (a
    score, notation). OpenAI's vision model looks at each one and says which, with a line
    describing it. Without a key, or when it fails, the layout decides: a vector drawing is
    a diagram, a raster picture an image, and a caption beginning "Table", "Figure" or
    naming a score tips it.
  * a section of text is explanation or an exercise (questions, tasks, activities), and
    has a few short topics, worded alike across sections so the library can gather them.
    DeepSeek reads each section's title and its opening.

Nothing is stored here: the app keeps the answers beside the material, never in place of it.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import re
from pathlib import Path
from typing import Any

from . import llm
from .errors import ServiceError
from .schemas import ClassifyRequest, ClassifyResponse, PictureClass, SectionClass

logger = logging.getLogger("talari")

VISION_MODEL_VARS = ("TALARI_VISION_MODEL", "OPENAI_MODEL")
VISION_DEFAULT = "gpt-5-nano"
# A picture is sent this size at most: enough to tell a score from a chart, and cheap.
VISION_PX = 768
VISION_CONCURRENCY = 4
VISION_TIMEOUT_S = 60
SECTION_BATCH = 40
SECTION_OPENING = 700
SECTION_CONCURRENCY = 4

EXERCISE = re.compile(
    r"\b(exercises?|activit(?:y|ies)|practice|questions?|worksheet|quiz|try (?:it|this)|your turn|problems?|tasks?|review)\b",
    re.IGNORECASE,
)
MUSIC = re.compile(r"\b(score|sheet music|notation|melody|measures?|bars?|staff|stave|chords?)\b", re.IGNORECASE)
DIAGRAM = re.compile(r"^(?:fig(?:ure)?|diagram|chart|graph|map)\b", re.IGNORECASE)
TABLE = re.compile(r"^table\b", re.IGNORECASE)

VISION_PROMPT = (
    "This picture comes from a book or document. Classify it as exactly one of:\n"
    '"image" — a photograph, painting, illustration or decorative art;\n'
    '"diagram" — a chart, graph, map, schematic, flowchart or labelled drawing;\n'
    '"table" — a grid of data or text shown as a picture;\n'
    '"music" — sheet music, a score or musical notation.\n'
    'Reply with JSON only: {"kind": "...", "description": "one plain sentence saying what it shows"}.'
)

# What a picture kept without a description shows (describe): what a remix starts from.
DESCRIBE_PROMPT = (
    "Describe what this picture shows, for an illustrator who will redraw it: the subject and what it is doing, "
    "the setting, the light, the main colors and the viewpoint. One paragraph of at most 60 words, in plain words. "
    "Say nothing about art style, medium, technique or artist.\n"
    'Reply with JSON only: {"description": "..."}'
)

SECTIONS_SYSTEM = (
    "You sort the sections of an imported book or document for an author's reference library. "
    "For each section, decide whether it is explanation (\"text\") or an exercise or activity for the reader — "
    "questions, tasks, practice problems, worksheets, quizzes (\"exercise\"). Give each one to three topics: "
    "short noun phrases of two to four words naming what it is about. Reuse the same wording for the same topic "
    "across sections, so sections on one topic share it."
)


def heuristic(kind: str, caption: str) -> tuple[str, str]:
    """A picture's kind from its layout and caption alone."""
    caption = caption or ""
    if kind == "table" or TABLE.match(caption):
        return "table", caption
    if MUSIC.search(caption):
        return "music", caption
    if kind == "drawing" or DIAGRAM.match(caption):
        return "diagram", caption
    return "image", caption


def _vision_model() -> str | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    for var in VISION_MODEL_VARS:
        value = (os.getenv(var) or "").split("#")[0].strip()
        if value:
            return value
    return VISION_DEFAULT


def _data_url(path: Path | bytes) -> str:
    from PIL import Image

    image = Image.open(path if isinstance(path, Path) else io.BytesIO(path))
    image.thumbnail((VISION_PX, VISION_PX))
    out = io.BytesIO()
    image.convert("RGB").save(out, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode("ascii")


async def _look(client: Any, model: str, path: Path | bytes, prompt: str = VISION_PROMPT, tokens: int = 400) -> dict[str, Any]:
    """One picture, looked at: {kind, description}, or what `prompt` asks of it."""
    url = await asyncio.to_thread(_data_url, path)
    reply = await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": url}}]}],
            response_format={"type": "json_object"},
            reasoning_effort="minimal",
            max_completion_tokens=tokens,
        ),
        VISION_TIMEOUT_S,
    )
    return llm.parse_json_loose(reply.choices[0].message.content or "{}")


async def describe(picture: bytes) -> str:
    """What a picture shows, in words, for one kept without a description (an upload, a library
    piece made a picture), so a remix knows what it is changing. Empty without a vision model,
    or when the picture cannot be looked at."""
    model = _vision_model()
    if not model:
        return ""
    try:
        seen = await _look(_vision_client(), model, picture, DESCRIBE_PROMPT, tokens=600)
    except Exception as error:  # noqa: BLE001 — the remix goes on from the picture's caption
        logger.warning("Talari: could not describe a picture: %s", error)
        return ""
    return " ".join(str(seen.get("description") or "").split())[:600]


def _vision_client():
    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def pictures(body: ClassifyRequest, images: Path) -> list[PictureClass]:
    model = _vision_model()
    client = _vision_client() if model and body.pictures else None
    gate = asyncio.Semaphore(VISION_CONCURRENCY)

    async def one(picture) -> PictureClass:
        kind, description = heuristic(picture.kind, picture.caption)
        name = Path(picture.src or "").name
        path = images / name
        if client is not None and re.fullmatch(r"[0-9a-f]{32}\.(png|jpg)", name) and path.is_file():
            try:
                async with gate:
                    seen = await _look(client, model, path)
                if seen.get("kind") in ("image", "diagram", "table", "music"):
                    kind = seen["kind"]
                description = " ".join(str(seen.get("description") or "").split())[:300] or description
            except Exception as error:  # noqa: BLE001 — one picture that cannot be read keeps its layout's answer
                logger.warning("Talari: could not classify a picture: %s", error)
        return PictureClass(id=picture.id, kind=kind, description=description)

    return list(await asyncio.gather(*(one(p) for p in body.pictures)))


async def sections(body: ClassifyRequest) -> list[SectionClass]:
    if not body.sections:
        return []
    batches = [body.sections[i : i + SECTION_BATCH] for i in range(0, len(body.sections), SECTION_BATCH)]
    gate = asyncio.Semaphore(SECTION_CONCURRENCY)

    def guess(section) -> SectionClass:
        return SectionClass(id=section.id, kind="exercise" if EXERCISE.search(section.title or "") else "text", topics=[])

    async def batch(items) -> list[SectionClass]:
        listing = "\n\n".join(f"=== [{s.id}] {s.title} ===\n{(s.text or '')[:SECTION_OPENING]}" for s in items)
        user = (
            f"Sections ({len(items)}):\n\n{listing}\n\n"
            'Return JSON: {"sections": [{"id": "...", "kind": "text" | "exercise", "topics": ["...", "..."]}]} — one entry for every section, by its id.'
        )
        try:
            async with gate:
                data = await llm.ask_json(SECTIONS_SYSTEM, user, max_tokens=min(60 * len(items) + 400, 6000), temperature=0.2)
        except ServiceError as error:
            logger.warning("Talari: could not classify sections: %s", error.message)
            return [guess(s) for s in items]
        found = {str(entry.get("id")): entry for entry in data.get("sections") or [] if isinstance(entry, dict)}
        out = []
        for section in items:
            entry = found.get(section.id)
            if not entry:
                out.append(guess(section))
                continue
            topics = [" ".join(str(t).split())[:60] for t in entry.get("topics") or [] if str(t).strip()][:3]
            out.append(SectionClass(id=section.id, kind="exercise" if entry.get("kind") == "exercise" else "text", topics=topics))
        return out

    results = await asyncio.gather(*(batch(items) for items in batches))
    return [entry for result in results for entry in result]


async def classify(body: ClassifyRequest, images: Path) -> ClassifyResponse:
    found_pictures, found_sections = await asyncio.gather(pictures(body, images), sections(body))
    return ClassifyResponse(pictures=found_pictures, sections=found_sections)
