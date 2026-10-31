"""What kind of material each piece of an import is (POST /sources/classify).

The app builds a project's Content Library from what an upload held (sources.py,
material.py) and asks here what each piece is:

  * a picture is an image (a photograph, a painting, an illustration), a diagram (a chart,
    a map, a labelled drawing), a table (a grid of data printed as a picture) or music (a
    score, notation). OpenAI's vision model looks at each one and says which, with a line
    describing it. Without a key, or when it fails, the layout decides: a vector drawing is
    a diagram, a raster picture an image, and a caption beginning "Table", "Figure" or
    naming a score tips it.
  * a section of text is explanation or an exercise (questions, tasks, activities). It also
    gets a broad subject and a primary topic (Subject → Topic → Content Section, the
    library's order), up to two related topics, and, when its heading is only a placeholder,
    a real title. The wording is the project's own where it fits: the app passes the
    subjects and topics already in use, and a name that differs only in case is made one.
    DeepSeek reads each section's title and its opening.

Nothing is stored here: the app keeps the answers beside the material, never in place of it.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from . import llm
from .errors import ServiceError
from .prompts import placeholder_title
from .schemas import KNOWN_NAME_MAX, ClassifyRequest, ClassifyResponse, ClassifySection, PictureClass, SectionClass

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
# What one section's answer takes at most: id, subject, topic, two more topics, kind, a title.
SECTION_TOKENS = 110
NAME_MAX = 60
TITLE_MAX = 100

EXERCISE = re.compile(
    r"\b(exercises?|activit(?:y|ies)|practice|questions?|worksheet|quiz|try (?:it|this)|your turn|problems?|tasks?|review)\b",
    re.IGNORECASE,
)
MUSIC = re.compile(r"\b(score|sheet music|notation|melody|measures?|bars?|staff|stave|chords?)\b", re.IGNORECASE)
DIAGRAM = re.compile(r"^(?:fig(?:ure)?|diagram|chart|graph|map)\b", re.IGNORECASE)
TABLE = re.compile(r"^table\b", re.IGNORECASE)
# A made-up title never starts with its place in the book: "Part 3: Osiris", "3. Osiris".
NUMBERING = re.compile(r"^(?:part\s+\d+|\d+)\s*[:.)\-–—]\s*", re.IGNORECASE)
QUOTES = str.maketrans("", "", '"“”„«»')

VISION_PROMPT = (
    "This picture comes from a book or document. Classify it as exactly one of:\n"
    '"image" — a photograph, painting, illustration or decorative art;\n'
    '"diagram" — a chart, graph, map, schematic, flowchart or labelled drawing;\n'
    '"table" — a grid of data or text shown as a picture;\n'
    '"music" — sheet music, a score or musical notation.\n'
    'Reply with JSON only: {"kind": "...", "description": "one plain sentence saying what it shows"}.'
)

SECTIONS_SYSTEM = (
    "You sort the sections of an imported book or document into an author's reference library, "
    "organized as subject, then topic, then section. For each section give:\n"
    '"subject" — the broad area it belongs to: a noun phrase of two to four words in Title Case, such as '
    '"Ancient Egyptian Religion". A whole book normally has only three to eight subjects, so be coarse. '
    "When a subject already in use fits, copy it exactly as spelled; coin a new one only when none does.\n"
    '"topic" — the specific theme of this section within that subject, two to five words, such as '
    '"Osiris and Resurrection". It is narrower than the subject, and sections that truly overlap share it. '
    "Reuse a topic already in use, exactly as spelled, when it fits.\n"
    '"topics" — the same topic first, then up to two more related ones, worded the same way.\n'
    '"kind" — "text" for explanation, or "exercise" for an exercise or activity for the reader: '
    "questions, tasks, practice problems, worksheets, quizzes.\n"
    'A section whose heading is only a placeholder ("Part 3", "Opening", "(untitled)" or empty) also gets '
    '"title": three to eight words naming what it covers, with no numbering, no word "Part" and no quotation marks.'
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


def _data_url(path: Path) -> str:
    from PIL import Image

    image = Image.open(path)
    image.thumbnail((VISION_PX, VISION_PX))
    out = io.BytesIO()
    image.convert("RGB").save(out, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(out.getvalue()).decode("ascii")


async def _look(client: Any, model: str, path: Path) -> dict[str, Any]:
    """One picture, looked at: {kind, description}."""
    url = await asyncio.to_thread(_data_url, path)
    reply = await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": [{"type": "text", "text": VISION_PROMPT}, {"type": "image_url", "image_url": {"url": url}}]}],
            response_format={"type": "json_object"},
            reasoning_effort="minimal",
            max_completion_tokens=400,
        ),
        VISION_TIMEOUT_S,
    )
    return llm.parse_json_loose(reply.choices[0].message.content or "{}")


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


class _Names:
    """The subjects or topics of a call, spelled the way each was first seen: those the project
    already uses first, then the model's in section order. A name that differs from one seen
    only in case takes its spelling, so "ancient egyptian religion" and "Ancient Egyptian
    Religion" are one subject."""

    def __init__(self, known: list[str]) -> None:
        self.seen: dict[str, str] = {}
        for name in known:
            self.seen.setdefault(name.casefold(), name)

    def spell(self, name: str) -> str:
        return self.seen.setdefault(name.casefold(), name)[:NAME_MAX].rstrip() if name else ""


def _clean(value: Any, limit: int) -> str:
    """A name or title as one tidy line: no quotation marks, spaces collapsed, at most ``limit`` long."""
    if not isinstance(value, str):
        return ""
    text = " ".join(value.translate(QUOTES).split()).strip("'‘’` ")
    return text[:limit].rstrip()


def _title(value: Any) -> str:
    """The title the model gave a section whose heading was a placeholder: "" if it gave none, or
    only another placeholder."""
    title = NUMBERING.sub("", _clean(value, TITLE_MAX)).strip()
    return "" if placeholder_title(title) else title


def _classified(section: ClassifySection, entry: dict[str, Any], subjects: _Names, topics: _Names) -> SectionClass:
    """One section's class from the model's entry for it, in the project's own wording."""
    related = entry.get("topics")
    related = [_clean(t, KNOWN_NAME_MAX) for t in related] if isinstance(related, list) else []
    related = [t for t in related if t]
    # An answer in the older style has only topics: the first of them is the primary one.
    primary = _clean(entry.get("topic"), KNOWN_NAME_MAX) or (related[0] if related else "")
    chosen: dict[str, str] = {}
    for name in map(topics.spell, [primary, *related]):
        if name:
            chosen.setdefault(name.casefold(), name)
    names = list(chosen.values())[:3]
    return SectionClass(
        id=section.id,
        kind="exercise" if entry.get("kind") == "exercise" else "text",
        subject=subjects.spell(_clean(entry.get("subject"), KNOWN_NAME_MAX)),
        topic=names[0] if names else "",
        topics=names,
        # Only a placeholder heading is given a title; a real heading is never replaced.
        title=_title(entry.get("title")) if placeholder_title(section.title) else "",
    )


def _sections_prompt(items: list[ClassifySection], subjects: list[str], topics: list[str]) -> str:
    known = ""
    if subjects:
        known += f"Subjects already in use: {json.dumps(subjects, ensure_ascii=False)}\n"
    if topics:
        known += f"Topics already in use: {json.dumps(topics, ensure_ascii=False)}\n"
    if known:
        known += "\n"
    listing = "\n\n".join(f"=== [{s.id}] {s.title} ===\n{(s.text or '')[:SECTION_OPENING]}" for s in items)
    untitled = [s.id for s in items if placeholder_title(s.title)]
    fields = '"id": "...", "subject": "...", "topic": "...", "topics": ["...", "..."], "kind": "text" | "exercise"'
    reply = f'Return JSON: {{"sections": [{{{fields}}}]}} — one entry for every section, by its id.'
    if untitled:
        reply += f' These sections have a placeholder heading and each also needs a "title": {json.dumps(untitled)}.'
    return f"{known}Sections ({len(items)}):\n\n{listing}\n\n{reply}"


async def sections(body: ClassifyRequest) -> list[SectionClass]:
    if not body.sections:
        return []
    batches = [body.sections[i : i + SECTION_BATCH] for i in range(0, len(body.sections), SECTION_BATCH)]
    gate = asyncio.Semaphore(SECTION_CONCURRENCY)

    def guess(section) -> SectionClass:
        return SectionClass(id=section.id, kind="exercise" if EXERCISE.search(section.title or "") else "text")

    async def batch(items) -> list[dict[str, Any] | None]:
        """The model's entry for each section, None where it gave none (or the call failed)."""
        user = _sections_prompt(items, body.subjects, body.topics)
        try:
            async with gate:
                data = await llm.ask_json(SECTIONS_SYSTEM, user, max_tokens=min(SECTION_TOKENS * len(items) + 500, 8000), temperature=0.2)
        except ServiceError as error:
            logger.warning("Talari: could not classify sections: %s", error.message)
            return [None] * len(items)
        found = {str(entry.get("id")): entry for entry in data.get("sections") or [] if isinstance(entry, dict)}
        return [found.get(section.id) for section in items]

    results = await asyncio.gather(*(batch(items) for items in batches))
    # The batches ran side by side, so their names are made one here, in section order.
    subjects, topics = _Names(body.subjects), _Names(body.topics)
    out = []
    for items, entries in zip(batches, results):
        for section, entry in zip(items, entries):
            out.append(_classified(section, entry, subjects, topics) if entry else guess(section))
    return out


async def classify(body: ClassifyRequest, images: Path) -> ClassifyResponse:
    found_pictures, found_sections = await asyncio.gather(pictures(body, images), sections(body))
    return ClassifyResponse(pictures=found_pictures, sections=found_sections)
