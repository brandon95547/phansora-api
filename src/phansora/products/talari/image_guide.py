"""A picture made by a conversation, not a prompt.

For a background, ``designs`` first recommends background designs made for the book and its
look — borders, corner ornaments, textures, patterns, emblems, minimal grounds, scenes — for
the author to pick from or combine; the questions after it are about what that design needs.

``guide`` asks the author the next question worth asking about one picture — with answers to
pick — and says, in plain words, what the picture is so far. ``compose`` turns the answers
(and, for a change to a picture already made, its description and the change asked for) into
the description the image model draws from. That description is for the model only: the
author is shown the summary.
"""

from __future__ import annotations

import re
from typing import Any

from . import llm, prompts
from .errors import ServiceError
from .schemas import (
    ImageBrief,
    ImageComposeRequest,
    ImageComposeResponse,
    ImageDesign,
    ImageDesignsRequest,
    ImageDesignsResponse,
    ImageGuideResponse,
    ImageQuestion,
)

# Questions in one conversation, at most; most pictures need fewer.
MAX_QUESTIONS = 6
SUGGESTIONS = 4
TEMPERATURE = {"guide": 0.8, "compose": 0.7, "designs": 0.95}
KINDS = set(prompts.DESIGN_KINDS)
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
# Words that give the machinery away, never put to an author.
JARGON = re.compile(r"\b(prompt|render(ing)?|aspect ratio|seed|negative|diffusion|stable|flux|comfy)\b", re.IGNORECASE)


def _text(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit].strip()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:40]


def _question(data: Any, request: ImageBrief) -> ImageQuestion | None:
    if not isinstance(data, dict):
        return None
    text = _text(data.get("text"), 200)
    if not text or JARGON.search(text):
        return None
    qid = _slug(_text(data.get("id"), 60)) or _slug(text)
    # Never the same question twice: by its name, or by its words.
    asked_ids = {a.id for a in request.answers}
    asked_words = {_slug(a.question) for a in request.answers}
    if qid in asked_ids or _slug(text) in asked_words:
        return None
    seen: set[str] = set()
    suggestions = []
    for item in data.get("suggestions") if isinstance(data.get("suggestions"), list) else []:
        s = _text(item, 80)
        key = s.lower()
        if s and key not in seen and "decide" not in key and not JARGON.search(s):
            seen.add(key)
            suggestions.append(s)
    return ImageQuestion(id=qid, text=text, suggestions=suggestions[:SUGGESTIONS])


async def guide(request: ImageBrief) -> ImageGuideResponse:
    remaining = MAX_QUESTIONS - len(request.answers)
    system, user = prompts.image_guide(request, max(0, remaining))
    data = await llm.ask_json(system, user, max_tokens=700, temperature=TEMPERATURE["guide"])
    summary = _text(data.get("summary"), 500)
    question = _question(data.get("question"), request) if remaining > 0 else None
    return ImageGuideResponse(question=question, summary=summary)


async def designs(request: ImageDesignsRequest) -> ImageDesignsResponse:
    """Background designs for this book and its look: varied, each with its kinds and colors."""
    system, user = prompts.image_designs(request)
    data = await llm.ask_json(system, user, max_tokens=1000, temperature=TEMPERATURE["designs"])
    seen = {x.strip().lower() for x in request.exclude}
    out: list[ImageDesign] = []
    for item in data.get("designs") if isinstance(data.get("designs"), list) else []:
        if not isinstance(item, dict):
            continue
        title = _text(item.get("title"), 80)
        if not title or title.lower() in seen or JARGON.search(title):
            continue
        kinds = [k for k in (item.get("kinds") if isinstance(item.get("kinds"), list) else []) if k in KINDS][:3] or ["scene"]
        palette = [str(c) for c in (item.get("palette") if isinstance(item.get("palette"), list) else []) if HEX.match(str(c))][:3]
        seen.add(title.lower())
        out.append(ImageDesign(title=title, description=_text(item.get("description"), 240), kinds=kinds, palette=palette))
    if not out:
        raise ServiceError("DeepSeek returned no background designs.")
    return ImageDesignsResponse(designs=out[: request.count])


async def compose(request: ImageComposeRequest) -> ImageComposeResponse:
    # Drawing the same picture again needs no new words.
    if request.previous.strip() and not request.instruction.strip():
        return ImageComposeResponse(prompt=request.previous.strip()[:4000], caption="", summary="", shape="square")
    system, user = prompts.image_compose(request)
    data = await llm.ask_json(system, user, max_tokens=700, temperature=TEMPERATURE["compose"])
    prompt = _text(data.get("prompt"), 3000)
    if not prompt:
        raise ServiceError("DeepSeek returned no picture description.")
    shape = data.get("shape") if data.get("shape") in ("portrait", "square", "landscape") else "square"
    return ImageComposeResponse(prompt=prompt, caption=_text(data.get("caption"), 80), summary=_text(data.get("summary"), 500), shape=shape)
