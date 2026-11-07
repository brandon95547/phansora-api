"""Planning a video's scenes before anything is recorded (POST /storyboard/plan).

The AI Storyboard (services/storyboard.py) cuts a narration that already exists, timed
against the recorded voice. This is the step before it: from an idea, or a script written
but not yet read aloud, the model plans the scenes — each with a short name, the words to be
spoken over it, the picture, search terms, and how long it should hold the screen. The studio
lays them end to end; once the narration is recorded, each scene is fitted to it there.

Nothing here is timed against audio, so the lengths are the model's estimate, checked
against how long the words take to say and made to add up to the length asked for.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from . import llm
from .storyboard import (  # the storyboard's own rules, so a planned shot reads like a cut one
    _SEARCH_RULES,
    _VISUAL_RULES,
    _search_terms,
    _style_block,
    _visual_prompt,
)

logger = logging.getLogger("narrava-studio.plan")

# Documentary pacing, as in the storyboard: 4-6 seconds a scene on average, none past 10.
_TARGET_SHOT_SEC = 5.0
_MIN_SHOT_SEC = 2.0
_MAX_SHOT_SEC = 10.0
# How fast narration is read: about 150 words a minute.
_WORDS_PER_SEC = 2.5

_SYSTEM = (
    "You are an experienced documentary director planning a video before anything has been "
    "recorded. You are given either an IDEA (a topic or a brief) or a SCRIPT already written "
    "to be read aloud. Plan the video as a list of scenes, in order. A new scene starts "
    "wherever the viewer's picture should change.\n\n"
    "If you were given a SCRIPT, use its words VERBATIM and in order, split across the "
    "scenes with no gaps or overlaps, so the scenes' text joins back into the script exactly. "
    "If you were given an IDEA, write the narration yourself: clear, spoken English, in the "
    "house style, sized to the length asked for at about 150 words a minute.\n\n"
    "For each scene give:\n"
    "  name          2-4 words naming the scene for an editor's scene list.\n"
    "  text          the words spoken over it.\n"
    "  visual        the picture, per the rules below.\n"
    "  media_type    \"image\" for a still, \"video\" for motion footage.\n"
    "  search_terms  5-10 stock-media queries, per the rules below.\n"
    "  seconds       how long it holds the screen: about as long as its words take to say.\n\n"
    "PACING. A scene averages 4-6 seconds. A fast run of facts can turn over every 3-4 "
    "seconds; a heavy moment can hold 6-10. Nothing longer than 10 seconds.\n\n"
    + _VISUAL_RULES + "\n" + _SEARCH_RULES + "\n"
    'Respond with ONLY JSON of the form: {"scenes":[{"name":"...","text":"...","visual":"...",'
    '"media_type":"image|video","search_terms":["..."],"seconds":5}]}'
)


def _name_of(raw: Any, text: str, n: int) -> str:
    name = re.sub(r"\s+", " ", str(raw or "")).strip().strip(".")
    if not name:
        words = re.findall(r"[\w'’-]+", text)
        name = " ".join(words[:3])
    return (name or f"Scene {n}")[:60]


def _seconds_of(raw: Any, text: str) -> float:
    try:
        sec = float(raw)
    except (TypeError, ValueError):
        sec = 0.0
    # Never shorter than its words take to say, whatever the model estimated.
    spoken = len(re.findall(r"[\w'’-]+", text)) / _WORDS_PER_SEC
    sec = max(sec, spoken) if sec > 0 else (spoken or _TARGET_SHOT_SEC)
    return min(_MAX_SHOT_SEC, max(_MIN_SHOT_SEC, sec))


def _fit_total(scenes: List[Dict[str, Any]], total: float) -> None:
    """Scale the scenes' lengths so they add up to ``total``, in place, to a tenth of a second."""
    have = sum(s["seconds"] for s in scenes)
    if have <= 0:
        return
    k = total / have
    for s in scenes:
        s["seconds"] = round(max(0.5, s["seconds"] * k), 1)
    # Rounding drift goes on the last scene, so the sum is exactly what was asked for.
    drift = round(total - sum(s["seconds"] for s in scenes), 1)
    scenes[-1]["seconds"] = round(max(0.5, scenes[-1]["seconds"] + drift), 1)


def plan_from_idea(idea: str, total_duration_sec: float, max_scenes: int = 24,
                   style: Optional[str] = None) -> Dict[str, Any]:
    """``{"scenes": [...]}`` in PlannedScene shape, adding up to ``total_duration_sec``.

    Raises RuntimeError when the model answers nothing usable, so the caller is refunded
    rather than handed a plan that is not one."""
    text = (idea or "").strip()
    total = max(5.0, float(total_duration_sec or 0))
    limit = max(1, min(80, int(max_scenes or 24)))
    target = max(1, min(limit, round(total / _TARGET_SHOT_SEC)))
    user = _style_block(style) + (
        f"The video should run about {total:.0f} seconds: roughly {target} scenes (never more "
        f"than {limit}). Use that as a check on your pacing, not a quota.\n\n"
        f"IDEA OR SCRIPT:\n{text}"
    )
    data = llm.generate_json(_SYSTEM, user, max_output_tokens=min(16000, 1500 + limit * 360))
    raw = data.get("scenes") if isinstance(data, dict) else None
    scenes: List[Dict[str, Any]] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        words = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
        visual = _visual_prompt(item.get("visual") or item.get("visual_prompt"))
        if not words and not visual:
            continue
        scenes.append({
            "name": _name_of(item.get("name"), words, len(scenes) + 1),
            "text": words,
            "visual_prompt": visual,
            "media_type": "video" if str(item.get("media_type") or "").lower() == "video" else "image",
            "search_terms": _search_terms(item.get("search_terms")),
            "seconds": _seconds_of(item.get("seconds"), words),
        })
        if len(scenes) >= limit:
            break
    if not scenes:
        logger.warning("Scene planning: the model returned nothing usable for %r", text[:80])
        raise RuntimeError("The plan came back empty.")
    _fit_total(scenes, total)
    return {"scenes": scenes}
