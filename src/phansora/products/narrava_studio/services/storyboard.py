"""AI Storyboard — a documentary editor's first visual pass over the narration.

Given the narration text and its real rendered duration on the timeline, an LLM
decides where the VISUAL should change based on the flow of the story — a new idea,
person, place, event, time period, or visual concept — NOT fixed intervals or every
sentence. Each returned scene carries media suggestions (search terms + visual ideas)
that the editor's sidebar turns into real fair-use results on demand.

Timing is computed here, deterministically: the model never guesses seconds. Each
scene's length is its share of the total word count, spread across
``total_duration_sec`` so the first scene starts at 0 and the last ends exactly at
the narration's end. That keeps the placeholders aligned with the narration under it.
"""
from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List

from ..models import StoryboardScene
from . import llm, script

# Below this, a scene is too short to be a distinct shot — merge it into the previous
# one so the editor isn't handed a strip of one-second placeholders.
_MIN_SCENE_SEC = 1.5

_SYSTEM = (
    "You are an experienced documentary editor planning the VISUALS for a narration. "
    "You are given the full narration script. Break it into an ordered list of scenes, "
    "where each scene is a stretch of narration that a single shot or image would cover. "
    "Start a new scene ONLY where an editor would naturally change the visual: when the "
    "narration introduces a new idea, person, place, event, time period, or visual "
    "concept. Do NOT cut on a fixed interval and do NOT start a new scene for every "
    "sentence — a scene often spans several sentences that share one visual subject. "
    "The first scene must cover the very beginning of the narration. "
    "For each scene, give the exact narration text it covers (verbatim, in order, with no "
    "gaps or overlaps so the pieces concatenate back to the original), a one-line rationale "
    "for why the visual changes there, whether it wants a still image or motion footage, "
    "5-10 STOCK-MEDIA SEARCH TERMS, and 2-3 short VISUAL IDEAS describing what to show. "
    "\n\n"
    "The SEARCH TERMS are typed into stock media sites (Pixabay, Pexels, Unsplash, Wikimedia "
    "Commons), so write them the way someone SEARCHES, not the way you would describe the "
    "scene. Rules for every term: "
    "(1) 1-3 common English words — never a phrase from the narration; "
    "(2) name a VISIBLE subject, object, place, or setting that a camera could point at; "
    "(3) no abstract concepts, emotions, time spans, or storytelling language; "
    "(4) prefer plain, common wording that returns MANY results over precise wording that "
    "returns none. "
    "\n"
    'Example narration: "In the shadowed edges of European folklore, where dense forests '
    'swallowed light and abandoned mines echoed with unseen movement, stories of goblins '
    'endured for centuries." '
    "\n"
    "GOOD terms: goblin, dark forest, misty forest, abandoned mine, forest at night, "
    "medieval village, ancient castle, cave tunnel. "
    "\n"
    "BAD terms: european folklore woods, shadowed edges of folklore, unseen movement, "
    "stories of goblins endured. "
    "\n\n"
    'Respond with ONLY JSON of the form: {"scenes":[{"text":"...","rationale":"...",'
    '"media_type":"image|video","search_terms":["..."],"visual_ideas":["..."]}]}'
)


def build_storyboard(full_text: str, total_duration_sec: float, max_scenes: int = 24) -> List[StoryboardScene]:
    text = (full_text or "").strip()
    total = max(0.1, float(total_duration_sec or 0.0))
    if not text:
        return []

    raw_scenes = _ask_llm(text, max_scenes)
    if not raw_scenes:
        raw_scenes = [_fallback_scene(text)]

    return _lay_out(raw_scenes, total)


def _ask_llm(text: str, max_scenes: int) -> List[Dict[str, Any]]:
    user = (
        f"Plan at most {max_scenes} scenes for this narration. Keep scenes substantial — "
        f"prefer fewer, meaningful visual changes over many tiny ones.\n\nNARRATION:\n{text}"
    )
    try:
        data = llm.generate_json(_SYSTEM, user, max_output_tokens=3000)
    except Exception:  # noqa: BLE001 — any LLM/parse failure degrades to one scene
        return []
    scenes = data.get("scenes") if isinstance(data, dict) else None
    if not isinstance(scenes, list):
        return []

    cleaned: List[Dict[str, Any]] = []
    for s in scenes:
        if not isinstance(s, dict):
            continue
        span = str(s.get("text") or "").strip()
        if not span:
            continue
        cleaned.append({
            "text": span,
            "rationale": str(s.get("rationale") or "").strip()[:280],
            "media_type": "video" if str(s.get("media_type") or "").lower() == "video" else "image",
            "search_terms": _search_terms(s.get("search_terms")),
            "visual_ideas": _str_list(s.get("visual_ideas"), limit=3),
        })
        if len(cleaned) >= max_scenes:
            break
    return cleaned


def _fallback_scene(text: str) -> Dict[str, Any]:
    """When the model gives us nothing usable, one placeholder over the whole narration
    with heuristic search terms is still a useful (if coarse) first pass."""
    return {
        "text": text,
        "rationale": "Whole-narration placeholder — regenerate for a finer pass.",
        "media_type": "image",
        "search_terms": script.extract_keywords(text, limit=8),
        "visual_ideas": [],
    }


def _lay_out(scenes: List[Dict[str, Any]], total: float) -> List[StoryboardScene]:
    """Spread the scenes across ``total`` seconds by word share, then merge any scene
    shorter than the minimum into its predecessor. First starts at 0; last ends at total."""
    counts = [max(1, script._word_count(s["text"])) for s in scenes]
    words = sum(counts) or 1

    # Provisional [start, end] by cumulative word share.
    bounds: List[List[float]] = []
    cursor = 0.0
    for i, c in enumerate(counts):
        start = cursor
        end = total if i == len(scenes) - 1 else round(start + (c / words) * total, 2)
        bounds.append([start, max(end, start)])
        cursor = end

    # Merge too-short scenes forward into the previous one (keeping the earlier scene's
    # metadata — it owns the visual — and extending its end).
    merged: List[Dict[str, Any]] = []
    merged_bounds: List[List[float]] = []
    for s, (start, end) in zip(scenes, bounds):
        if merged and (end - start) < _MIN_SCENE_SEC:
            merged_bounds[-1][1] = end
            merged[-1]["text"] = f"{merged[-1]['text']} {s['text']}".strip()
        else:
            merged.append(dict(s))
            merged_bounds.append([start, end])
    if merged_bounds:
        merged_bounds[-1][1] = total  # guard against drift from merges

    out: List[StoryboardScene] = []
    for i, (s, (start, end)) in enumerate(zip(merged, merged_bounds)):
        out.append(StoryboardScene(
            id=f"sb_{uuid.uuid4().hex[:8]}",
            index=i,
            text=s["text"],
            start_sec=round(start, 2),
            end_sec=round(end, 2),
            rationale=s["rationale"],
            media_type=s["media_type"],
            search_terms=s["search_terms"] or script.extract_keywords(s["text"], limit=8),
            visual_ideas=s["visual_ideas"],
        ))
    return out


def _str_list(value: Any, *, limit: int) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        s = str(item or "").strip()
        if s:
            out.append(s[:80])
        if len(out) >= limit:
            break
    return out


# Words that carry no visual meaning on a stock site — a term made only of these (or one
# that ends on one after trimming) is useless as a query.
_CONNECTORS = {
    "of", "the", "a", "an", "in", "on", "at", "with", "and", "or", "for", "to", "from",
    "by", "into", "over", "under", "as", "that", "this", "these", "those", "its", "their",
}
_MAX_TERM_WORDS = 3
_MAX_TERMS = 10


def _search_terms(value: Any) -> List[str]:
    """Normalize the model's search terms into things a stock media site can actually match.

    The prompt asks for 1-3 plain words, but models still slip in narration phrases. Trimming
    such a phrase to its first 3 words can strand a connector ("shadowed edges of"), which
    searches worse than useless — so trailing/leading connectors are stripped and anything
    left empty (or purely connectors) is dropped. Deduped case-insensitively.
    """
    raw = _str_list(value, limit=_MAX_TERMS * 2)  # over-fetch: cleaning discards some
    out: List[str] = []
    seen: set = set()
    for term in raw:
        words = re.sub(r"[^\w\s'-]", " ", term).split()[:_MAX_TERM_WORDS]
        while words and words[0].lower() in _CONNECTORS:
            words.pop(0)
        while words and words[-1].lower() in _CONNECTORS:
            words.pop()
        if not words:
            continue
        cleaned = " ".join(words)
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= _MAX_TERMS:
            break
    return out
