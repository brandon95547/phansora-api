"""Preliminary timeline assembly.

Given timed script beats, place one media clip per beat at the exact moment the
narration talks about it. The *timing* is deterministic (it comes straight from
each beat's start/end). The LLM is used only to turn a beat of narration into a
short, concrete visual search query so the sourced media actually matches what is
being said — done in one batched call for cost. If the LLM is unavailable the
beat's extracted keywords are used instead, so a timeline always builds.
"""
from __future__ import annotations

from typing import Dict, List

from ..models import ScriptSegment, Timeline, TimelineItem, Transition
from . import llm, media

_QUERY_SYSTEM = (
    "You turn narration beats into short visual stock-media search queries. For "
    "each beat, give 2-5 concrete, visual words describing what to SHOW on screen "
    "while that line is spoken (objects, places, scenes — not abstractions). "
    'Reply ONLY as JSON: {"queries": [{"id": "<beat id>", "query": "<terms>"}]}.'
)


def _visual_queries(segments: List[ScriptSegment]) -> Dict[str, str]:
    """Map each segment id -> a visual search query (LLM-assisted, keyword fallback)."""
    fallback = {s.id: (" ".join(s.keywords) or s.text[:60]) for s in segments}
    if not segments:
        return fallback
    try:
        beats = "\n".join(f'{s.id}: {s.text}' for s in segments)
        data = llm.generate_json(
            _QUERY_SYSTEM,
            f"Beats:\n{beats}",
            max_output_tokens=1500,
        )
        out = dict(fallback)
        for item in (data.get("queries") or []):
            sid = str(item.get("id", "")).strip()
            q = str(item.get("query", "")).strip()
            if sid in out and q:
                out[sid] = q
        return out
    except Exception:
        return fallback


def build_timeline(
    segments: List[ScriptSegment],
    *,
    voice_id: str | None = None,
    media_types: List[str] | None = None,
    per_segment: int = 1,
) -> Timeline:
    media_type = (media_types or ["image"])[0]
    queries = _visual_queries(segments)

    items: List[TimelineItem] = []
    for i, seg in enumerate(segments):
        clips = media.search_media(
            queries.get(seg.id, ""),
            segment_id=seg.id,
            media_type=media_type,
            limit=per_segment,
        )
        clip = clips[0]
        clip.start_sec = seg.start_sec
        clip.end_sec = seg.end_sec
        # First clip cuts in; the rest fade for a smoother preliminary cut.
        transition = Transition(type="cut", duration_sec=0.0) if i == 0 else Transition(type="fade", duration_sec=0.5)
        items.append(TimelineItem(segment_id=seg.id, clip=clip, transition_in=transition))

    total = segments[-1].end_sec if segments else 0.0
    return Timeline(voice_id=voice_id, items=items, total_duration_sec=total)
