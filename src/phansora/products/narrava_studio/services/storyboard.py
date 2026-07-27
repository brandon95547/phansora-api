"""AI Storyboard — a documentary editor's first visual pass over the narration.

Edit to ideas, not sentences. The unit is a VISUAL BEAT: a stretch of narration over which
the viewer's mental picture does not change. One sentence often holds several beats and is
split mid-sentence; several sentences often share one beat and are not split at all. Cuts
never fall on a fixed interval or on a full stop for its own sake.

Each beat carries a written description of the picture, complete enough for an image or
video generator to render from it alone, plus search terms the editor's sidebar turns into
real fair-use results on demand.

The model never guesses seconds — it chooses WHERE in the text a visual should change,
and the audio decides WHEN. Each scene is anchored back into the real narration text to
get a character offset, and measured word timings (services/align.py) turn that offset
into a second. Without those timings nothing is laid out at all: a placeholder that is
only nearly on the voice is the bug, not a cheaper version of the right answer.
"""
from __future__ import annotations

import logging
import re
import uuid
from bisect import bisect_right
from typing import Any, Dict, List, Optional, Tuple

from ..models import StoryboardScene
from . import llm, script
from . import styles as doc_styles
from .align import WORD_RE as _WORD_RE, normalize as _normalize

logger = logging.getLogger("narrava-studio.storyboard")

# Documentary pacing. A visual beat averages 4-6 seconds; a fast run of facts can turn over
# every 3-4, a heavy or dramatic moment can hold 6-10, and nothing sits past 10. These are
# the shape of the edit, not a metronome — where the cut falls is decided by where the
# picture changes, and only the outer bounds are enforced here.
_TARGET_SHOT_SEC = 5.0
_MAX_SHOT_SEC = 10.0

# The fastest pace worth cutting to when re-cutting a long stretch; below this the beats
# stop reading as separate images and start reading as a flicker.
_MIN_REFINED_SHOT_SEC = 3.0

# Below this a placeholder is a flash rather than a shot, so it folds into the one before
# it. Kept under the 3-4s "fast factual" floor on purpose: a deliberate quick cutaway is
# the editor's call to make, and this only catches the degenerate cases.
_MIN_SCENE_SEC = 2.0

# How far ahead of a word's measured start to place the cut that belongs to it. Covers the
# late bias in whisper's word timestamps; see _clock_for.
_CUT_LEAD_SEC = 0.12

# The editing philosophy, stated the way a director would state it. The rules that matter
# most are the two the model gets wrong by default: it will cut on full stops unless told
# not to, and it will describe the narration back to you instead of describing a picture.
_EDIT_RULES = (
    "EDIT TO IDEAS, NOT SENTENCES. This is the whole job. After every phrase, ask: has the "
    "viewer's mental image changed? If it has, start a new scene. If it has not, stay on "
    "the one you are on.\n"
    "  - A single sentence usually carries more than one image. SPLIT IT, mid-sentence, at "
    "the exact word where the picture turns. Do not wait for the full stop.\n"
    "  - Several sentences that describe the same image are ONE scene. Do not start a new "
    "scene merely because a sentence ended.\n"
    "  - Never cut on a fixed interval, and never cut because a scene 'feels due'.\n"
    "  - A new scene means: a new place, person, object, action, event, time period, or a "
    "new idea the viewer would picture differently.\n"
    "  - Cut where the phrase can breathe. Splitting mid-sentence is right; splitting "
    "mid-phrase is not. Never break between an article, preposition, conjunction or "
    "adjective and the word it belongs to. WRONG: \"the concerns behind the\" / "
    "\"allegations disappear\". RIGHT: \"the concerns behind the allegations\" / "
    "\"do not disappear\". Read your split points aloud — if the first half ends on a word "
    "left dangling, move the cut.\n"
    "\n"
    "PACING. A scene averages 4-6 seconds of narration. A fast run of facts can turn over "
    "every 3-4 seconds; a heavy, emotional or dramatic moment can hold for 6-10. Nothing "
    "may sit longer than 10 seconds — if a stretch is running long under one image, there "
    "is a beat inside it you have not found yet. Let the pacing vary with the material; "
    "an evenly-spaced storyboard is a sign you cut on the clock instead of the meaning.\n"
)

# The `visual` field is the deliverable — it is handed to an image/video generator with no
# other context, so it has to stand alone, and it must never be the narration restated.
_VISUAL_RULES = (
    "The `visual` field is what the viewer SEES. It is handed to an image or video "
    "generator with nothing else to go on, so it must stand entirely on its own. Describe, "
    "in one or two sentences: the primary subject; the action taking place; the "
    "environment or location; the mood and atmosphere; camera framing where it helps "
    "(wide, close-up, overhead, tracking); and period-accurate clothing, architecture and "
    "objects whenever the subject is historical.\n"
    "NEVER quote, paraphrase or narrate the script in the `visual` field. Describe the "
    "picture, not the words. 'A story spreading faster than facts' is not a picture. "
    "'Close-up of a phone screen as a headline is shared, reshared and multiplies across a "
    "feed, cold blue light on a face in a darkened room' is a picture.\n"
)

_SEARCH_RULES = (
    "The SEARCH TERMS are typed into stock media sites (Pixabay, Pexels, Unsplash, Wikimedia "
    "Commons), so write them the way someone SEARCHES, not the way you would describe the "
    "scene. Rules for every term: "
    "(1) 1-3 common English words — never a phrase from the narration; "
    "(2) name a VISIBLE subject, object, place, or setting that a camera could point at; "
    "(3) no abstract concepts, emotions, time spans, or storytelling language; "
    "(4) prefer plain, common wording that returns MANY results over precise wording that "
    "returns none.\n"
    'Example narration: "In the shadowed edges of European folklore, where dense forests '
    'swallowed light and abandoned mines echoed with unseen movement, stories of goblins '
    'endured for centuries."\n'
    "GOOD terms: goblin, dark forest, misty forest, abandoned mine, forest at night, "
    "medieval village, ancient castle, cave tunnel.\n"
    "BAD terms: european folklore woods, shadowed edges of folklore, unseen movement, "
    "stories of goblins endured.\n"
)

_FIELDS = (
    "For each scene give:\n"
    "  text          the narration this visual covers — VERBATIM, in order, with no gaps "
    "and no overlaps, so the pieces concatenate back into the original script exactly. It "
    "may begin and end mid-sentence; that is normal and expected. No scene numbers, labels "
    "or prefixes.\n"
    "  visual        the picture, per the rules above.\n"
    "  rationale     one short line: why the picture changes here.\n"
    "  media_type    \"image\" for a still, \"video\" for motion footage.\n"
    "  search_terms  5-10 stock-media queries, per the rules above.\n"
)

# ── House style ──────────────────────────────────────────────────────────────
# The style is chosen once for the whole documentary, next to the documentary type, and the
# same entry is read by the script writer (services/script.py) and by every call below. It
# lives in services/styles.py rather than here precisely because it is no longer the
# storyboard's private setting: two copies of the direction would be two films.
#
# Re-exported under the names this module has always used, so nothing downstream (or in the
# tests) has to care where the registry moved to.
STYLES = doc_styles.DOC_STYLES
_style_block = doc_styles.visual_block


_SYSTEM = (
    "You are an experienced documentary editor deciding what the viewer sees while a "
    "narration plays. You are given the script. Break it into visual beats and describe "
    "the picture for each one.\n\n"
    + _EDIT_RULES + "\n" + _FIELDS + "\n" + _VISUAL_RULES + "\n" + _SEARCH_RULES + "\n"
    "The first scene must cover the very beginning of the narration.\n\n"
    'Respond with ONLY JSON of the form: {"scenes":[{"text":"...","visual":"...",'
    '"rationale":"...","media_type":"image|video","search_terms":["..."]}]}'
)


def build_storyboard(
    full_text: str,
    total_duration_sec: float,
    max_scenes: int = 24,
    word_times: Optional[List[Tuple[float, float]]] = None,
    style: Optional[str] = None,
) -> List[StoryboardScene]:
    """``word_times`` is one (start, end) per word of ``full_text``, measured from the
    rendered audio (see services/align.py). Required: without it, or with a count that does
    not match the script, this raises NarrationNotTimed rather than approximating.

    ``style`` is one of STYLES — the documentary's house style, the same one the narration
    was written in. It reaches the model, where it shapes what the pictures show and how
    fast they turn over; it never reaches the layout below, which is arithmetic over the
    model's spans and the measured clock."""
    text = (full_text or "").strip()
    total = max(0.1, float(total_duration_sec or 0.0))
    if not text:
        return []

    raw_scenes = _ask_llm(text, max_scenes, total, style)
    if not raw_scenes:
        raw_scenes = [_fallback_scene(text)]

    return _lay_out(raw_scenes, total, text, word_times, style=style)


def _ask_llm(text: str, max_scenes: int, total: float, style: Optional[str] = None) -> List[Dict[str, Any]]:
    # The model cannot pace beats without knowing how long the narration runs — asked
    # blind it returns a handful of chapter-sized scenes whatever the length. Give it the
    # duration and the beat count that implies, as a sanity check on its own pacing rather
    # than as a quota to fill.
    target = max(1, min(max_scenes, round(total / _TARGET_SHOT_SEC)))
    user = _style_block(style) + (
        f"This narration runs about {total:.0f} seconds of speech. At documentary pace that "
        f"is roughly {target} visual beats (never more than {max_scenes}). Use it as a check "
        f"on your own pacing, not a quota: cut where the picture changes, and if you find "
        f"yourself far under that number you are holding images too long.\n\n"
        f"NARRATION:\n{text}"
    )
    return _scenes_from(_ask_json(user, max_scenes), max_scenes)


def _ask_json(user: str, scene_budget: int, system: str = None, budget: int = None) -> Any:
    """One JSON call, sized for how much writing the answer needs."""
    # Each scene now carries a written visual, so the ceiling has to scale with the count
    # or a long narration gets truncated mid-array and parses to nothing. `budget` lets a
    # caller size it directly when the unit of work is not a scene (see _ask_ideas).
    budget = budget or min(16000, 1500 + scene_budget * 320)
    try:
        return llm.generate_json(system or _SYSTEM, user, max_output_tokens=budget)
    except Exception:  # noqa: BLE001 — any LLM/parse failure degrades to one scene
        logger.exception("Storyboard LLM call failed")
        return None


# Second pass. The first pass has to guess how long its beats run — it has the words but
# not the voice. Once the narration is measured we know exactly, so anything still holding
# too long goes back to the editor WITH its real duration attached. Finding the beat inside
# a long stretch is a judgement about meaning, and belongs to the thing that understands
# meaning; the mechanical splitter below is only there for when this cannot answer.
_REFINE_SYSTEM = (
    "You are an experienced documentary editor. Each stretch of narration below is "
    "currently covered by a SINGLE image, and has been measured against the recorded "
    "voice — the exact number of seconds it holds the screen is given. Every one of them "
    "holds too long. Find the visual beats inside each stretch and break it up.\n\n"
    + _EDIT_RULES + "\n" + _FIELDS + "\n" + _VISUAL_RULES + "\n" + _SEARCH_RULES + "\n"
    "Within a stretch your scenes must cover its text completely and in order, verbatim, "
    "with no gaps or overlaps; the first scene begins at the stretch's first word. Splitting "
    "mid-sentence is expected — that is usually where the picture actually turns.\n\n"
    'Respond with ONLY JSON of the form: {"stretches":[{"id":1,"scenes":[{"text":"...",'
    '"visual":"...","rationale":"...","media_type":"image|video","search_terms":["..."]}]}]}'
)

# Beyond this many over-long stretches, re-asking costs more than it returns and the prompt
# stops being a focused question. The mechanical splitter takes the rest.
_MAX_REFINE_STRETCHES = 12


def _refine_long(
    full_text: str,
    scenes: List[Dict[str, Any]],
    spans: List[List[int]],
    clock: "_Clock",
    style: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[List[int]]]:
    """Re-cut the scenes that measure too long, by asking rather than by regex.

    Styled too: this pass writes the `visual` for the sub-beats it creates, and a storyboard
    where the re-cut scenes are unstyled and the rest are not would read as an inconsistency
    in the edit rather than as the seam it actually is."""
    lengths = [clock.at(b) - clock.at(a) for a, b in spans]
    targets = [i for i, sec in enumerate(lengths) if sec > _MAX_SHOT_SEC]
    if not targets:
        return scenes, spans
    if len(targets) > _MAX_REFINE_STRETCHES:
        logger.info("Storyboard: %d over-long scenes, more than one question's worth — "
                    "leaving them to the splitter", len(targets))
        return scenes, spans

    parts, budget = [], 0
    for n, i in enumerate(targets, start=1):
        a, b = spans[i]
        parts.append(f"STRETCH {n} ({lengths[i]:.1f} seconds):\n{full_text[a:b].strip()}")
        budget += max(2, round(lengths[i] / _TARGET_SHOT_SEC))
    data = _ask_json(_style_block(style) + "\n\n".join(parts), budget, system=_REFINE_SYSTEM)

    by_stretch: Dict[int, List[Dict[str, Any]]] = {}
    for entry in (data.get("stretches") if isinstance(data, dict) else None) or []:
        if not isinstance(entry, dict):
            continue
        try:
            sid = int(entry.get("id"))
        except (TypeError, ValueError):
            continue
        # No more beats than the fastest documentary pace would justify for its length.
        idx = targets[sid - 1] if 0 < sid <= len(targets) else None
        if idx is None:
            continue
        sub = _scenes_from(entry, max(2, round(lengths[idx] / _MIN_REFINED_SHOT_SEC)))
        if len(sub) >= 2:
            by_stretch[sid] = sub

    out_scenes: List[Dict[str, Any]] = []
    out_spans: List[List[int]] = []
    for i, (scene, (a, b)) in enumerate(zip(scenes, spans)):
        sub = by_stretch.get(targets.index(i) + 1) if i in targets else None
        # Anchoring the sub-beats INSIDE the parent's own text keeps them from wandering
        # off into a similar phrase elsewhere in the narration.
        placed = _anchor_spans(full_text[a:b], sub) if sub else None
        if not placed or len(placed) < 2:
            out_scenes.append(scene)
            out_spans.append([a, b])
            continue
        for sub_scene, sub_a, sub_b in placed:
            out_scenes.append(sub_scene)
            out_spans.append([a + sub_a, a + sub_b])

    logger.info("Storyboard: re-cut %d over-long scene(s) into %d",
                len(by_stretch), len(out_scenes) - len(scenes) + len(by_stretch))
    return out_scenes, out_spans


def _scenes_from(data: Any, limit: int) -> List[Dict[str, Any]]:
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
        cleaned.append(_clean_scene(s, span))
        if len(cleaned) >= limit:
            break
    return cleaned


def _clean_scene(s: Dict[str, Any], span: str) -> Dict[str, Any]:
    return {
        "text": span,
        "visual_prompt": _visual_prompt(s.get("visual")),
        "rationale": str(s.get("rationale") or "").strip()[:280],
        "media_type": "video" if str(s.get("media_type") or "").lower() == "video" else "image",
        "search_terms": _search_terms(s.get("search_terms")),
        # Superseded by visual_prompt; still read so a model that answers in the old shape
        # (or an older saved project being re-suggested) does not come back empty-handed.
        "visual_ideas": _str_list(s.get("visual_ideas"), limit=3),
    }


def _visual_prompt(value: Any) -> str:
    """The generator-facing description. A list is accepted and joined — models hand back
    ["subject", "mood", ...] often enough that rejecting it would lose real work."""
    if isinstance(value, list):
        value = ". ".join(str(v).strip() for v in value if str(v).strip())
    return re.sub(r"\s+", " ", str(value or "")).strip()[:900]


def _fallback_scene(text: str) -> Dict[str, Any]:
    """When the model gives us nothing usable, one placeholder over the whole narration
    with heuristic search terms is still a useful (if coarse) first pass."""
    return {
        "text": text,
        "visual_prompt": "",
        "rationale": "Whole-narration placeholder — regenerate for a finer pass.",
        "media_type": "image",
        "search_terms": script.extract_keywords(text, limit=8),
        "visual_ideas": [],
    }


# ── Alternative pictures for one scene ───────────────────────────────────────
# The sidebar shows one idea at a time and steps between them with arrows, so what it
# needs is several DIFFERENT answers to the same question — not one answer elaborated.
#
# Three, not five. Every extra idea is another few hundred reasoning tokens on a call the
# sidebar fires the moment a scene is opened, and the long ones were timing out at the
# proxy — which the editor sees as "the narration service isn't responding" on a scene that
# would have been fine with three. Arrowing past three is rare; waiting is not.
_MIN_IDEAS = 3
_MAX_IDEAS = 8

_IDEAS_SYSTEM = (
    "You are an experienced documentary editor. You are given ONE scene of a documentary "
    "that is already cut, and the narration the viewer hears over it. Propose several "
    "different pictures that could fill this one scene.\n\n"
    "They are ALTERNATIVES, not a sequence. Each one stands alone as the entire shot and "
    "the editor will choose exactly one, so make them genuinely different from each "
    "other — a different subject, setting, era, scale, or camera treatment — never the "
    "same picture reworded. Order them best-first.\n\n"
    "For each idea give:\n"
    "  visual        the picture, per the rules below.\n"
    "  media_type    \"image\" for a still, \"video\" for motion footage.\n"
    "  search_terms  5-10 stock-media queries for THAT idea specifically, per the rules "
    "below. Two ideas must not carry the same terms — the terms are how the editor finds "
    "that particular picture.\n\n"
    + _VISUAL_RULES + "\n" + _SEARCH_RULES + "\n"
    'Respond with ONLY JSON of the form: {"ideas":[{"visual":"...",'
    '"media_type":"image|video","search_terms":["..."]}]}'
)


def _ask_ideas(user: str, count: int, style: Optional[str] = None) -> Any:
    """The ideas call, sized per idea rather than per scene.

    The ceiling covers reasoning tokens as well as visible output, and a truncated array
    parses to nothing rather than to fewer ideas — so this budgets generously. Roughly
    400 tokens an idea against ~110 of visible text is deliberate headroom.

    Returns the ideas LIST, not the envelope the model answers with. `generate_json` hands
    back the whole `{"ideas": [...]}` object, and unwrapping here mirrors what `_scenes_from`
    does for the storyboard call — the alternative, handing the envelope to `_clean_ideas`,
    silently yields zero ideas for every scene however good the model's answer was.
    """
    data = _ask_json(_style_block(style) + user, 1, system=_IDEAS_SYSTEM,
                     budget=min(16000, 1500 + count * 400))
    return data.get("ideas") if isinstance(data, dict) else data


def _clean_ideas(value: Any, *, span: str, limit: int) -> List[Dict[str, Any]]:
    """Normalize the model's ideas into SceneIdea shape, dropping near-duplicates.

    The dedupe is an OR, not an AND: either the same normalized term set or the same
    opening words is enough to drop one. Requiring both would only catch verbatim clones,
    which is the case a model is least likely to produce — what it actually produces is
    "archival newsreel of a crowd" next to "archival footage of a crowd", two arrow
    positions that return the same media.

    Accepts either the bare list or the `{"ideas": [...]}` envelope. Tolerating both is
    deliberate: passing the envelope in produces an empty result that is indistinguishable
    from a real model failure, so the shape mismatch hides instead of announcing itself.
    """
    if isinstance(value, dict):
        value = value.get("ideas")
    raw = value if isinstance(value, list) else []
    out: List[Dict[str, Any]] = []
    seen_terms: set = set()
    seen_openings: set = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        # 400, not _visual_prompt's 900: five long paragraphs per scene is a lot of project
        # JSON, and a tighter line is easier to compare when you are scanning several.
        prompt = _visual_prompt(item.get("visual") or item.get("visual_prompt"))[:400]
        terms = _search_terms(item.get("search_terms"))
        if not prompt and not terms:
            continue
        # An idea with no terms would blank the media grid the moment the user arrows onto
        # it, so derive some rather than discarding real direction.
        if not terms:
            terms = script.extract_keywords(prompt or span, limit=8)
        if not terms:
            continue
        term_key = tuple(sorted(t.lower() for t in terms))
        opening = " ".join(prompt.lower().split()[:6])
        if term_key in seen_terms or (opening and opening in seen_openings):
            continue
        seen_terms.add(term_key)
        if opening:
            seen_openings.add(opening)
        out.append({
            "visual_prompt": prompt,
            "search_terms": terms,
            "media_type": "video" if str(item.get("media_type") or "").lower() == "video" else "image",
        })
        if len(out) >= limit:
            break
    return out


def suggest_for_span(text: str, count: int = _MIN_IDEAS, have: List[str] = None,
                     style: Optional[str] = None) -> Dict[str, Any]:
    """Alternative pictures for one already-placed scene — no timing, no re-cutting.

    The storyboard sidebar wants several options for a placeholder whose position is
    already correct. Routing that through build_storyboard would demand word timings it
    has no way to supply, and would re-cut a scene the user did not ask to re-cut.

    The flat visual_prompt/media_type/search_terms in the result mirror ideas[0], so a
    caller written before ideas existed reads exactly what it used to. `degraded` is the
    honest signal: without it a total LLM outage returns the same shape as a good answer,
    and the client would record "this scene only has one idea" permanently.
    """
    span = (text or "").strip()
    if not span:
        return {**_fallback_scene(""), "ideas": [], "degraded": True}

    want = max(1, min(_MAX_IDEAS, int(count or _MIN_IDEAS)))
    avoid = [str(h).strip() for h in (have or []) if str(h).strip()][:_MAX_IDEAS]
    user = (
        "This is ONE scene of a documentary that is already cut — do not split it and do "
        f"not change its narration. Give me {want} different pictures that could fill it.\n\n"
        f"NARRATION FOR THIS SCENE:\n{span}"
    )
    if avoid:
        listed = "\n".join(f"- {a}" for a in avoid)
        user += (
            "\n\nThe editor has already seen the ideas below. Do not repeat or reword them "
            f"— propose {want} genuinely different ones:\n{listed}"
        )

    ideas = _clean_ideas(_ask_ideas(user, want, style), span=span, limit=want)
    if not ideas:
        # The model failed or answered unusably. Say so rather than passing a fallback off
        # as a real result. Logged because the endpoint answers 200 either way: without this
        # line the only trace of a failure here is a "no ideas" message in the browser.
        logger.warning("Scene ideas: model returned nothing usable for span %r", span[:80])
        return {**_fallback_scene(span), "text": span, "ideas": [], "degraded": True}

    first = ideas[0]
    return {
        "text": span,          # its text is fixed; only the direction is new
        "visual_prompt": first["visual_prompt"],
        "media_type": first["media_type"],
        "search_terms": first["search_terms"],
        "rationale": "",       # nothing renders it any more; kept so the field still exists
        "visual_ideas": [],
        "ideas": ideas,
        "degraded": len(ideas) < want,
    }


def _lay_out(
    scenes: List[Dict[str, Any]],
    total: float,
    full_text: str,
    word_times: Optional[List[Tuple[float, float]]] = None,
    style: Optional[str] = None,
) -> List[StoryboardScene]:
    """Anchor each scene into the real narration, then read its bounds off the clock.

    Every boundary in here — scene and shot alike — is a character offset in the narration
    turned into a second by the same clock, so the measured and estimated paths can never
    produce differently-shaped output. First starts at 0; last ends at total.
    """
    clock = _clock_for(full_text, total, word_times)
    placed = _anchor_spans(full_text, scenes) or [(scenes[0], 0, len(full_text))]

    # Merge too-short scenes forward into the previous one (keeping the earlier scene's
    # metadata — it owns the visual — and extending its span).
    merged: List[Dict[str, Any]] = []
    merged_spans: List[List[int]] = []
    for s, start_ch, end_ch in placed:
        if merged and (clock.at(end_ch) - clock.at(start_ch)) < _MIN_SCENE_SEC:
            merged_spans[-1][1] = end_ch
        else:
            merged.append(dict(s))
            merged_spans.append([start_ch, end_ch])
    if merged_spans:
        merged_spans[-1][1] = len(full_text)  # the last scene always runs to the end

    # Anything still holding too long goes back to the editor with its measured duration —
    # finding the beat inside a long stretch is a judgement about meaning.
    merged, merged_spans = _refine_long(full_text, merged, merged_spans, clock, style=style)

    # Last resort only. If the re-cut did not happen or did not go far enough, split on
    # grammar so no placeholder sits on screen indefinitely. This cuts on sentences and
    # clauses, which is exactly what the edit is not supposed to do, so it is a floor under
    # the worst case rather than part of the design.
    shots: List[Dict[str, Any]] = []
    shot_bounds: List[List[float]] = []
    for s, (start_ch, end_ch) in zip(merged, merged_spans):
        for n, (a_ch, b_ch) in enumerate(_shots(full_text, start_ch, end_ch, clock)):
            # The anchored span is what is actually spoken here; the model's own `text` is
            # only a fallback for a scene we could not place in the narration.
            text = full_text[a_ch:b_ch].strip() or s["text"]
            if n == 0:
                shots.append({**s, "text": text})
            else:
                # A grammar-split continuation inherits the visual it was cut out of, which
                # would render as the same picture twice. Say so, in the field the generator
                # actually reads, so the second slot at least varies.
                visual = s.get("visual_prompt") or ""
                shots.append({
                    **s, "text": text,
                    "visual_prompt": (visual + " Alternative angle or framing of the same "
                                               "subject.").strip() if visual else visual,
                    "rationale": "Continues the previous shot's subject — this one was split "
                                 "on grammar to keep it off the screen too long, so give it a "
                                 "different angle or replace it.",
                })
            shot_bounds.append([clock.at(a_ch), clock.at(b_ch)])
    if shot_bounds:
        shot_bounds[0][0] = 0.0
        shot_bounds[-1][1] = total

    out: List[StoryboardScene] = []
    for i, (s, (start, end)) in enumerate(zip(shots, shot_bounds)):
        out.append(StoryboardScene(
            id=f"sb_{uuid.uuid4().hex[:8]}",
            index=i,
            text=s["text"],
            # Milliseconds, not centiseconds: rounding a boundary to 2dp moves it by up to
            # 5ms, and it rounds UP as readily as down — which lands it just inside the word
            # the clock had deliberately placed it in front of. Cosmetic precision is not
            # worth breaking the one invariant this whole file exists to hold.
            start_sec=round(start, 3),
            end_sec=round(end, 3),
            rationale=s["rationale"],
            media_type=s["media_type"],
            search_terms=s["search_terms"] or script.extract_keywords(s["text"], limit=8),
            visual_prompt=s.get("visual_prompt", ""),
            visual_ideas=s.get("visual_ideas") or [],
        ))
    return out


# ── the clock: a character offset in the narration -> a second on the timeline ───────
# One question, asked by everything downstream, answered two ways. Given measured word
# times it reports where the voice actually is. Without them it falls back to modelling how
# long the text takes to say. Routing both through the same interface is what stops the
# measured and estimated paths from drifting into different behaviour — the layout code
# above cannot tell which one it is holding.


class NarrationNotTimed(ValueError):
    """The narration has no measured word timings, so there is nothing honest to lay out."""


class _Clock:
    """Linear interpolation over ``(char_offset, second)`` marks, one per word."""

    def __init__(self, marks: List[Tuple[int, float]], total: float):
        self._offsets = [o for o, _ in marks]
        self._times = [t for _, t in marks]
        self._total = total

    def at(self, offset: int) -> float:
        i = bisect_right(self._offsets, offset) - 1
        if i < 0:
            return 0.0
        if i >= len(self._offsets) - 1:
            return self._times[-1]
        o0, o1 = self._offsets[i], self._offsets[i + 1]
        t0, t1 = self._times[i], self._times[i + 1]
        if o1 <= o0:
            return t0
        # Within a word we can only assume characters take even time; the marks either side
        # are real, so the error is bounded by one word either way.
        return round(t0 + (t1 - t0) * ((offset - o0) / (o1 - o0)), 3)


def _clock_for(
    full_text: str,
    total: float,
    word_times: Optional[List[Tuple[float, float]]],
) -> _Clock:
    starts = [m.start() for m in _WORD_RE.finditer(full_text)]
    if not word_times:
        raise NarrationNotTimed(
            "This storyboard has no measured word timings for the narration, so its "
            "placeholders cannot be placed on the voice."
        )
    if len(word_times) != len(starts):
        # A length mismatch means the timings were measured against different text. Using
        # them positionally would put every scene on the wrong word — quietly, and in a way
        # only visible after the user has filled the placeholders.
        raise NarrationNotTimed(
            f"The narration timings do not match the script ({len(word_times)} timed words "
            f"against {len(starts)} in the script)."
        )

    # Cut a hair BEFORE the word, not on it: whisper derives word starts from
    # cross-attention and they run slightly late, so cutting exactly on the timestamp put
    # the boundary inside the first phoneme and the picture changed a beat late.
    #
    # But only ever back into SILENCE. Word timings are frequently contiguous — the voice
    # runs "trafficking" straight into "with" with no gap at all — and there the lead has
    # nowhere to go, so taking it anyway moves the cut into the end of the previous word.
    # That is the same defect in the other direction, and it is the one you actually see:
    # the placeholder changes while the voice is still finishing the word before it.
    marks: List[Tuple[int, float]] = []
    previous_end = 0.0
    for offset, span in zip(starts, word_times):
        start, end = float(span[0]), float(span[1])
        mark = max(previous_end, start - _CUT_LEAD_SEC)
        if marks and mark < marks[-1][1]:
            mark = marks[-1][1]  # never go backwards
        marks.append((offset, min(mark, start)))
        previous_end = max(previous_end, end)
    # Sentinels so an offset anywhere in the text — including before the first word and
    # after the last — interpolates instead of falling off the end.
    marks = [(0, 0.0)] + [m for m in marks if m[0] > 0] + [(len(full_text), total)]
    return _Clock(marks, total)


# ── cutting a long scene into shots ──────────────────────────────────────────
# A sentence is the smallest place a visual can change without fighting the narration, so
# that is where the cuts go; a sentence too long to be one shot is cut at its clauses
# instead, which is what an editor does rather than sit on a dead frame. Splitting by
# estimated speech cost rather than by count keeps shots even when the sentences are not.
_SENTENCE_END_RE = re.compile(r'[.!?…]+["\'”’)\]]*\s+')
# A dash is a clause break whether or not it is spaced — "messages—mostly" is written tight
# far more often than not, and requiring the space meant a long sentence hinging on one had
# nowhere to cut and sat on screen past the ceiling. Comma/semicolon/colon still need the
# space so "1,000" is not mistaken for a break.
_CLAUSE_END_RE = re.compile(r'(?:[,;:]+\s+|\s*[—–]+\s*)')


# Periods that end a word without ending a sentence. Cutting the picture inside "Washington,
# D.C. pizzeria" puts a shot boundary in the middle of a noun phrase — measured right, edited
# wrong.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "rev", "sr", "jr", "st", "mt", "ft",
    "vs", "etc", "eg", "ie", "al", "approx", "est", "fig", "vol", "no",
    "inc", "ltd", "co", "corp", "dept", "gov", "sen", "rep", "gen", "capt", "sgt", "lt", "col",
}
_TRAILING_WORD_RE = re.compile(r"([A-Za-z]+)$")


def _is_sentence_end(text: str, match: "re.Match") -> bool:
    """Whether a full stop really closes a sentence."""
    # A lowercase word after it means the sentence carried on ("D.C. pizzeria", "vs. the").
    following = text[match.end():match.end() + 1]
    if following and following.islower():
        return False
    word = _TRAILING_WORD_RE.search(text[:match.start()])
    if not word:
        return True
    token = word.group(1)
    # A single letter before the stop is an initial or part of an acronym, never a sentence.
    return len(token) > 1 and token.lower() not in _ABBREVIATIONS


_INITIAL_AHEAD_RE = re.compile(r"^[A-Z]\.|^[A-Z]$")


def _is_clause_break(text: str, match: "re.Match") -> bool:
    """Whether a comma really separates clauses.

    "Washington, D.C." is one place, not two, and a shot that starts on "D.C. pizzeria"
    reads as a machine that cannot hear. Anything followed by an initial stays joined.
    """
    return not _INITIAL_AHEAD_RE.match(text[match.end():match.end() + 2])


def _split_keeping_text(text: str, pattern: "re.Pattern", accept=None) -> List[str]:
    """Split on ``pattern`` WITHOUT discarding it. ``"".join(result) == text`` exactly —
    re.split() would swallow the separator, and a separator here is a closing quote or a
    paragraph break: punctuation the reader sees and silence the layout has to charge for.

    ``accept(text, match)`` can reject a candidate boundary, which is how a full stop that
    is only an abbreviation stays out of the cut list.
    """
    out: List[str] = []
    last = 0
    for m in pattern.finditer(text):
        if accept and not accept(text, m):
            continue
        out.append(text[last:m.end()])
        last = m.end()
    if last < len(text):
        out.append(text[last:])
    return [p for p in out if p.strip()]


def _unit_offsets(full_text: str, start_ch: int, end_ch: int, clock: _Clock) -> List[int]:
    """Character offsets inside ``[start_ch, end_ch)`` where a cut could land: sentence
    ends, plus the clauses of any sentence that by itself would hold the screen past the
    ceiling. Cutting inside a long sentence is what an editor does rather than sit on a
    dead frame — without it, a scene of one sprawling sentence and three short ones could
    never be brought under the limit."""
    text = full_text[start_ch:end_ch]
    pieces = _split_keeping_text(text, _SENTENCE_END_RE, _is_sentence_end)
    if len(pieces) < 2:
        pieces = _split_keeping_text(text, _CLAUSE_END_RE, _is_clause_break)

    offsets: List[int] = []
    cursor = start_ch
    for piece in pieces:
        # How long this piece really holds the screen, straight off the measured clock.
        if clock.at(cursor + len(piece)) - clock.at(cursor) > _MAX_SHOT_SEC:
            clauses = _split_keeping_text(piece, _CLAUSE_END_RE, _is_clause_break)
            if len(clauses) > 1:
                for clause in clauses:
                    offsets.append(cursor)
                    cursor += len(clause)
                continue
        offsets.append(cursor)
        cursor += len(piece)
    return offsets


def _shots(full_text: str, start_ch: int, end_ch: int, clock: _Clock) -> List[Tuple[int, int]]:
    """``[(start_char, end_char)]`` for one scene — itself if it is already shot-length."""
    start, end = clock.at(start_ch), clock.at(end_ch)
    span = end - start
    if span <= _MAX_SHOT_SEC:
        return [(start_ch, end_ch)]

    offsets = _unit_offsets(full_text, start_ch, end_ch, clock)
    if len(offsets) < 2:
        return [(start_ch, end_ch)]  # nothing to cut on without fighting the narration
    offsets.append(end_ch)

    # Close a shot once it has run about a target's worth, or as soon as one more unit
    # would push it past the maximum. That enforces the ceiling directly rather than
    # hoping an even split lands under it.
    cuts: List[int] = [offsets[0]]
    opened = start
    for i in range(1, len(offsets) - 1):
        here, nxt = clock.at(offsets[i]), clock.at(offsets[i + 1])
        if (here - opened) >= _TARGET_SHOT_SEC or (nxt - opened) > _MAX_SHOT_SEC:
            cuts.append(offsets[i])
            opened = here
    cuts.append(end_ch)

    # A final sliver folds back into the shot before it rather than flashing on screen —
    # unless that would push the merged shot back over the ceiling we just enforced.
    if len(cuts) > 2:
        last_start, last_end = clock.at(cuts[-2]), clock.at(cuts[-1])
        if (last_end - last_start) < _MIN_SCENE_SEC and (last_end - clock.at(cuts[-3])) <= _MAX_SHOT_SEC:
            del cuts[-2]

    return [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]


# ── anchoring the model's scenes back into the narration ─────────────────────
# The prompt asks for verbatim, gapless spans, but models paraphrase, drop clauses and
# renormalize punctuation often enough that costing their text directly is what put
# placeholders out of step with the voice. So the model is trusted for WHERE a scene
# begins and for its visual metadata — never for the words themselves. Each scene's
# opening words are located in the real narration and the text between one scene's anchor
# and the next IS that scene. Every character then belongs to exactly one scene: no gaps,
# no overlaps, nothing counted twice, and the placeholder shows the words spoken under it.

_ANCHOR_WORDS = 6  # long enough to be unique in a narration, short enough to survive a paraphrase


def _anchor_spans(full_text: str, scenes: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], int, int]]:
    """``(scene, start_char, end_char)`` for each scene — in order, contiguous, and
    together covering the whole narration exactly once.

    A scene whose opening words appear nowhere in the narration is DROPPED rather than
    given a zero-length slot: the model invented it, and a placeholder sitting over no
    narration at all is worse than one fewer visual change.
    """
    words = [(_normalize(m.group(0)), m.start()) for m in _WORD_RE.finditer(full_text)]
    lowered = [w for w, _ in words]

    # Character offset where each scene starts. The search only ever moves forward, so a
    # phrase that recurs later in the narration cannot pull a scene backwards.
    placed: List[tuple] = []
    cursor = 0
    for i, s in enumerate(scenes):
        if i == 0:
            placed.append((0, s))  # the first scene always covers the start of the narration
            continue
        needle = [_normalize(m.group(0)) for m in _WORD_RE.finditer(s["text"])][:_ANCHOR_WORDS]
        at = _find_words(lowered, needle, cursor) if needle else -1
        if at < 0:
            continue
        cursor = at + 1
        placed.append((words[at][1], s))

    # Two scenes anchored to the same word means the earlier one covers nothing. Keep the
    # later (its metadata is what the model attached to those words), then pull the first
    # survivor back to the top so the narration's opening is never left uncovered.
    deduped: List[tuple] = []
    for start, s in placed:
        if deduped and deduped[-1][0] == start:
            deduped[-1] = (start, s)
        else:
            deduped.append((start, s))
    if deduped:
        deduped[0] = (0, deduped[0][1])

    # Character offsets, not text: the whitespace at a span's tail is where the paragraph
    # break lives and the voice really does stop there, so the clock has to see it. Handing
    # back a trimmed string would charge that pause to nobody.
    return [
        (s, start, (deduped[i + 1][0] if i + 1 < len(deduped) else len(full_text)))
        for i, (start, s) in enumerate(deduped)
    ]


def _find_words(hay: List[str], needle: List[str], start: int) -> int:
    """Index in ``hay`` at or after ``start`` where ``needle`` begins. -1 if nothing matches.

    Every window of the needle is tried, longest first, so the anchor survives junk at
    EITHER end: a label the model prefixed ("Scene 3: In the age of…") as readily as a
    paraphrased tail. Only shrinking from the tail — which is what this did first — meant
    one stray prefix made every probe start on a word that is nowhere in the narration,
    and the whole storyboard collapsed onto a single placeholder.
    """
    for size in range(len(needle), 1, -1):
        for offset in range(0, len(needle) - size + 1):
            probe = needle[offset:offset + size]
            for i in range(start, len(hay) - size + 1):
                if hay[i:i + size] == probe:
                    return i
    return -1


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
