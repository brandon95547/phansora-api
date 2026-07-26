"""AI Storyboard — a documentary editor's first visual pass over the narration.

Given the narration text and its real rendered duration on the timeline, an LLM
decides where the VISUAL should change based on the flow of the story — a new idea,
person, place, event, time period, or visual concept — NOT fixed intervals or every
sentence. Each returned scene carries media suggestions (search terms + visual ideas)
that the editor's sidebar turns into real fair-use results on demand.

Timing is computed here, deterministically: the model never guesses seconds. Each
scene is anchored back into the real narration text, its span costed as speech time
plus the pauses that span actually contains, and those costs scaled onto
``total_duration_sec`` so the first scene starts at 0 and the last ends exactly at
the narration's end. That keeps the placeholders aligned with the voice under them.
"""
from __future__ import annotations

import logging
import re
import uuid
from bisect import bisect_right
from typing import Any, Dict, List, Optional, Tuple

from ..models import StoryboardScene
from . import llm, script

logger = logging.getLogger("narrava-studio.storyboard")

# Below this, a scene is too short to be a distinct shot — merge it into the previous
# one so the editor isn't handed a strip of one-second placeholders.
_MIN_SCENE_SEC = 1.5

# How often a finished documentary actually changes picture. A first pass that leaves one
# image up for the whole narration is a slide, not a storyboard: real cutting is far denser
# than a writer expects, and B-roll under narration turns over every few seconds. The model
# is asked to aim for _TARGET_SHOT_SEC, and any scene that still outruns _MAX_SHOT_SEC is
# split at its sentence boundaries afterwards so the rhythm does not depend on it complying.
_TARGET_SHOT_SEC = 5.0
_MAX_SHOT_SEC = 10.0

_SYSTEM = (
    "You are an experienced documentary editor planning the VISUALS for a narration. "
    "You are given the full narration script. Break it into an ordered list of scenes, "
    "where each scene is a stretch of narration that a single shot or image would cover. "
    "Change the visual wherever the narration gives you a reason to: a new idea, person, "
    "place, event, time period, or visual concept. "
    "PACING MATTERS AS MUCH AS MEANING. A finished documentary holds a shot for a few "
    "seconds and then moves; one image left up for thirty seconds reads as a dead frame and "
    "is the single most common mistake in a first pass. Cut on meaning rather than on a "
    "stopwatch, but keep actively looking for the next reason to cut — if a stretch of "
    "narration is running long under one visual, find the beat inside it where the picture "
    "could change. "
    "The first scene must cover the very beginning of the narration. "
    "For each scene, give the exact narration text it covers (verbatim, in order, with no "
    "gaps or overlaps so the pieces concatenate back to the original, and with NO scene "
    "number, label or prefix in front of it), a one-line rationale "
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


def build_storyboard(
    full_text: str,
    total_duration_sec: float,
    max_scenes: int = 24,
    word_times: Optional[List[Tuple[float, float]]] = None,
) -> List[StoryboardScene]:
    """``word_times`` is one (start, end) per word of ``full_text``, measured from the
    rendered audio (see services/align.py). Given it, every boundary sits on a real word;
    without it the layout estimates, which is close but not frame-accurate."""
    text = (full_text or "").strip()
    total = max(0.1, float(total_duration_sec or 0.0))
    if not text:
        return []

    raw_scenes = _ask_llm(text, max_scenes, total)
    if not raw_scenes:
        raw_scenes = [_fallback_scene(text)]

    return _lay_out(raw_scenes, total, text, word_times)


def _ask_llm(text: str, max_scenes: int, total: float) -> List[Dict[str, Any]]:
    # The model cannot pace shots without knowing how long the narration runs — asked
    # blind it returns a handful of chapter-sized scenes whatever the length. Give it the
    # duration and the shot count that implies.
    target = max(1, min(max_scenes, round(total / _TARGET_SHOT_SEC)))
    user = (
        f"This narration runs about {total:.0f} seconds. Plan roughly {target} scenes "
        f"(never more than {max_scenes}) — that is a visual change about every "
        f"{_TARGET_SHOT_SEC:.0f} seconds, which is the pace a documentary cuts at. "
        f"Returning only a few long scenes for a {total:.0f}-second narration is wrong.\n\n"
        f"NARRATION:\n{text}"
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


def _lay_out(
    scenes: List[Dict[str, Any]],
    total: float,
    full_text: str,
    word_times: Optional[List[Tuple[float, float]]] = None,
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

    # Split anything still longer than a shot. Same visual subject, so the search terms and
    # visual ideas carry over — the editor just gets a slot per shot instead of one image
    # asked to hold the screen for half a minute, which is how a real cut list is built.
    # Deterministic, so the pacing holds even when the model returns three scenes for a
    # four-minute narration.
    shots: List[Dict[str, Any]] = []
    shot_bounds: List[List[float]] = []
    for s, (start_ch, end_ch) in zip(merged, merged_spans):
        for n, (a_ch, b_ch) in enumerate(_shots(full_text, start_ch, end_ch, clock)):
            # The anchored span is what is actually spoken here; the model's own `text` is
            # only a fallback for a scene we could not place in the narration.
            text = full_text[a_ch:b_ch].strip() or s["text"]
            shots.append({**s, "text": text} if n == 0 else {
                **s, "text": text,
                "rationale": "Same subject as the previous shot — vary the angle or framing "
                             "so the picture keeps moving.",
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
            start_sec=round(start, 2),
            end_sec=round(end, 2),
            rationale=s["rationale"],
            media_type=s["media_type"],
            search_terms=s["search_terms"] or script.extract_keywords(s["text"], limit=8),
            visual_ideas=s["visual_ideas"],
        ))
    return out


# ── the clock: a character offset in the narration -> a second on the timeline ───────
# One question, asked by everything downstream, answered two ways. Given measured word
# times it reports where the voice actually is. Without them it falls back to modelling how
# long the text takes to say. Routing both through the same interface is what stops the
# measured and estimated paths from drifting into different behaviour — the layout code
# above cannot tell which one it is holding.


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

    if word_times and len(word_times) == len(starts):
        marks = [(o, float(t[0])) for o, t in zip(starts, word_times)]
    else:
        if word_times:
            # A length mismatch means the timings were aligned against different text.
            # Using them positionally would put every scene on the wrong word.
            logger.warning(
                "Narrava storyboard: %d word timings for %d narration words — ignoring them "
                "and estimating instead", len(word_times), len(starts),
            )
        marks = _estimated_marks(full_text, starts, total)

    # Sentinels so an offset anywhere in the text — including before the first word and
    # after the last — interpolates instead of falling off the end.
    marks = [(0, 0.0)] + [m for m in marks if m[0] > 0] + [(len(full_text), total)]
    return _Clock(marks, total)


def _estimated_marks(full_text: str, starts: List[int], total: float) -> List[Tuple[int, float]]:
    """Where each word falls if the voice speaks at the modelled rate and pauses."""
    marks: List[Tuple[int, float]] = []
    running = 0.0
    previous = 0
    for start in starts:
        running += _speech_seconds(full_text[previous:start])
        marks.append((start, running))
        previous = start
    running += _speech_seconds(full_text[previous:])
    scale = (total / running) if running > 0 else 0.0
    return [(o, t * scale) for o, t in marks]


# ── cutting a long scene into shots ──────────────────────────────────────────
# A sentence is the smallest place a visual can change without fighting the narration, so
# that is where the cuts go; a sentence too long to be one shot is cut at its clauses
# instead, which is what an editor does rather than sit on a dead frame. Splitting by
# estimated speech cost rather than by count keeps shots even when the sentences are not.
_SENTENCE_END_RE = re.compile(r'[.!?…]+["\'”’)\]]*\s+')
_CLAUSE_END_RE = re.compile(r'[,;:—–]+\s+')


def _split_keeping_text(text: str, pattern: "re.Pattern") -> List[str]:
    """Split on ``pattern`` WITHOUT discarding it. ``"".join(result) == text`` exactly —
    re.split() would swallow the separator, and a separator here is a closing quote or a
    paragraph break: punctuation the reader sees and silence the layout has to charge for.
    """
    out: List[str] = []
    last = 0
    for m in pattern.finditer(text):
        out.append(text[last:m.end()])
        last = m.end()
    if last < len(text):
        out.append(text[last:])
    return [p for p in out if p.strip()]


def _units(text: str, span: float) -> List[str]:
    """The pieces a scene may be cut into: sentences, and clauses where a sentence is by
    itself long enough to hold the screen past the ceiling. Cutting inside a long sentence
    is what an editor does rather than sit on a dead frame — and without this a scene made
    of one sprawling sentence and three short ones could never be split under the limit."""
    units = _split_keeping_text(text, _SENTENCE_END_RE)
    if len(units) < 2:
        return _split_keeping_text(text, _CLAUSE_END_RE)

    budget = sum(_speech_seconds(u) for u in units) or 1.0
    out: List[str] = []
    for unit in units:
        if _speech_seconds(unit) / budget * span > _MAX_SHOT_SEC:
            pieces = _split_keeping_text(unit, _CLAUSE_END_RE)
            out.extend(pieces if len(pieces) > 1 else [unit])
        else:
            out.append(unit)
    return out


def _shots(full_text: str, start_ch: int, end_ch: int, clock: _Clock) -> List[Tuple[int, int]]:
    """``[(start_char, end_char)]`` for one scene — itself if it is already shot-length."""
    start, end = clock.at(start_ch), clock.at(end_ch)
    span = end - start
    if span <= _MAX_SHOT_SEC:
        return [(start_ch, end_ch)]

    text = full_text[start_ch:end_ch]
    units = _units(text, span)
    if len(units) < 2:
        return [(start_ch, end_ch)]  # nothing to cut on without fighting the narration

    # Units concatenate back to the text exactly, so their lengths give the character
    # offset each one begins at — and the clock turns that into the second the voice
    # reaches it. The grouping below is therefore driven by real screen time.
    offsets: List[int] = []
    cursor = start_ch
    for unit in units:
        offsets.append(cursor)
        cursor += len(unit)
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

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_ANCHOR_WORDS = 6  # long enough to be unique in a narration, short enough to survive a paraphrase


def _anchor_spans(full_text: str, scenes: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], int, int]]:
    """``(scene, start_char, end_char)`` for each scene — in order, contiguous, and
    together covering the whole narration exactly once.

    A scene whose opening words appear nowhere in the narration is DROPPED rather than
    given a zero-length slot: the model invented it, and a placeholder sitting over no
    narration at all is worse than one fewer visual change.
    """
    words = [(m.group(0).lower(), m.start()) for m in _WORD_RE.finditer(full_text)]
    lowered = [w for w, _ in words]

    # Character offset where each scene starts. The search only ever moves forward, so a
    # phrase that recurs later in the narration cannot pull a scene backwards.
    placed: List[tuple] = []
    cursor = 0
    for i, s in enumerate(scenes):
        if i == 0:
            placed.append((0, s))  # the first scene always covers the start of the narration
            continue
        needle = [m.group(0).lower() for m in _WORD_RE.finditer(s["text"])][:_ANCHOR_WORDS]
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


# ── how long a span takes to say ─────────────────────────────────────────────
# Word share was the other half of the drift: it charges nothing for the silence the voice
# puts at a full stop or a paragraph break, so that silence got smeared evenly across every
# word in the narration and pushed each boundary progressively later than the audio. Cost a
# span as speech time PLUS the pauses it actually contains and the silence stays in the
# scene that owns it. Characters rather than words, too — "a" and "extraordinarily" do not
# take the same time to say.
_CHARS_PER_SEC = 15.0     # ≈150 wpm of ordinary English prose
_PAUSE_SENTENCE = 0.40    # . ! ? …
_PAUSE_CLAUSE = 0.18      # , ; : — –
_PAUSE_PARAGRAPH = 0.75   # on top of the sentence stop that usually precedes it

_SENTENCE_RE = re.compile(r"[.!?…]+")
_CLAUSE_RE = re.compile(r"[,;:—–]")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")
_WHITESPACE_RE = re.compile(r"\s+")


def _speech_seconds(text: str) -> float:
    """Rough seconds to speak ``text``. The absolute number does not matter — every span is
    scaled onto the narration's real measured duration — only that spans are costed
    relative to each other the way the voice actually renders them."""
    if not text or not text.strip():
        return 0.0
    return (
        len(_WHITESPACE_RE.sub(" ", text.strip())) / _CHARS_PER_SEC
        + len(_SENTENCE_RE.findall(text)) * _PAUSE_SENTENCE
        + len(_CLAUSE_RE.findall(text)) * _PAUSE_CLAUSE
        + len(_PARAGRAPH_RE.findall(text)) * _PAUSE_PARAGRAPH
    )


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
