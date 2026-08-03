"""Script generation and segmentation.

Two entry points:
  - ``generate_script`` — prompt -> narrator-formatted script (via the LLM).
  - ``segment_script`` — any script text -> timed beats with per-beat keywords.

Segmentation is pure/deterministic (no LLM): it splits the narration into
sentence-level beats and estimates each beat's start/end from its word count at a
words-per-minute pace. Those timings are what let the timeline place each media
clip at the moment the narration talks about it.
"""
from __future__ import annotations

import re
import uuid
from typing import List

from .. import config
from ..models import Script, ScriptSegment
from . import llm, styles

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
# Split on sentence-ending punctuation followed by whitespace. Good enough for
# narration prose; abbreviations are rare in scripts and a stray split is harmless.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Short, common words that never make useful media-search terms.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "as", "at", "by", "from", "into", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "these", "those", "their", "there", "here",
    "we", "you", "they", "he", "she", "his", "her", "our", "your", "them", "us",
    "not", "no", "so", "if", "then", "than", "when", "while", "which", "who",
    "what", "how", "why", "will", "would", "can", "could", "may", "might", "have",
    "has", "had", "do", "does", "did", "about", "over", "under", "up", "down",
    "out", "very", "just", "more", "most", "some", "any", "all", "one", "also",
}


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _estimate_seconds(text: str, wpm: int) -> float:
    words = _word_count(text)
    if words == 0:
        return 0.0
    return round(words / max(1, wpm) * 60.0, 2)


def extract_keywords(text: str, limit: int = 5) -> List[str]:
    """Heuristic keywords for media search: proper nouns first, then salient words."""
    tokens = _WORD_RE.findall(text or "")
    proper: List[str] = []
    other: List[str] = []
    seen = set()
    for i, tok in enumerate(tokens):
        low = tok.lower()
        if low in _STOPWORDS or len(tok) < 3:
            continue
        if low in seen:
            continue
        seen.add(low)
        # Capitalised mid-sentence -> likely a proper noun / named entity.
        if tok[0].isupper() and i > 0:
            proper.append(tok)
        else:
            other.append(tok)
    ordered = proper + other
    return ordered[:limit]


def segment_script(
    full_text: str,
    *,
    title: str = "",
    wpm: int | None = None,
    source: str = "provided",
) -> Script:
    """Split narration text into timed beats with keywords."""
    settings = config.get_settings()
    wpm = wpm or settings.narrava_words_per_minute
    text = (full_text or "").strip()

    raw_sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]

    # Merge very short fragments into the previous sentence so a beat is long
    # enough to warrant its own clip (avoids one-word beats like "Right.").
    beats: List[str] = []
    for sentence in raw_sentences:
        if beats and _word_count(sentence) < 4:
            beats[-1] = f"{beats[-1]} {sentence}"
        else:
            beats.append(sentence)

    segments: List[ScriptSegment] = []
    cursor = 0.0
    for index, beat in enumerate(beats):
        dur = _estimate_seconds(beat, wpm)
        segments.append(
            ScriptSegment(
                id=f"seg_{uuid.uuid4().hex[:8]}",
                index=index,
                text=beat,
                start_sec=round(cursor, 2),
                end_sec=round(cursor + dur, 2),
                keywords=extract_keywords(beat),
            )
        )
        cursor += dur

    if not title:
        title = _derive_title(text)

    return Script(
        title=title,
        full_text=text,
        segments=segments,
        estimated_duration_sec=round(cursor, 2),
        source="prompt" if source == "prompt" else "provided",
    )


def _derive_title(text: str) -> str:
    first = _SENTENCE_RE.split(text.strip(), maxsplit=1)[0] if text.strip() else "Untitled Video"
    words = first.split()
    return " ".join(words[:8]) or "Untitled Video"


_SCRIPT_SYSTEM = (
    "You are a professional video narration writer. Given a topic, write a spoken "
    "narration script the way a seasoned documentary narrator would deliver it: "
    "clear, engaging, in flowing prose. Rules:\n"
    "- Output ONLY the words to be spoken. No scene directions, no camera notes, "
    "no speaker labels, no markdown, no headings, no bracketed cues.\n"
    "- Write in short-to-medium sentences that read naturally aloud.\n"
    "- Open with a hook and give the piece a clear beginning, middle and end.\n"
    "- Do not address 'the viewer' with meta commentary about the video itself."
)

# Appended to the system prompt ONLY when research is attached, so the plain
# brief-only path stays byte-identical to what it always sent.
_RESEARCH_SYSTEM = (
    "\n\nA RESEARCH KNOWLEDGE BASE is provided with this request. It is the factual "
    "record of the story:\n"
    "- Base every factual statement in the narration on the research. Do not invent "
    "names, dates, numbers, or events that are not in it.\n"
    "- The brief describes HOW to tell the story (tone, angle, pacing, length); the "
    "research describes WHAT the story contains.\n"
    "- Keep allegations attributed the way the research does (e.g. 'Police allege'); "
    "never present an allegation as settled fact.\n"
    "- You do not have to use every fact — choose what serves the narration."
)

# Prompt-size ceiling for the rendered research block. Big dossiers can carry far
# more research than a narration needs; the renderer emits the most story-critical
# sections first (summary, people, timeline, findings) so a cut loses the least.
_RESEARCH_MAX_CHARS = 18000


def _render_research_block(research: dict | None) -> str:
    """Dossier Nova's research dataset -> a readable knowledge-base block.

    Tolerant of missing keys and unexpected shapes: the dataset is produced by a
    different service and may evolve; anything unrecognized is simply skipped.
    Returns "" when there is nothing factual to show.
    """
    if not isinstance(research, dict):
        return ""

    def clean(value) -> str:
        return str(value).strip() if value is not None else ""

    def str_list(items) -> list[str]:
        if not isinstance(items, list):
            return []
        return [s for s in (clean(x) for x in items) if s]

    out: list[str] = []

    subject = clean(research.get("subject"))
    if subject:
        out.append(f"Subject: {subject}")

    es = research.get("executive_summary") or {}
    if isinstance(es, dict):
        if clean(es.get("overview")):
            out.append(f"Overview: {clean(es.get('overview'))}")
        if clean(es.get("current_status")):
            out.append(f"Current status: {clean(es.get('current_status'))}")
        if clean(es.get("evidence_confidence")):
            out.append(f"Overall evidence confidence: {clean(es.get('evidence_confidence'))}")
        key_findings = str_list(es.get("key_findings"))
        if key_findings:
            out.append("Key findings:")
            out.extend(f"- {x}" for x in key_findings)
        unknowns = str_list(es.get("major_unknowns"))
        if unknowns:
            out.append("Major unknowns (do not present these as settled):")
            out.extend(f"- {x}" for x in unknowns)

    entities = research.get("entities") or {}
    if isinstance(entities, dict):
        people = [p for p in (entities.get("people") or []) if isinstance(p, dict)]
        if people:
            out.append("People:")
            for p in people:
                line = clean(p.get("name"))
                if not line:
                    continue
                if clean(p.get("role")):
                    line += f" ({clean(p.get('role'))})"
                if clean(p.get("description")):
                    line += f": {clean(p.get('description'))}"
                out.append(f"- {line}")
        orgs = [o for o in (entities.get("organizations") or []) if isinstance(o, dict)]
        if orgs:
            out.append("Organizations:")
            for o in orgs:
                line = clean(o.get("name"))
                if not line:
                    continue
                if clean(o.get("role")):
                    line += f": {clean(o.get('role'))}"
                out.append(f"- {line}")
        locations = [l for l in (entities.get("locations") or []) if isinstance(l, dict)]
        if locations:
            out.append("Locations:")
            for l in locations:
                line = clean(l.get("name"))
                if not line:
                    continue
                if clean(l.get("relevance")):
                    line += f": {clean(l.get('relevance'))}"
                out.append(f"- {line}")

    timeline = [t for t in (research.get("timeline") or []) if isinstance(t, dict)]
    if timeline:
        out.append("Timeline:")
        for t in timeline:
            when = " ".join(x for x in (clean(t.get("date")), clean(t.get("time"))) if x) or "Undated"
            event = clean(t.get("event"))
            if not event:
                continue
            line = f"- {when}: {event}"
            if clean(t.get("discrepancy")):
                line += f" [sources disagree: {clean(t.get('discrepancy'))}]"
            out.append(line)

    findings = [f for f in (research.get("findings") or []) if isinstance(f, dict)]
    if findings:
        out.append("Facts and allegations:")
        for f in findings:
            statement = clean(f.get("statement"))
            if not statement:
                continue
            kind = "ALLEGATION" if clean(f.get("type")).lower() == "allegation" else "FACT"
            tags = [kind]
            if clean(f.get("confidence")):
                tags.append(f"confidence: {clean(f.get('confidence'))}")
            line = f"- [{', '.join(tags)}]"
            if kind == "ALLEGATION" and clean(f.get("attribution")):
                line += f" {clean(f.get('attribution'))}:"
            out.append(f"{line} {statement}")

    cs = research.get("cross_source") or {}
    if isinstance(cs, dict):
        conflicting = [c for c in (cs.get("conflicting") or []) if isinstance(c, dict)]
        if conflicting:
            out.append("Conflicting reporting (acknowledge, do not pick a side):")
            for c in conflicting:
                versions = "; ".join(
                    f"{clean(v.get('source'))}: {clean(v.get('claim'))}"
                    for v in (c.get("versions") or [])
                    if isinstance(v, dict) and clean(v.get("claim"))
                )
                topic = clean(c.get("topic"))
                out.append(f"- {topic} -- {versions}" if topic else f"- {versions}")
        unresolved = str_list(cs.get("unresolved"))
        if unresolved:
            out.append("Unresolved questions:")
            out.extend(f"- {x}" for x in unresolved)

    src_lines = []
    for s in research.get("sources") or []:
        if not isinstance(s, dict):
            continue
        label = clean(s.get("label"))
        if not label:
            continue
        line = f"- {label}"
        if clean(s.get("central_argument")):
            line += f": {clean(s.get('central_argument'))}"
        src_lines.append(line)
    if src_lines:
        out.append("Sources the research was built from:")
        out.extend(src_lines)

    if not out:
        return ""
    block = "RESEARCH KNOWLEDGE BASE:\n" + "\n".join(out)
    if len(block) > _RESEARCH_MAX_CHARS:
        block = block[:_RESEARCH_MAX_CHARS].rsplit("\n", 1)[0] + "\n[research truncated]"
    return block


def generate_script(
    prompt: str,
    *,
    style: str = "documentary",
    tone: str | None = None,
    target_duration_sec: int | None = None,
    doc_style: str | None = None,
    wpm: int | None = None,
    research: dict | None = None,
) -> Script:
    """Prompt -> narrator-formatted script, then segmented into timed beats.

    Two different "styles", which is confusing enough to be worth spelling out:

      ``style``      what KIND of piece this is — documentary, explainer, story,
                     promotional, educational.
      ``doc_style``  the house style the whole film is made in (services/styles.py):
                     cinematic, investigative, historical, and so on. The storyboard is
                     built from the same value, which is what keeps the words and the
                     pictures reading as one film. Unknown or missing writes unstyled.

    ``research`` is Dossier Nova's structured research dataset. When present, the
    brief becomes creative direction only and every fact comes from the research;
    when absent, generation works from the brief alone exactly as it always has.
    """
    settings = config.get_settings()
    wpm = wpm or settings.narrava_words_per_minute

    research_block = _render_research_block(research)

    instructions = [f"Topic / brief: {prompt.strip()}", f"Narration style: {style}."]
    if tone:
        instructions.append(f"Tone: {tone}.")
    if target_duration_sec:
        target_words = int(target_duration_sec / 60.0 * wpm)
        instructions.append(
            f"Target length: about {target_duration_sec} seconds of narration "
            f"(~{target_words} words). Stay close to this length."
        )
    if research_block:
        instructions.append(
            "The brief above is creative direction only; base every factual statement "
            "on the research knowledge base below."
        )
    # The house style leads the message: it is the frame the brief is written inside, and
    # a model reads the top of a user turn as the standing instruction for the rest of it.
    user = styles.narration_block(doc_style) + "\n".join(instructions)
    if research_block:
        user += "\n\n" + research_block

    system = _SCRIPT_SYSTEM + (_RESEARCH_SYSTEM if research_block else "")
    body = llm.generate_text(system, user, max_output_tokens=2500).strip()
    body = _strip_artifacts(body)

    return segment_script(body, wpm=wpm, source="prompt")


# ── Enhance ──────────────────────────────────────────────────────────────────
# The Write tab's "Enhance Narration". Everything here is written to protect one thing:
# the piece the user already wrote. The model may reword freely, and may not add to,
# remove from, or reorder what was said — a "better" narration that argues something
# slightly different is a worse answer than leaving the text alone.
_ENHANCE_SYSTEM = """
You are an editor polishing narration that will be read aloud in a video.

The words are the writer's. Return the SAME narration, written better.

You may:
- Fix grammar, spelling, punctuation and tense.
- Reword a sentence when different wording says the same thing more clearly, or reads
  better aloud.
- Split a sentence that is hard to follow, or join two that are stronger together.
- Replace a vague or repeated word with a more precise one.

You must not:
- Add any fact, name, number, claim, opinion or example that is not already there.
- Remove anything the writer said, or soften a point they made.
- Change the order in which the ideas are presented.
- Change the language, the person (I / we / you), or the tense the piece is written in.
- Change the register: keep casual writing casual and formal writing formal.
- Add a title, headings, labels, stage directions or markdown.
- Add or remove paragraph breaks.

If a passage is already good, leave it exactly as it is. Returning the text unchanged is a
valid answer.

Output the narration and nothing else: no preamble, no explanation, no quotation marks
around it, no notes about what you changed.
""".strip()

# Below this share of the original word count, the reply is not a polish — it is a summary,
# or an answer the model ran out of room to finish. Either way it is not what was asked for,
# and handing it back would quietly replace the writer's script with a shorter one.
_ENHANCE_MIN_RATIO = 0.6


def enhance_narration(text: str) -> str:
    """Narration the user wrote -> the same narration, written better.

    Pinned to DeepSeek rather than following the provider switch (see llm.deepseek_text).

    The output budget is sized from the input instead of fixed. The reply is the whole
    piece rewritten, so one ceiling for every length would truncate a long script and give
    back a narration missing its ending — which is exactly the failure the ratio check
    below refuses to pass on.
    """
    src = (text or "").strip()
    if not src:
        return ""

    words = _word_count(src)
    budget = max(600, min(8000, int(words * 2.2) + 300))
    out = _strip_artifacts(llm.deepseek_text(_ENHANCE_SYSTEM, src, max_output_tokens=budget))

    if not out:
        raise RuntimeError("The model returned nothing to use.")
    if _word_count(out) < words * _ENHANCE_MIN_RATIO:
        raise RuntimeError(
            "The model shortened the narration instead of polishing it, so it was discarded."
        )
    return out


def _strip_artifacts(text: str) -> str:
    """Remove markdown/label artifacts an LLM sometimes adds despite instructions."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        # Drop a leading "Narrator:" / "Title:" style label.
        stripped = re.sub(r"^(narrator|voiceover|vo|title)\s*:\s*", "", stripped, flags=re.I)
        # Drop surrounding markdown emphasis / heading markers.
        stripped = re.sub(r"^#{1,6}\s*", "", stripped)
        stripped = stripped.strip("*_` ")
        lines.append(stripped)
    return "\n".join(lines).strip()
