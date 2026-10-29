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

import logging
import re
import uuid
from typing import List, Optional

from .. import config
from ..models import Script, ScriptSegment
from . import enhance_styles, llm, neutrality, styles

logger = logging.getLogger("narrava-studio.script")

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

# Always part of the writer's instructions. The narration is the person's, about the subject
# as THEY frame it: asked for a documentary on a disputed subject, the writer used to open on
# its aftermath and call it a conspiracy theory in its own voice, which settles the question
# for them and leaves the people who see it otherwise no way to tell it. Checked, too, not
# only asked for (generate_script, services/neutrality.py).
_NEUTRAL_SYSTEM = (
    "\n\nStay neutral about real people and events. The narrator reports; it does not rule:\n"
    "- Never pass a verdict on the subject in the narrator's own voice. Never call it, or "
    "anything in it, a rumor, hoax, myth, conspiracy theory, misinformation, propaganda, "
    "debunked, baseless, fake, true or false, unless the brief does. Who called it what is "
    "part of the story: say who said it.\n"
    "- On a disputed subject, the first lines are the claim itself: what is alleged, against "
    "whom, and by whom. Not an event that came after it (an attack, an arrest, a ruling), "
    "and not a verdict. This holds whatever opening the house style asks for: a cold open "
    "opens cold on the claim.\n"
    "- Set out the claims and theories in full, in the terms of the people who hold them: "
    "say what is alleged and whom it names. Then the answers of the people who reject them. "
    "Each attributed, and each side given its strongest case.\n"
    "- State as fact only what is on the record: what happened and when, what was said and "
    "by whom. What police, courts or officials found or ruled is theirs: 'police said they "
    "found no evidence', never 'there was no evidence' in the narrator's voice. An accusation "
    "against a real person is told as an accusation, never as a fact, and never waved away "
    "either.\n"
    "- No closing line that sums up who was right, or that sets the facts against the "
    "claims.\n"
    "- Follow the angle in the brief. Asked to tell it as the people who hold a view see it, "
    "tell it from inside that view, attributed to them. Asked for a point of view, argue "
    "theirs. Otherwise leave the question open for the viewer to decide."
)

# Sent back with a narration that is less than neutral, for one more draft: what is wrong
# with it (_faults_note), then the draft.
_NEUTRAL_REWRITE = (
    "Your draft is below. It is not neutral.\n{faults}\n\nWrite the narration again with the "
    "same subject, shape and length, changing only what you must. Output only the "
    "narration.\n\nYour draft:\n{draft}"
)

# Fiction is the one kind the check stays out of: a story's narrator may call a character's
# word a lie. The instructions above still hold for a story about real people.
_UNCHECKED_KINDS = {"story"}

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
    """Research Atlas's research dataset -> a readable knowledge-base block.

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

    ``research`` is Research Atlas's structured research dataset. When present, the
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

    system = _SCRIPT_SYSTEM + _NEUTRAL_SYSTEM + (_RESEARCH_SYSTEM if research_block else "")
    body = _strip_artifacts(llm.generate_text(system, user, max_output_tokens=2500).strip())
    if style not in _UNCHECKED_KINDS:
        body = _neutral(body, system, user, theirs=f"{prompt}\n{research_block}")

    return segment_script(body, wpm=wpm, source="prompt")


def _neutral(body: str, system: str, user: str, *, theirs: str) -> str:
    """A narration whose narrator passes no verdict the brief did not (see _NEUTRAL_SYSTEM).

    A draft with faults (neutrality.faults) is written once more, told what they are. The
    redraft is kept only if it has fewer and is not a stub of the first; otherwise the first
    stands, as a narration that judges in a sentence is still more use than none.
    """
    found = neutrality.faults(body, theirs)
    if not neutrality.fault_count(found):
        return body
    logger.info("Narration is not neutral (%s); writing it again", found)
    ask = _NEUTRAL_REWRITE.format(faults=_faults_note(found), draft=body)
    try:
        again = _strip_artifacts(llm.generate_text(system, f"{user}\n\n{ask}", max_output_tokens=2500).strip())
    except Exception:  # noqa: BLE001 - the first draft is still a narration
        logger.warning("Narration redraft failed; keeping the first draft", exc_info=True)
        return body
    if not again or _word_count(again) < _word_count(body) * 0.6:
        return body
    still = neutrality.faults(again, theirs)
    return again if neutrality.fault_count(still) < neutrality.fault_count(found) else body


def _faults_note(found: dict) -> str:
    lines = []
    if found["aftermath"]:
        lines.append(
            "It opens on something that came after the claim. Open on the claim itself: what "
            "is alleged, against whom, and by whom."
        )
    if found["sentences"]:
        lines.append(
            "These sentences pass a verdict on the subject in the narrator's own voice. Say "
            "whose verdict or finding each one is, or tell it without one:"
        )
        lines += [f"- {s}" for s in found["sentences"]]
    return "\n".join(lines)


def from_chapters(
    book_title: str | None,
    chapters: list[dict],
    *,
    direction: str | None = None,
    style: str = "documentary",
    tone: str | None = None,
    target_duration_sec: int | None = None,
    doc_style: str | None = None,
) -> Script:
    """The chosen chapters of an imported document -> one narration about them.

    The Import tab's way to a script: the topics someone picked (each a chapter's title and
    the summary documents.read wrote) are the material, and the conversation's brief is the
    direction. Written by generate_script, so it is timed, styled and segmented exactly as a
    narration written from a prompt.
    """
    title = (book_title or "").strip() or "the imported document"
    lines = []
    if direction and direction.strip():
        lines += [direction.strip(), ""]
    lines.append(
        f'Adapt these parts of "{title}" into one narration. Stay grounded in what they say: '
        "this is an adaptation, not an invention. Carry one through-line across them so it "
        "plays as one piece, not a list of recaps."
    )
    for ch in chapters:
        lines.append(f"- {(ch.get('title') or 'Untitled').strip()}: {(ch.get('summary') or '').strip()}")
    return generate_script(
        "\n".join(lines), style=style, tone=tone, target_duration_sec=target_duration_sec, doc_style=doc_style,
    )


# ── Enhance ──────────────────────────────────────────────────────────────────
# The Write tab's "Enhance Narration". Everything here is written to protect one thing:
# the piece the user already wrote. The model may reword freely, and may not add to,
# remove from, or reorder what was said — a "better" narration that argues something
# slightly different is a worse answer than leaving the text alone.
#
# The writer also chooses a VOICE (see services/enhance_styles) — dark, angry,
# investigative, cinematic, calm or light. That decides how the narration sounds, and
# nothing else: it is spliced into the prompt below between the standing permissions and
# the standing prohibitions, precisely so it can never be read as permission to break one.
_ENHANCE_SYSTEM = """
You are an editor rewriting narration that will be read aloud in a video.

The words are the writer's. Return the SAME narration, in the voice described below.

{voice}

You may:
- Fix grammar, spelling, punctuation and tense.
- Reword a sentence, freely, so it carries the voice above — including changing its
  register, its rhythm and how formal or informal it sounds.
- Split a sentence that is hard to follow, or join two that are stronger together.
- Replace a vague or repeated word with a more precise one.
- Re-order the words WITHIN a sentence for emphasis.

You must not:
- Add any fact, name, number, claim, opinion or example that is not already there.
- Add imagery, atmosphere, drama or emphasis that the writer's text does not support —
  the voice is in how a thing is said, never in saying more than was said.
- Remove anything the writer said, or soften a point they made.
- Change the order in which the ideas are presented.
- Change the language, the person (I / we / you), or the tense the piece is written in.
- Add a title, headings, labels, stage directions or markdown.
- Add or remove paragraph breaks.

Where a passage already reads in this voice, leave it exactly as it is. Returning the text
unchanged is a valid answer.

Output the narration and nothing else: no preamble, no explanation, no quotation marks
around it, no notes about what you changed.
""".strip()

# Below this share of the original word count, the reply is not a polish — it is a summary,
# or an answer the model ran out of room to finish. Either way it is not what was asked for,
# and handing it back would quietly replace the writer's script with a shorter one.
_ENHANCE_MIN_RATIO = 0.6


def enhance_narration(text: str, style: Optional[str] = None) -> str:
    """Narration the user wrote -> the same narration, in the voice they picked.

    `style` is one of services/enhance_styles.ENHANCE_STYLES. Missing or unrecognised
    resolves to the calm documentary voice rather than raising — see enhance_styles.entry
    for why a creative preference is never worth failing a request over.

    Pinned to DeepSeek rather than following the provider switch (see llm.deepseek_text).

    The output budget is sized from the input instead of fixed. The reply is the whole
    piece rewritten, so one ceiling for every length would truncate a long script and give
    back a narration missing its ending — which is exactly the failure the ratio check
    below refuses to pass on.
    """
    src = (text or "").strip()
    if not src:
        return ""

    system = _ENHANCE_SYSTEM.format(voice=enhance_styles.voice_block(style))
    words = _word_count(src)
    budget = max(600, min(8000, int(words * 2.2) + 300))
    out = _strip_artifacts(llm.deepseek_text(system, src, max_output_tokens=budget))

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
