"""The narration form's conversation: what the video should say, worked out by asking.

The AI tab opens on one box, "Describe your narration"; the Import tab on the topics a
document was found to have. Whatever the person writes first is ``description``. From
there DeepSeek asks one short follow-up question at a time, each with a few suggested
answers, about what would most change the narration and is not yet answered. When it knows
enough (or after MAX_QUESTIONS, or when the person says to write now) it stops, and returns
a brief for the writer and, for a document, which of its topics to use.

Every turn carries the whole conversation, so this keeps no state: the editor holds it.
"""
from __future__ import annotations

import logging
from typing import Any

from . import llm

logger = logging.getLogger("narrava-studio.converse")

# Follow-up questions in one conversation, at most. Most need two or three.
MAX_QUESTIONS = 4
MAX_CHOICES = 4
QUESTION_CHARS = 200
CHOICE_CHARS = 60
BRIEF_CHARS = 3000
# A topic's summary as shown to the model: enough to tell topics apart.
TOPIC_SUMMARY_CHARS = 400

SYSTEM = """
You help someone plan the narration for a video by asking short follow-up questions, one at
a time, about what is still unclear. They have already said what the video is about.

Ask about what would change the narration most and has not been answered yet: the angle or
focus, who it is for, the tone or feeling, how it should open or end, what to include or
leave out. Never ask something they have already answered. Ask one thing only: a question
that joins two with "and" or "or" is two questions, so ask the one that matters more.

A question is plain and friendly, under 20 words, with no preamble and no compliments.
Offer 2 to 4 short suggested answers (under 8 words each) that fit what they have said.

Stop as soon as you know enough to write a good narration, usually after two or three
questions. When you stop, set "done" to true and write the brief: one paragraph of
direction for the writer, covering the subject, the angle, the audience, the tone and the
shape of the piece. Put in only what they said or clearly meant. Never add facts.
""".strip()

TOPICS_SYSTEM = """
They are choosing from the topics found in a document they imported, listed below by the
number they see beside each. Work out which topics they want to use. "topics" is the numbers
of the chosen topics, in the order the narration should use them. Choose from the list only.
If they have not said, use the recommended ones.
""".strip()

REPLY = """
Return JSON only:
{"done": bool, "question": str, "choices": [str], "brief": str, "topics": [int]}
While asking: "done" false, "question" and "choices" set, "brief" "".
When done: "done" true, "question" "", "choices" [], "brief" set.
""".strip()

FINISH = "Do not ask another question. Set \"done\" to true and write the brief now."


def _clip(value: Any, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit].strip()


def _user(req: Any, finishing: bool) -> str:
    lines = []
    if req.video:
        lines.append(f"The video: {_clip(req.video, 200)}")
    if req.kind == "topics":
        lines.append(f"Imported document: {_clip(req.document_title, 200) or 'Untitled'}")
        lines.append("Topics found:")
        # Numbered as the editor shows them, from 1, so "use 1 and 3" means what it says.
        for n, t in enumerate(req.topics, start=1):
            mark = " (recommended)" if t.recommended else ""
            lines.append(f"{n}. {_clip(t.title, 200)}{mark}: {_clip(t.summary, TOPIC_SUMMARY_CHARS)}")
        lines.append("")
        lines.append(f"Which topics they want, in their words: {req.description.strip()}")
    else:
        lines.append(f"Their description of the narration: {req.description.strip()}")
    if req.turns:
        lines.append("")
        lines.append("The conversation so far:")
        for turn in req.turns:
            lines.append(f"Q: {turn.question.strip()}")
            lines.append(f"A: {turn.answer.strip()}")
    if finishing:
        lines.append("")
        lines.append(FINISH)
    return "\n".join(lines)


def fallback_brief(req: Any) -> str:
    """Their own words, as a brief: what the writer gets if the model gives none."""
    parts = [req.description.strip()]
    parts += [f"{t.question.strip()} {t.answer.strip()}" for t in req.turns]
    return _clip(" ".join(p for p in parts if p), BRIEF_CHARS)


def _topics(raw: Any, req: Any) -> list[int]:
    """The chosen topics, from the numbers the model names (as shown, from 1) to the chapter
    indexes the document has, without repeats; the recommended ones (then all of them) when
    it named none it could have meant."""
    by_number = {n: t.index for n, t in enumerate(req.topics, start=1)}
    chosen: list[int] = []
    for n in raw if isinstance(raw, list) else []:
        index = by_number.get(n) if isinstance(n, int) and not isinstance(n, bool) else None
        if index is not None and index not in chosen:
            chosen.append(index)
    if chosen:
        return chosen
    recommended = [t.index for t in req.topics if t.recommended]
    return recommended or [t.index for t in req.topics]


def next_turn(req: Any) -> dict:
    """The next question, or the brief. Always a usable answer: a reply the model got wrong
    becomes "done" with their own words as the brief, never an error mid-conversation."""
    finishing = bool(req.finish) or len(req.turns) >= MAX_QUESTIONS
    system = SYSTEM + ("\n\n" + TOPICS_SYSTEM if req.kind == "topics" else "") + "\n\n" + REPLY
    data = llm.deepseek_json(system, _user(req, finishing), max_output_tokens=1200)
    if not isinstance(data, dict):
        data = {}

    question = _clip(data.get("question"), QUESTION_CHARS)
    choices: list[str] = []
    for c in data.get("choices") or []:
        text = _clip(c, CHOICE_CHARS)
        if text and text.lower() not in (x.lower() for x in choices):
            choices.append(text)
    done = finishing or bool(data.get("done")) or not question
    out = {
        "done": done,
        "question": None if done else question,
        "choices": [] if done else choices[:MAX_CHOICES],
        "brief": (_clip(data.get("brief"), BRIEF_CHARS) or fallback_brief(req)) if done else "",
        "topics": _topics(data.get("topics"), req) if req.kind == "topics" else [],
    }
    return out
