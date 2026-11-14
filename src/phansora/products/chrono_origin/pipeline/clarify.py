"""Before research runs: is it clear which subject was meant?

"Revelation" may be the last book of the Bible, a film or an album; "Mercury" a planet, a
god or an element. A trace of the wrong one costs a credit and a minute or two, so the
dashboard asks first, with one light call and no search: the model's own knowledge is
enough to know that a word names several things. When it does, the person picks one, and
the pick becomes the request's context.

Only WHICH subject is asked. Scope, period and detail within one subject are the
research's own business, and a question about them would stand in front of every request.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..models import ClarifyOption, ClarifyRequest, ClarifyResponse

MAX_OPTIONS = 5
LABEL_MAX = 60
CONTEXT_MAX = 80
QUESTION_MAX = 120

# What each kind of research is about to make, in the words of the question.
_RESEARCH = {
    "chronology": "a timeline of",
    "genealogy": "the genealogy of",
    "origins": "the origin of",
    "interpretations": "interpretations of",
    "parallels": "parallels to",
    "translations": "how translations word",
    "establishment": "the power structure of",
}


def clarify_prompt(req: ClarifyRequest) -> str:
    research = _RESEARCH.get(req.kind or "", "research on")
    context = (req.context or "").strip()
    context_line = f', with this context given: "{context}"' if context else ", with no context given"
    return f"""\
A person is about to research {research} "{req.title.strip()}"{context_line}. Before it runs, decide whether it is clear which subject they mean.

The subject is UNCLEAR when the words could reasonably name more than one different thing this research could be about: a book and an event, a person and a place, a historical event and a film or album, two people or places of the same name. It is CLEAR when one meaning is so dominant that nobody would mean another for this research, or when the context settles it.

Ask only WHICH subject is meant. Never ask about scope, period or detail within one subject.

Return a JSON object and nothing else, either

{{"clear": true}}

or

{{"clear": false, "question": "Which Revelation do you mean?", "options": [{{"label": "The Book of Revelation", "context": "the last book of the New Testament"}}, {{"label": "Revelation (2014 film)", "context": "the 2014 film"}}]}}

Give 2 to {MAX_OPTIONS} options, the likeliest first. "label" names the subject as a reader would recognize it, at most {LABEL_MAX} characters. "context" is a short phrase, at most {CONTEXT_MAX} characters, that tells the research which one it is."""


def _text(value: Any, limit: int) -> str:
    return " ".join(str(value).split())[:limit] if isinstance(value, str) else ""


def parse_clarify(data: Optional[Dict[str, Any]]) -> ClarifyResponse:
    """The model's answer, kept only where it is usable: fewer than two meanings is clear."""
    if not isinstance(data, dict) or data.get("clear") is not False:
        return ClarifyResponse(clear=True)
    options: List[ClarifyOption] = []
    for raw in data.get("options") or []:
        if not isinstance(raw, dict):
            continue
        label, context = _text(raw.get("label"), LABEL_MAX), _text(raw.get("context"), CONTEXT_MAX)
        if label and context and label.casefold() not in {o.label.casefold() for o in options}:
            options.append(ClarifyOption(label=label, context=context))
    if len(options) < 2:
        return ClarifyResponse(clear=True)
    question = _text(data.get("question"), QUESTION_MAX) or "Which one do you mean?"
    return ClarifyResponse(clear=False, question=question, options=options[:MAX_OPTIONS])


def clarify(client: Any, req: ClarifyRequest) -> ClarifyResponse:
    """One light call through the research client, without a search."""
    return parse_clarify(client.reason_json(clarify_prompt(req), use_reasoning_model=False))
