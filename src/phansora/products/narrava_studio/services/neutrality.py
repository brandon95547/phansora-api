"""Neutrality: the AI passes no verdict, in its own voice, on the subject someone brings it.

Narrava Studio is for telling the story the person wants told. On a disputed subject, a
model left to itself settles the question before they have said a word: the conversation
offered only angles that assumed the answer, and the writer opened a documentary by calling
its subject a conspiracy theory. Both are asked not to (converse.SYSTEM, script's
_NEUTRAL_SYSTEM), and both are checked here, because asking is not enough.

The check is for verdict words: rumor, hoax, myth, conspiracy theory, debunked, baseless,
fake, "is false" and their kind. In the person's own words (or an imported document's) they
are that framing, and may be echoed back; in the AI's voice they are not. In a narration a
verdict someone else passed is reportable too, so long as it says whose it is ("critics
called it a hoax"): that is reporting, not ruling.

A narration can rule without a verdict word ("No such ring was found." "The story, in the
end, was not there.") and can open on the aftermath, which tells the viewer the answer
before the question. Neither shows in its words, so a narration is also read by the model
(faults), which is a far easier task than writing neutrally in a house style that wants a
dramatic cold open.
"""
from __future__ import annotations

import logging
import re

from . import llm

logger = logging.getLogger("narrava-studio.neutrality")

# "True" and "false" count only where they judge. "Lies" and "fringe" not where they are
# something else: the town lies in a valley, on the fringe of the city.
_VERDICT = re.compile(r"\b(?:" + "|".join((
    r"rumou?rs?", r"hoax(?:es)?", r"myths?", r"conspirac(?:y|ies)(?:\s+theor(?:y|ies))?",
    r"mis-?information", r"dis-?information", r"propaganda", r"pseudo-?scien(?:ce|tific)",
    r"fringe(?!\s+(?:of|benefits?)\b)", r"debunk(?:s|ed|ing)?", r"dispro(?:ve|ves|ven|ved|ving)",
    r"discredit(?:s|ed)?", r"refut(?:e|es|ed|ations?)", r"baseless", r"unfounded", r"fake",
    r"lies(?!\s+(?:in|on|at|with|within|beneath|under|behind|between|beyond|ahead|before|"
    r"buried|hidden|dormant|still|open|empty|across|along|near|outside|inside)\b)",
    r"fact-?check(?:s|ed|ing|ers?)?",
    r"(?:is|are|was|were|proven|proved|shown\s+to\s+be)\s+(?:not\s+)?(?:true|false)",
    r"false\s+(?:claims?|stor(?:y|ies)|accounts?|allegations?|narratives?)",
)) + r")\b", re.IGNORECASE)

# A sentence that says whose verdict it is: someone said it, or a court or an official
# body found it. On the record, so the narrator may report it.
_ATTRIBUTED = re.compile(r"\b(?:" + "|".join((
    r"says?", r"said", r"saying", r"call(?:s|ed|ing)?", r"claim(?:s|ed|ing)?",
    r"argu(?:e|es|ed|ing)", r"insist(?:s|ed)?", r"maintain(?:s|ed)?", r"believ(?:e|es|ed|ers)",
    r"according\s+to", r"den(?:y|ies|ied)", r"reject(?:s|ed)?", r"dismiss(?:es|ed)?",
    r"label(?:s|ed|led)?", r"brand(?:s|ed)?", r"rul(?:e|es|ed|ing)", r"found",
    r"conclud(?:e|es|ed)", r"report(?:s|ed)?", r"wrote", r"writes?", r"told",
    r"counter(?:s|ed)?", r"answer(?:s|ed)?", r"accus(?:e|es|ed)", r"alleg(?:e|es|ed)",
    r"charged", r"convicted", r"indicted", r"pleaded", r"sentenced", r"critics", r"skeptics",
    r"sceptics", r"supporters", r"to\s+(?:them|him|her)",
)) + r")\b", re.IGNORECASE)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


_COPULAS = {"is", "are", "was", "were", "proven", "proved", "shown"}


def _stem(verdict: str) -> str:
    """A verdict's core word without its ending, so "rumors" in their words covers "rumor",
    "conspiracy theory" covers "conspiracy", and "is false" covers "false claims"."""
    words = verdict.lower().split()
    core = words[-1] if words[0] in _COPULAS else words[0]
    return re.sub(r"(?:es|e|s|ed|ing|ers|er)$", "", re.sub(r"ies$", "y", core))


def verdicts(text: str, theirs: str) -> list[str]:
    """The verdict words in ``text`` that ``theirs`` (what they said) does not use."""
    own = {_stem(m.group(0)) for m in _VERDICT.finditer(theirs or "")}
    return sorted({m.group(0).lower() for m in _VERDICT.finditer(text or "") if _stem(m.group(0)) not in own})


def judging_sentences(text: str, theirs: str) -> list[str]:
    """The sentences of a narration that pass a verdict in the narrator's own voice: a
    verdict word they did not use, in a sentence that does not say whose it is."""
    return [
        s for s in (p.strip() for p in _SENTENCE_RE.split(text or ""))
        if s and verdicts(s, theirs) and not _ATTRIBUTED.search(s) and '"' not in s and "“" not in s
    ]


REVIEW = """
You check a video narration for neutrality. The narrator may report; it may not rule.

You are given the brief the narration was written from, then the narration. The brief is the
person's own framing: where it takes a side, or asks for the story as one side sees it, a
narration that does so is following the brief, not judging.

Find each sentence in which the narrator, in its own voice and with no one named as its
source, passes a verdict on a disputed claim: says or implies it is true or false, real or
unreal, proven or baseless ("No such ring was found.", "The story was not there.", "No
evidence substantiated it.", "The claims travelled further than the facts."), or calls the
people who hold or reject it fooled, credulous or the like. A finding stated as the
narrator's own fact counts ("No evidence was found."); the same finding attributed does not
("Police said they found no evidence."). Events on the record do not count: an attack, an
arrest, a sentence, a denial.

Then say how it opens. "claim": its first sentences are the disputed claim itself, what is
alleged and against whom. "aftermath": it opens on something that came after the claim, such
as an attack, an arrest, a ruling or a debunking, before saying what was claimed. "other":
the subject is not disputed, or the brief asked for that opening.

Return JSON only: {"opening": "claim" | "aftermath" | "other", "sentences": [str]}, each
sentence copied exactly from the narration. When the subject is not disputed:
{"opening": "other", "sentences": []}.
""".strip()

# The brief as the reviewer reads it: what it needs to know whose framing is whose.
_REVIEW_BRIEF_CHARS = 4000


def faults(text: str, theirs: str) -> dict:
    """What makes a narration less than neutral: the sentences in which the narrator rules
    on the subject ("sentences"), and whether it opens on the aftermath ("aftermath").

    The verdict words always (judging_sentences); the model's reading on top, when it can be
    had. Without it (no DeepSeek, a timeout) the words alone are the check.
    """
    found = judging_sentences(text, theirs)
    aftermath = False
    try:
        data = llm.deepseek_json(
            REVIEW, f"The brief:\n{theirs[:_REVIEW_BRIEF_CHARS]}\n\nThe narration:\n{text}", max_output_tokens=800,
        )
    except Exception:  # noqa: BLE001 - the words still stand as the check
        logger.warning("Neutrality review failed; checking the words alone", exc_info=True)
        data = None
    if isinstance(data, dict):
        aftermath = data.get("opening") == "aftermath"
        for sentence in data.get("sentences") or []:
            sentence = str(sentence or "").strip()
            # Only what is really there: a sentence the reviewer misquoted cannot be pointed at.
            if sentence and sentence in text and sentence not in found:
                found.append(sentence)
    return {"sentences": found, "aftermath": aftermath}


def fault_count(found: dict) -> int:
    return len(found["sentences"]) + int(found["aftermath"])
