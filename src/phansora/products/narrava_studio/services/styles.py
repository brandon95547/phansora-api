"""The documentary's house style — ONE choice that governs the whole pipeline.

The editor picks a style once, beside the documentary type, before a word is written. From
then on every AI pass reads the same entry: the narration is WRITTEN in it, the storyboard
is SHOT in it, and anything added later dresses itself the same way. A film whose script
is investigative and whose pictures are cinematic is the failure this module exists to
prevent.

Each style therefore carries two facets of the same intent:

  ``narration``  how the script is written — voice, tone, sentence rhythm, structure and
                 storytelling approach.
  ``visual``     what the camera is pointed at — subjects, light, framing, mood.
  ``pacing``     how quickly the picture turns over, within the bounds the edit rules fix.

They are written as the concrete things a writer or a camera would DO, not as adjectives:
"cinematic" on its own moves an LLM barely at all, while "golden-hour light, slow push-ins,
subject treated as story rather than record" moves it a long way.

The ids are the contract with the browser (assets/js/admin/studio/util/doc-style.js).
Renaming one on either side silently unstyles every generation rather than erroring, so
both sides pin the list in a test.
"""
from __future__ import annotations

from typing import Dict, Optional

DOC_STYLES: Dict[str, Dict[str, str]] = {
    "cinematic": {
        "label": "Cinematic",
        "narration": (
            "Write it as a feature documentary. Evocative, image-led prose that sets a "
            "scene before it states a fact; concrete sensory detail over abstraction. "
            "Sentences vary in length and are built to be spoken slowly, with room to "
            "breathe. Open on a moment rather than a summary, carry one emotional through-"
            "line from beginning to end, and close on resonance rather than recap. Tone is "
            "elevated and humane — the subject is a story, not a record."
        ),
        "visual": (
            "Feature-documentary polish. Composed, deliberate frames with shallow depth of "
            "field and strong foreground/background separation; golden-hour or motivated "
            "practical light; wide anamorphic-feeling establishing shots and slow push-ins "
            "or tracking moves. Mood is emotive and elevated — the subject treated as story "
            "rather than as record. Favour texture, silhouette and atmosphere (haze, dust, "
            "rain, backlight) over literal illustration of the words."
        ),
        "pacing": (
            "Let images hold. Lean towards the longer end of the range so a frame can land "
            "before the next one replaces it."
        ),
    },
    "investigative": {
        "label": "Investigative",
        "narration": (
            "Write it as a current-affairs investigation. Plain declarative sentences, "
            "precise nouns, no ornament; every claim attributed to who said it or what "
            "shows it. Structure the piece as a case being laid out — the question, then "
            "the evidence in the order it makes sense, then what it implies. Pacing is "
            "measured and accumulating, tightening as the evidence stacks. Tone is sober "
            "and probing; state what is known, name what is not, and end on the open "
            "question rather than tying a bow on it."
        ),
        "visual": (
            "The look of a current-affairs investigation. Observational framing that feels "
            "captured rather than staged: documents, ledgers, records, screens, evidence "
            "boards, signage, building exteriors, doorways, and long-lens shots from across "
            "a street. Cool, neutral, slightly desaturated light. Mood is sober and "
            "probing. Favour the concrete artefact — the file, the receipt, the nameplate — "
            "over anything symbolic."
        ),
        "pacing": (
            "Change the picture on each new piece of evidence and hold it while that piece "
            "is explained — steady and deliberate, never restless."
        ),
    },
    "historical": {
        "label": "Historical",
        "narration": (
            "Write it as a history. Formal, authoritative prose in the past tense, anchored "
            "in dates, places and named people; period vocabulary where it is natural and "
            "never anachronistic. Structure it chronologically, setting each event in the "
            "wider context that made it possible. Pacing is unhurried and even, giving "
            "consequence its weight. Tone is measured and elegiac — close by reflecting on "
            "what the events came to mean rather than on what happened next."
        ),
        "visual": (
            "Period reconstruction and archive. Grainy monochrome or faded colour stock, "
            "archival footage and stills, sepia photographs, maps, letters, newsprint and "
            "museum artefacts; period-correct clothing, architecture, vehicles and objects "
            "in every frame. Static frames or slow moves across stills. Mood is "
            "authoritative and elegiac. Never allow a modern object, vehicle or garment "
            "into a historical shot."
        ),
        "pacing": (
            "Unhurried. Stills and archive need time to be read, so favour longer holds and "
            "change the picture when the period or the place changes."
        ),
    },
    "educational": {
        "label": "Educational",
        "narration": (
            "Write it as a clear explainer. Short, plain sentences carrying one idea each; "
            "everyday words in place of jargon, and any necessary term defined the first "
            "time it is used. Structure it as what it is, why it matters, then how it "
            "works, building each point on the one before and recapping briefly at the "
            "end. Pacing is steady and even, with no sudden leaps in difficulty. Tone is "
            "calm, friendly and confident; use a concrete analogy wherever an abstraction "
            "would otherwise stand alone."
        ),
        "visual": (
            "Clear explanatory visuals. Bright, evenly lit, uncluttered frames with a "
            "single unambiguous subject; diagrams, cutaways, scale models, demonstrations, "
            "labelled close-ups and clean overhead shots of a process. Neutral or bright "
            "backgrounds. Mood is calm, friendly and legible. Prefer the plainly readable "
            "image to the atmospheric one — if the viewer has to decode the picture, it is "
            "the wrong picture."
        ),
        "pacing": (
            "One idea, one picture. Change the visual when the point changes and not "
            "before — a diagram the viewer is still reading must not be cut away from."
        ),
    },
    "dark": {
        "label": "Dark / Suspense",
        "narration": (
            "Write it as suspense. Short, clipped sentences; concrete, unsettling detail; "
            "no reassurance and no summarising ahead of yourself. Open cold on a detail "
            "that is wrong before explaining anything, withhold the full picture, and "
            "release it late. Structure it as a slow escalation towards the thing the "
            "viewer has been half-told. Pacing tightens as it goes and leaves deliberate "
            "silences. Tone is tense and withholding; end on a question, not an answer."
        ),
        "visual": (
            "Suspense and unease. Low-key lighting with deep shadow and hard pools of "
            "light; night, fog, rain, empty corridors, treelines, headlights, reflections "
            "and half-seen movement. Tight or off-centre framing, obscured faces, slow "
            "creeping moves. Desaturated palette of cold blues and sickly amber. Mood is "
            "tense and withholding — show the edge of the thing rather than the thing."
        ),
        "pacing": (
            "Hold long and uncomfortably early on, then cut tighter as the tension rises. "
            "The turnover itself is part of the dread."
        ),
    },
    "news": {
        "label": "News / Broadcast",
        "narration": (
            "Write it as a broadcast news package. Short, self-contained sentences in the "
            "active voice; the most important fact first, then the detail behind it, then "
            "the context. Attribute every claim to its source and stay impartial — no "
            "adjectives that judge. Pacing is brisk and even, with no dwelling. Tone is "
            "neutral and current; close on what happens next rather than on a reflection."
        ),
        "visual": (
            "Broadcast news package. Straightforward, well-lit, immediately readable "
            "shots: establishing exteriors, crowds, traffic, press conferences, government "
            "and corporate buildings, and plain b-roll of the sector in question. "
            "Eye-level framing, steady camera, neutral daylight or standard studio light. "
            "Mood is impartial and current. Nothing stylised — the picture supports the "
            "fact rather than interpreting it."
        ),
        "pacing": (
            "Brisk and regular. Keep the visuals moving at the shorter end of the range so "
            "the package never stalls on one frame."
        ),
    },
}


def entry(style: Optional[str]) -> Optional[Dict[str, str]]:
    """The style's entry, or None for a missing or unrecognised name.

    Unknown and missing both resolve to None rather than raising: the style is a creative
    preference, and refusing to write a script or build a storyboard because a client sent
    a name this version does not know would be a worse answer than doing it unstyled.
    """
    return DOC_STYLES.get((style or "").strip().lower())


def style_label(style: Optional[str]) -> str:
    found = entry(style)
    return found["label"] if found else ""


def visual_block(style: Optional[str]) -> str:
    """The style preamble for a picture-writing prompt, or "" when unstyled.

    It rides in the USER message rather than the system prompt because it is
    request-scoped, exactly like the narration and the avoid-list: the system prompts stay
    the stable statement of HOW TO EDIT, and nothing about the job changes when the style
    does.
    """
    found = entry(style)
    if not found:
        return ""
    # The last paragraph is the guard rail. A model handed a mood will otherwise start
    # rewriting the narration spans to suit it, and the spans are what every placeholder's
    # position is anchored to — so the style is told exactly how far its authority runs.
    return (
        f"DOCUMENTARY STYLE — {found['label']}.\n"
        f"{found['visual']}\n"
        f"Pacing within this style: {found['pacing']}\n"
        "This governs what each picture shows, how it looks, and how quickly the visuals "
        "turn over. It does not license you to change the words: every scene's text must "
        "still be the narration verbatim, in order, with no gaps and no overlaps, and the "
        "pacing bounds you were given still hold. Make the edit as instructed, then shoot "
        "it in this style.\n\n"
    )


def narration_block(style: Optional[str]) -> str:
    """The style preamble for the script writer, or "" when unstyled.

    Same registry as the storyboard reads, which is the entire point: the words and the
    pictures are written from one description of the film rather than two that drift.
    """
    found = entry(style)
    if not found:
        return ""
    return (
        f"DOCUMENTARY STYLE — {found['label']}.\n"
        f"{found['narration']}\n"
        "This governs how the narration is WRITTEN — voice, tone, sentence rhythm, "
        "structure and storytelling approach. It does not change the subject, and it never "
        "overrides the output rules or the target length: still only the words to be "
        "spoken, with no scene directions, labels or headings.\n\n"
    )
