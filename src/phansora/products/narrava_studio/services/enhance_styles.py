"""Writing voices for "Enhance Narration" — how the same narration is SAID.

Separate registry from services/styles.DOC_STYLES, deliberately. That one is the film's
house style: picked once, before a word is written, and read by the script writer and the
storyboard alike so the words and the pictures match. This one is picked at the moment the
button is pressed, applies to one pass over one piece of text, and never leaves the enhance
prompt — the same narration can be polished calm today and dark tomorrow without changing
what the film looks like.

Each entry carries two things:

  ``direction``  the concrete moves that produce the voice — sentence length, which words
                 to prefer, how a paragraph is built — written as things a rewriter DOES,
                 because "make it dramatic" moves a model barely at all.
  ``example``    one line in that voice. Every example is the SAME underlying fact rewritten
                 six ways, which is what makes them useful both to the model (a sample of
                 the register, ahead of any adjective) and to the user, who reads them in
                 the picker to choose.

None of them license new content. The writer's facts, order and claims are fixed by the
rules in script._ENHANCE_SYSTEM; a voice only decides how those facts are worded. A "more
dramatic" narration that says something slightly different is the failure this file has to
avoid, so every direction below says so in its own terms.

The ids are the contract with the browser (assets/js/admin/studio/util/enhance-styles.js).
Renaming one on either side silently drops the voice rather than erroring, so both sides
pin the list in a test.
"""
from __future__ import annotations

from typing import Dict, Optional

ENHANCE_STYLES: Dict[str, Dict[str, str]] = {
    "dark": {
        "label": "Dark / Suspenseful",
        "direction": (
            "Rewrite it in the voice of a suspense documentary. Short declarative "
            "sentences, some of them deliberate fragments, set against longer ones so the "
            "rhythm keeps stalling and starting again. State the fact, then let a short "
            "sentence land on what it implies. Prefer plain cold nouns to adjectives, and "
            "repetition (\"again and again\") to intensifiers. The tension comes from "
            "pacing and word order only — never manufacture dread with a detail, a threat "
            "or an implication the writer did not already put on the page."
        ),
        "example": (
            "Over the years, the same names kept appearing around the same organization. "
            "Again and again. And eventually, the pattern became difficult to ignore."
        ),
    },
    "angry": {
        "label": "Angry / Confrontational",
        "direction": (
            "Rewrite it in the voice of a presenter who has run out of patience with the "
            "subject. Active constructions with the accountable party as the grammatical "
            "subject. Rhetorical pressure — \"yet somehow nobody asked why\" — rather than "
            "insult or name-calling. Short sentences that land hard, contrasts joined with "
            "\"yet\" and \"and still\". Sharpen the point the writer already made; do not "
            "escalate it into a new accusation, and never blame anyone the text does not "
            "already hold responsible."
        ),
        "example": (
            "For years, the same people kept turning up around the same organization, yet "
            "somehow nobody seemed interested in asking why."
        ),
    },
    "investigative": {
        "label": "Investigative / Serious",
        "direction": (
            "Rewrite it in the voice of a filed report. Plain declarative sentences, "
            "precise nouns, no ornament and no adverbs of judgment. Reach for the language "
            "of the record — \"a review of the records shows\", \"according to\", \"the "
            "filings list\" — but only where the writer's text already rests on such a "
            "source; never invent an attribution to make a line sound sourced. State what "
            "is established as established, and leave what is uncertain sounding uncertain."
        ),
        "example": (
            "A review of the records shows that several individuals repeatedly intersected "
            "with the same organization over a number of years."
        ),
    },
    "cinematic": {
        "label": "Cinematic / Dramatic",
        "direction": (
            "Rewrite it in the voice of a feature documentary. Set the scene before stating "
            "the fact; let sentences turn on \"at first… but then\". Vary sentence length "
            "widely, build a paragraph toward the line that pays it off, and leave room to "
            "breathe. Every image must come from what the writer already wrote — reword and "
            "re-sequence for effect, but add no new picture, place, moment or feeling."
        ),
        "example": (
            "At first, the names seemed unrelated. But as the timeline unfolded, the same "
            "organization kept appearing between them."
        ),
    },
    "calm": {
        "label": "Calm / Documentary",
        "direction": (
            "Rewrite it in the voice of a measured documentary narrator. Even, unhurried "
            "sentences of moderate length. Keep the writer's careful qualifiers — "
            "\"indicate\", \"over an extended period\" — wherever they were careful. No "
            "drama, no salesmanship: the tone of someone reading a well-checked script "
            "aloud. This is the default voice, so when a choice is close, favour clarity "
            "over effect and leave a passage that already reads well exactly as it is."
        ),
        "example": (
            "The available records indicate that several of these individuals had "
            "connections to the same organization over an extended period."
        ),
    },
    "light": {
        "label": "Light / Conversational",
        "direction": (
            "Rewrite it in the voice of a presenter talking to one person. Contractions, "
            "everyday words, and second person where the writer already addresses the "
            "viewer. Lead with the human framing — \"at first glance… but look a little "
            "closer\" — and keep sentences short enough to say in one breath. Friendly, "
            "never flippant: a serious fact stays serious, it is simply said plainly."
        ),
        "example": (
            "At first glance, these people may not seem connected. But look a little "
            "closer, and one organization keeps showing up between them."
        ),
    },
}

# What an enhance with no style asked for gets. Calm is the narrator voice that changes the
# least about a piece — the safest thing to hand someone who never opened the picker, and
# the same default the picker itself opens on.
DEFAULT_ENHANCE_STYLE = "calm"


def entry(style: Optional[str]) -> Dict[str, str]:
    """The voice's entry, falling back to the default for a missing or unknown name.

    Unlike the house styles, this never resolves to None: an enhance always happens in some
    voice, and the calm one is the least surprising answer to a name this version does not
    know. Refusing to polish a paragraph over a spelling of a creative preference would be
    a worse outcome than polishing it calmly.
    """
    return ENHANCE_STYLES.get((style or "").strip().lower(), ENHANCE_STYLES[DEFAULT_ENHANCE_STYLE])


def style_label(style: Optional[str]) -> str:
    return entry(style)["label"]


def voice_block(style: Optional[str]) -> str:
    """The VOICE section of the enhance system prompt.

    Placed in the system prompt rather than beside the text, because for this call the
    voice IS the job: everything else in that prompt is the standing rule about what may
    not change, and the one instruction that differs between two presses of the button is
    which of these six the result should sound like.
    """
    found = entry(style)
    return (
        f"THE VOICE — {found['label']}.\n"
        f"{found['direction']}\n"
        f"A line written in this voice sounds like this:\n\"{found['example']}\"\n"
        "That example shows the register, the sentence rhythm and the word choice to aim "
        "for. It is not content: do not borrow its subject, its wording or its structure, "
        "and do not let it pull the narration toward what it happens to be about.\n"
        "This governs HOW the narration is said — nothing else. The facts, the claims, the "
        "order they arrive in and the language they are written in all stay exactly as the "
        "writer left them."
    )
