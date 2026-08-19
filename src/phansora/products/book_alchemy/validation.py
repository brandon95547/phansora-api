"""Source-grounding validation for generated session scripts.

Before a session is finalized, its script is checked against the exact source
chunks it was built from. A session that drifts is regenerated; if it still
drifts after a few attempts it is marked ``flagged`` (never silently shipped).

``scrub_apparatus`` is the deterministic floor under that check — see its
docstring for why a floor is needed at all.
"""
from __future__ import annotations

import re
from typing import Any

from . import prompts
from .deepseek_client import DeepSeekClient


# ------------------------------------------------------------------- scrub
# A spoken lesson never contains an email address, a web address, or a filename.
# Four model-level guards stand above this one — the front-matter boundary at
# parse time, the packaging rule in the indexer, the same rule in the writer,
# and the checker's `attributed` flag — and each lowers the odds without being a
# floor. "flagged" is not a stop: a session that fails validation on every
# attempt is still written out and still narrated (pipeline._phase_sessions), so
# without something deterministic here, "the course read out the author's email"
# stays possible no matter how the prompts are worded.
#
# Sentence granularity is deliberate. Excising the address alone leaves a
# dangling fragment mid-narration ("He can be reached at ."), and an address
# never appears in a sentence that is otherwise teaching the subject.
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]*[a-z]{2,}\b", re.I)
_URL = re.compile(r"\bhttps?://\S+|\bwww\.[\w-]+\.\w{2,}", re.I)
_FILENAME = re.compile(
    r"\b[\w-]+\.(?:pdf|txt|docx?|epub|mobi|azw3?|html?|rtf|zip|exe)\b", re.I
)
# The splitter the rest of the codebase uses (shared/utils/chunking.py).
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def scrub_apparatus(script: str) -> tuple[str, list[str]]:
    """Drop every sentence naming an email address, a URL, or a filename.

    Returns ``(clean_script, removed_sentences)``. The caller logs what came out:
    a scrub that fires means one of the guards above it leaked, which is worth
    chasing even though the listener never hears the result.
    """
    if not script or not script.strip():
        return script, []

    removed: list[str] = []
    paragraphs: list[str] = []
    for para in re.split(r"\n\s*\n+", script):
        kept: list[str] = []
        for sentence in _SENTENCE_SPLIT.split(para):
            text = sentence.strip()
            if not text:
                continue
            if _EMAIL.search(text) or _URL.search(text) or _FILENAME.search(text):
                removed.append(text)
                continue
            kept.append(text)
        if kept:
            paragraphs.append(" ".join(kept))

    if not removed:
        return script, []
    return "\n\n".join(paragraphs), removed


async def validate_script(
    client: DeepSeekClient,
    *,
    script: str,
    chunks: list[dict],
    concepts: list[dict] | None = None,
    previously_taught: list[dict] | None = None,
) -> dict[str, Any]:
    """Return {supported: bool, flagged: [...], notes: str}.

    ``concepts`` is the lesson's coverage contract — the ideas the analyze phase
    indexed in these segments — and ``previously_taught`` is what earlier lessons
    already covered. BOTH must be the same lists the writer was given. Hand the
    checker a different contract and it marks the lesson down for not doing
    something nobody asked it to do, and the retry loop then chases that phantom:
    with ``concepts`` missing, coverage falls back to the raw segments, every
    dropped clause reads as an omission, and the regeneration loop rebuilds the
    1:1 re-voicing this product exists to avoid.

    Fails closed on parse errors (treats as unsupported) so a broken check can
    never wave unsupported content through."""
    try:
        result = await client.chat_json(
            system=prompts.VALIDATION_SYSTEM,
            user=prompts.validation_user(
                script, chunks, concepts=concepts, previously_taught=previously_taught
            ),
            max_output_tokens=1500,
        )
    except Exception as exc:  # noqa: BLE001
        return {"supported": False, "flagged": [], "notes": f"validation error: {exc}"}

    if not isinstance(result, dict):
        return {"supported": False, "flagged": [], "notes": "invalid validation response"}
    return {
        "supported": bool(result.get("supported", False)),
        "flagged": result.get("flagged") or [],
        "notes": str(result.get("notes") or ""),
    }
