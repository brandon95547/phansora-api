"""Source-grounding validation for generated session scripts.

Before a session is finalized, its script is checked against the exact source
chunks it was built from. A session that drifts is regenerated; if it still
drifts after a few attempts it is marked ``flagged`` (never silently shipped).
"""
from __future__ import annotations

from typing import Any

from . import prompts
from .deepseek_client import DeepSeekClient


async def validate_script(
    client: DeepSeekClient,
    *,
    script: str,
    chunks: list[dict],
    previously_taught: list[dict] | None = None,
) -> dict[str, Any]:
    """Return {supported: bool, flagged: [...], notes: str}.

    ``previously_taught`` is what earlier lessons already covered, and must be the
    same list the writer was given: without it, a lesson that correctly passes over
    settled ground reads as an omission and the retry loop restores the repetition.

    Fails closed on parse errors (treats as unsupported) so a broken check can
    never wave unsupported content through."""
    try:
        result = await client.chat_json(
            system=prompts.VALIDATION_SYSTEM,
            user=prompts.validation_user(
                script, chunks, previously_taught=previously_taught
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
