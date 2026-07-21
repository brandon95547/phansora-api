"""Salvaging JSON out of LLM responses.

Two failure modes this handles, both routine with chat models:

  * The model wraps valid JSON in prose or a ```json fence.
  * The model was cut off mid-generation (``finish_reason == "length"``), leaving a
    structurally invalid document whose *completed* items are still perfectly good.

Extracted from Book Alchemy's DeepSeek client so the reasoning-model client can reuse it
rather than carry a second copy of logic this fiddly. ``deepseek-reasoner`` needs it more,
not less: it does not support JSON mode at all, so every structured response from it
arrives as free text that has to be parsed leniently.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional



def parse_json_loose(raw: str) -> Any:
    """Best-effort JSON extraction from a model response."""
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Empty DeepSeek JSON response.")
    # Strip ```json ... ``` fences if present.
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced { } or [ ] span.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = raw.find(open_ch)
        end = raw.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from DeepSeek response: {raw[:300]}")


def repair_truncated_json(raw: str) -> Optional[str]:
    """Salvage a JSON doc that was cut off mid-generation (finish=='length').

    Walks the text tracking string/escape and bracket state, rewinds to the last
    point where a container ({…} or […]) closed cleanly, and appends closers for
    whatever is still open. For Book Alchemy's schema — an object of arrays of
    ``{title, body}`` objects — this keeps every complete item and drops only the
    trailing, cut-off one. Returns a parseable string, or None if nothing
    complete came through."""
    raw = (raw or "").strip()
    fenced = re.match(r"^```(?:json)?\s*(.*)$", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1)
    start = raw.find("{")
    alt = raw.find("[")
    if start == -1 or (alt != -1 and alt < start):
        start = alt
    if start == -1:
        return None

    stack: list[str] = []
    in_string = False
    escape = False
    cut: Optional[int] = None          # index (exclusive) of a safe truncation point
    cut_stack: tuple[str, ...] = ()     # containers still open at that point

    for i in range(start, len(raw)):
        ch = raw[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
            # A safe cut point right after an (empty) container opens, so a doc
            # truncated *before its first element closes* still salvages to a valid
            # empty container instead of failing (drops only the cut-off element).
            cut, cut_stack = i + 1, tuple(stack)
        elif ch == "[":
            stack.append("]")
            cut, cut_stack = i + 1, tuple(stack)
        elif ch in "}]":
            if not stack:
                break  # unbalanced close; best guess is everything before it
            stack.pop()
            cut, cut_stack = i + 1, tuple(stack)

    if cut is None:
        return None
    return raw[start:cut] + "".join(reversed(cut_stack))
