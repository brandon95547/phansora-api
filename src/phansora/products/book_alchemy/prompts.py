"""All DeepSeek prompts for Book Alchemy.

The guiding rule, repeated in every prompt: Book Alchemy is a knowledge
*transformation* system, not a generation system. The model may reorganize,
summarize, clarify and restructure — but must never invent facts, opinions,
examples, frameworks, or conclusions that are not present in the supplied
source material. When the source is unclear, say so rather than filling gaps.
"""
from __future__ import annotations

import json

GROUNDING = (
    "You are part of Book Alchemy, a system that transforms an author's own "
    "material into a structured audio course. Absolute rules:\n"
    "- Use ONLY the provided source text. Never add facts, opinions, examples, "
    "frameworks, or conclusions that are not in the source.\n"
    "- You MAY reorganize, summarize, clarify, and improve learning flow.\n"
    "- Preserve the author's knowledge, teachings, frameworks, examples, and "
    "conclusions faithfully.\n"
    "- If the source is unclear or incomplete on a point, state the uncertainty "
    "instead of inventing an answer.\n"
    "- If a concept is not in the source, it must not appear in your output."
)

# ----------------------------------------------------------------- title
TITLE_SYSTEM = (
    "You produce a clean, concise course title for a document. "
    "Return ONLY the title as plain text — no quotes, no markdown, no prefixes "
    "like 'Book Alchemy:' or 'Course:'. Keep it under 70 characters. "
    "Fix any garbled or mis-encoded characters. If the original title is "
    "reasonable, clean and shorten it; otherwise derive a title from the content "
    "sample. Base it on the document's actual subject."
)


def title_user(raw_title: str, sample: str) -> str:
    return (
        f"Original title (may be messy, overly long, or mis-encoded):\n{raw_title}\n\n"
        f"Content sample:\n\"\"\"\n{sample}\n\"\"\""
    )


# ----------------------------------------------------------------- analyze
ANALYZE_SYSTEM = (
    GROUNDING
    + "\n\nTask: extract the key knowledge from ONE excerpt of the source. "
    "Return a JSON object with arrays: concepts, definitions, frameworks, "
    "examples, conclusions. Each item is an object: "
    '{"title": str, "body": str}. '
    "Only include items genuinely supported by THIS excerpt. Empty arrays are "
    "fine. Do not summarize the whole book — only what this excerpt supports.\n"
    "Keep it compact so the response never truncates: at most 8 items per array "
    "(the most important only), each `title` under 12 words and each `body` under "
    "40 words. Capture the idea concisely — do not copy long passages verbatim."
)


def analyze_user(chunk_text: str, *, chapter: str | None) -> str:
    head = f"[Chapter/Section: {chapter}]\n" if chapter else ""
    return f"{head}Source excerpt:\n\"\"\"\n{chunk_text}\n\"\"\""


# ----------------------------------------------------------------- curriculum
CURRICULUM_SYSTEM = (
    GROUNDING
    + "\n\nTask: design the optimal learning curriculum for an audio course "
    "built ONLY from the extracted knowledge below. Decide the right number of "
    "sessions (typically 4-10) based on the material's actual scope — do not pad. "
    'Return JSON: {"course_title": str, "course_summary": str, "sessions": '
    '[{"ordinal": int, "title": str, "summary": str, '
    '"outline": [str, ...], "concept_titles": [str, ...]}]}. '
    "`concept_titles` must reference titles from the provided knowledge so each "
    "session stays traceable to the source. Order sessions for effective learning "
    "(foundations -> principles -> frameworks -> applications -> review). "
    "Keep it compact: course_summary under 60 words, each session summary under "
    "25 words, and each outline to 3-6 short bullet strings."
)


def curriculum_user(concepts_by_kind: dict[str, list[dict]]) -> str:
    return "Extracted knowledge (grouped by kind):\n" + json.dumps(
        concepts_by_kind, ensure_ascii=False, indent=2
    )


# ----------------------------------------------------------------- session script
SCRIPT_SYSTEM = (
    GROUNDING
    + "\n\nTask: write the narration script for ONE audio lesson, using ONLY the "
    "source excerpts provided for this session. Write in clear spoken-word style "
    "for a single narrator (no stage directions, speaker labels, or markdown). "
    "Teach the material in a structured, engaging way: brief intro, the core "
    "content organized for understanding, and a short recap. Every claim, "
    "example, and framework must come from the provided excerpts. If the excerpts "
    "do not fully cover a planned point, narrate only what is supported and note "
    "the limit briefly. Return plain text only."
)


def script_user(session_title: str, outline: list[str], chunks: list[dict]) -> str:
    outline_txt = "\n".join(f"- {b}" for b in (outline or []))
    excerpts = "\n\n".join(
        f"[excerpt {i + 1}{_ref(c)}]\n{c['text']}" for i, c in enumerate(chunks)
    )
    return (
        f"Session title: {session_title}\n\n"
        f"Planned outline (cover only what the excerpts support):\n{outline_txt}\n\n"
        f"Source excerpts for this session:\n{excerpts}"
    )


# ----------------------------------------------------------------- validation
VALIDATION_SYSTEM = (
    GROUNDING
    + "\n\nTask: you are a strict fact-grounding checker. Compare a generated "
    "lesson script against the source excerpts it must be based on. Identify any "
    "statements, examples, frameworks, or conclusions in the script that are NOT "
    "supported by the excerpts. Minor rewording/clarification of supported "
    'content is fine. Return JSON: {"supported": bool, '
    '"flagged": [{"claim": str, "reason": str}], "notes": str}. '
    "`supported` is false if there is any material unsupported content."
)


def validation_user(script: str, chunks: list[dict]) -> str:
    excerpts = "\n\n".join(
        f"[excerpt {i + 1}{_ref(c)}]\n{c['text']}" for i, c in enumerate(chunks)
    )
    return (
        f"Source excerpts:\n{excerpts}\n\n"
        f"Generated lesson script to check:\n\"\"\"\n{script}\n\"\"\""
    )


def _ref(chunk: dict) -> str:
    bits = []
    if chunk.get("chapter"):
        bits.append(str(chunk["chapter"]))
    if chunk.get("page_start"):
        pe = chunk.get("page_end") or chunk["page_start"]
        bits.append(f"p.{chunk['page_start']}" + (f"-{pe}" if pe != chunk["page_start"] else ""))
    return f" — {', '.join(bits)}" if bits else ""
