"""All DeepSeek prompts for Book Alchemy.

Book Alchemy is an *adaptation* system, not an authoring system. It converts a
written work into spoken-word audio that carries the same information, in the
same order, at roughly the same density. It is deliberately NOT a course
generator: it does not teach, expand, editorialise, or restructure the author's
material, and it does not manufacture lessons where the source has none.

The rule repeated in every prompt: everything in the output must be traceable to
something the author actually wrote. Where the source is silent, the output is
silent.

Note on naming: the pipeline phase and the DB column are still called
``curriculum`` (changing them would break in-flight rows), but the job that
phase now performs is *segmentation* — deciding where to cut one continuous work
into parts — which is what ``SEGMENT_SYSTEM`` below asks for.
"""
from __future__ import annotations

import json

GROUNDING = (
    "You are part of Book Alchemy, which adapts an author's written work into "
    "spoken-word audio. You are adapting a work, not teaching a course, "
    "summarising a text, or writing about it.\n"
    "\n"
    "Hard rules:\n"
    "- Use ONLY the supplied source text. Never add facts, examples, analogies, "
    "background, or conclusions that are not in it.\n"
    "- Never state or imply WHY the author wrote something, what they intended, "
    "felt, believed, or hoped to achieve, unless the source says so explicitly. "
    "Report what the source says, never what it means.\n"
    "- Never interpret, evaluate, or draw out implications. No 'this shows', "
    "'this means', 'in other words', 'the significance is', 'notice how', "
    "'this illustrates'.\n"
    "- Keep the author's order, emphasis, and level of detail. A point the source "
    "makes once is made once.\n"
    "- Preserve the author's terminology, names, figures, and quotations exactly. "
    "Do not modernise, correct, or normalise them.\n"
    "- If the source is unclear, incomplete, or self-contradictory, carry that "
    "across unchanged. Do not resolve it, and do not remark on it.\n"
    "- You MAY re-word for the ear: unpack abbreviations, turn print-only "
    "constructs (tables, footnotes, 'see figure 3') into speakable phrasing, and "
    "smooth sentences that only parse on the page. Meaning must not change."
)

# ----------------------------------------------------------------- title
TITLE_SYSTEM = (
    "You produce a clean, concise title for a written work being adapted to "
    "audio. Return ONLY the title as plain text — no quotes, no markdown, no "
    "prefixes like 'Book Alchemy:' or 'Course:'. Keep it under 70 characters. "
    "Fix any garbled or mis-encoded characters. If the original title is "
    "reasonable, clean and shorten it; otherwise derive one from the content "
    "sample. Name the work as it is — do not add course-style framing such as "
    "'A Guide to', 'Mastering', or 'Foundations of'."
)


def title_user(raw_title: str, sample: str) -> str:
    return (
        f"Original title (may be messy, overly long, or mis-encoded):\n{raw_title}\n\n"
        f"Content sample:\n\"\"\"\n{sample}\n\"\"\""
    )


# ----------------------------------------------------------------- analyze
ANALYZE_SYSTEM = (
    GROUNDING
    + "\n\nTask: index ONE excerpt of the source so the pipeline can later decide "
    "where to cut the work into parts. This index is planning metadata — it is "
    "never narrated and never reaches the listener, so it should be terse.\n"
    "Return a JSON object with arrays: concepts, definitions, frameworks, "
    'examples, conclusions. Each item is {"title": str, "body": str}.\n'
    "- `title`: what the source covers at that point, in the author's own words "
    "where possible (under 12 words).\n"
    "- `body`: one line recording what this excerpt actually says about it "
    "(under 30 words).\n"
    "Only include what THIS excerpt supports; empty arrays are fine and normal. "
    "At most 8 items per array. You are labelling the source, not explaining or "
    "assessing it."
)


def analyze_user(chunk_text: str, *, chapter: str | None) -> str:
    head = f"[Chapter/Section: {chapter}]\n" if chapter else ""
    return f"{head}Source excerpt:\n\"\"\"\n{chunk_text}\n\"\"\""


# ----------------------------------------------------------------- segmentation
# (the pipeline phase is still named "curriculum" — see the module docstring)
SEGMENT_SYSTEM = (
    GROUNDING
    + "\n\nTask: cut the source into the FEWEST audio lessons that still give a "
    "comfortable listen. You are dividing one continuous work into parts — you "
    "are not designing a syllabus.\n"
    "\n"
    "Rules:\n"
    "- Lessons follow the source's own order. Never reorder, never regroup by "
    "theme, and never add an introduction, overview, summary, or review lesson.\n"
    "- Cut only where the source itself changes subject — a new chapter or "
    "section, or a clear shift of topic — or where a part would otherwise run "
    "past the length limit you are given.\n"
    "- Covering several topics is NOT a reason to split. A short work that "
    "touches five topics is still ONE lesson.\n"
    "- Every source segment belongs to exactly one lesson. None is skipped and "
    "none appears twice.\n"
    "- Use the smallest lesson count allowed by the limits you are given. If one "
    "lesson is allowed, return one lesson.\n"
    "\n"
    'Return JSON: {"work_title": str, "sessions": [{"ordinal": int, "title": str, '
    '"summary": str, "start_segment": int, "topics": [str, ...]}]}\n'
    "- `start_segment` is the number of the first source segment in that lesson. "
    "The first lesson must start at segment 0, and `start_segment` must strictly "
    "increase. Each lesson runs up to the segment before the next one starts.\n"
    "- `title`: what the source actually covers there, in the author's terms. No "
    "invented course-style titles ('Foundations of...', 'Key Takeaways').\n"
    "- `summary`: one plain sentence under 20 words describing the content.\n"
    "- `topics`: the segment topics in that range, in source order — a coverage "
    "checklist for the narration step."
)


def segment_user(
    digests: list[dict],
    *,
    min_lessons: int,
    suggested_lessons: int,
    max_lessons: int,
    max_lesson_words: int,
) -> str:
    lines = []
    for d in digests:
        head = f"[segment {d['ordinal']}]"
        if d.get("chapter"):
            head += f" ({d['chapter']})"
        head += f" ~{d['words']} words"
        topics = "; ".join(d.get("topics") or []) or "(no index entries)"
        lines.append(f"{head}\n  {topics}")

    return (
        f"The source is {len(digests)} segments in order.\n\n"
        f"Lesson count: you must return at least {min_lessons} and at most "
        f"{max_lessons} lessons. {suggested_lessons} is the expected number for a "
        f"source this long. Prefer fewer.\n"
        f"No single lesson may cover more than about {max_lesson_words} words of "
        f"source.\n\n"
        f"Source segments:\n" + "\n".join(lines)
    )


# ----------------------------------------------------------------- session script
SCRIPT_SYSTEM = (
    GROUNDING
    + "\n\nTask: turn the source segments below into spoken narration — a "
    "faithful audio edition of this part of the work.\n"
    "\n"
    "- Work through the segments in the order given. Nothing in them is dropped, "
    "and nothing is added.\n"
    "- No lesson framing whatsoever. Do not open by announcing what will be "
    "covered ('In this lesson...', 'We explore...'), do not close with a recap, "
    "summary, takeaways, or 'to recap', and do not write bridging commentary "
    "between topics ('First...', 'Second...', 'Now let us turn to...'). Begin on "
    "the source's own first point and end on its last.\n"
    "- Do not refer to the source from outside it. No 'the author says', 'the "
    "letter states', 'the writer reveals', 'this passage describes'. Narrate the "
    "content directly, in the source's own voice.\n"
    "- Where the author writes in the first person, stay in the first person. "
    "Where the source is a letter, a list, or a quotation, keep it recognisable "
    "as one.\n"
    "- Write for a single narrator reading aloud: plain sentences, no markdown, "
    "no headings, no bullets, no speaker labels, no stage directions.\n"
    "- Spell out what the ear cannot resolve: currency and figures as words, "
    "'e.g.' as 'for example', symbols as their spoken form. Keep initialisms the "
    "author uses, but make sure they read cleanly aloud.\n"
    "\n"
    "Return plain text only."
)


def script_user(
    session_title: str,
    outline: list[str],
    chunks: list[dict],
    *,
    source_words: int,
    min_words: int,
    max_words: int,
    feedback: list[dict] | None = None,
) -> str:
    outline_txt = "\n".join(f"- {b}" for b in (outline or [])) or "- (follow the segments)"
    excerpts = "\n\n".join(
        f"[segment {i + 1}{_ref(c)}]\n{c['text']}" for i, c in enumerate(chunks)
    )

    fix = ""
    if feedback:
        problems = "\n".join(
            f"- {f.get('type', 'problem')}: {f.get('claim', '')}"
            + (f" ({f.get('reason')})" if f.get("reason") else "")
            for f in feedback[:12]
        )
        fix = (
            "\nA previous attempt at this narration was rejected for the following. "
            "Write a new one that does not repeat them:\n"
            f"{problems}\n"
        )

    return (
        f"Part title: {session_title}\n\n"
        f"Topics in this part, in source order — a coverage checklist; every one "
        f"must be present in the narration:\n{outline_txt}\n"
        f"{fix}\n"
        f"Length: the source below runs about {source_words} words. Your narration "
        f"must be between {min_words} and {max_words} words. Coming in under that "
        f"means you dropped information; going over means you padded. Neither is "
        f"acceptable.\n\n"
        f"Source segments, in order:\n{excerpts}"
    )


# ----------------------------------------------------------------- validation
VALIDATION_SYSTEM = (
    GROUNDING
    + "\n\nTask: check a narration script against the source segments it was "
    "built from. Check fidelity in BOTH directions — what was added and what was "
    "lost.\n"
    "\n"
    "Flag each of:\n"
    '- "added": a statement, example, figure, or conclusion in the script that '
    "the segments do not support.\n"
    '- "inferred": the script gives a motive, intent, feeling, cause, or '
    "significance the source does not state, or explains what something means.\n"
    '- "omitted": something the segments say that the script leaves out.\n'
    '- "filler": narration carrying no source information — an introduction '
    "announcing what will be covered, a recap or takeaways, a transition, "
    "commentary about the source or its author, or the same point made twice.\n"
    "\n"
    "Re-wording for the ear is expected and is NOT a problem. Following the "
    "author's own wording closely is expected and is NOT a problem. Judge only "
    "against the segments supplied — never against outside knowledge.\n"
    "\n"
    'Return JSON: {"supported": bool, "flagged": [{"type": '
    '"added"|"inferred"|"omitted"|"filler", "claim": str, "reason": str}], '
    '"notes": str}. Set `supported` to false if any flagged item is material.'
)


def validation_user(script: str, chunks: list[dict]) -> str:
    excerpts = "\n\n".join(
        f"[segment {i + 1}{_ref(c)}]\n{c['text']}" for i, c in enumerate(chunks)
    )
    return (
        f"Source segments:\n{excerpts}\n\n"
        f"Narration script to check:\n\"\"\"\n{script}\n\"\"\""
    )


def _ref(chunk: dict) -> str:
    bits = []
    if chunk.get("chapter"):
        bits.append(str(chunk["chapter"]))
    if chunk.get("page_start"):
        pe = chunk.get("page_end") or chunk["page_start"]
        bits.append(f"p.{chunk['page_start']}" + (f"-{pe}" if pe != chunk["page_start"] else ""))
    return f" — {', '.join(bits)}" if bits else ""
