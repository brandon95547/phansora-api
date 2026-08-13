"""All DeepSeek prompts for Book Alchemy.

Book Alchemy turns a written work into a spoken audio course — the source taught
back the way a college lecturer would teach it after reading it.

The narrator is the INSTRUCTOR, teaching the subject in the first person. It is
not a reviewer describing a book: no "the author states", no "he argues", no "in
this chapter". A listener should finish the course knowing the material without
ever being told where it came from. That is a voice rule, not a license — every
sentence still has to be something the source says.

A course also has a memory. Books circle back; a course that circles back with
them feels padded, so each lesson is told what the earlier lessons already
taught (see ``already_taught_block``) and passes over settled ground. The
validator is given the SAME list, because a lesson that correctly skips a repeat
would otherwise be marked down for omitting it and the retry loop would put the
repetition straight back in.

Every prompt separates TWO INDEPENDENT AXES. Conflating them is what produced the
worst bug this module has had:

  CONTENT is bounded by the source. Every fact, name, figure, example and
  conclusion comes from the excerpts. Where the source is silent, so is the
  lesson.

  EXPRESSION is entirely the instructor's. Every sentence is built fresh from an
  understanding of the material, in the instructor's own vocabulary and
  structure. None of the original's wording survives.

An earlier version stated these as one rule — "restating is the job, adding is
never allowed" — and surrounded it with instructions that all pulled toward
fidelity: keep quoted words exact, define a term the way the source defines it,
quote a striking phrase. Under that prompt the safest way to satisfy every
constraint was to stay close to the original, and a measured course came back
**46% verbatim** (5,580 of 12,155 words; 72 unbroken twenty-word runs; a single
143-word transcription). The model was not disobeying — copying is maximally
faithful, adds nothing, omits nothing, and lands inside the length band. It was
the optimal strategy the prompt permitted.

The first fix was framing: the excerpts are NOTES to understand, not text to
edit, and the lesson is written from memory afterwards; the rule was made
countable (~8 consecutive words) because "in your own words" is a quality a
model cannot check itself against, while a word run is. That held on short test
passages and failed on a real book — **82.1% verbatim** (2,881 of 3,509 words in
runs of eight or more, 107 lifted passages, the longest 158 words). At length, a
model with the prose in its context follows the prose, whatever the
instructions around it say.

So the writer no longer sees the prose. The analyze phase distills each excerpt
into meaning-complete notes in its own words (exact only for proper nouns,
numbers, dates and terms of art), the script phase teaches from those notes
with the source closed, and only the validator reads the original — checking
truth, coverage and copying after the fact. What is not in the writer's context
cannot be copied out of it; the residual risk is source wording that survives
into a note, which is why the analyze prompt bans it there.

The opposite failure is still real — prompts loose enough to teach once produced
a four-lesson course out of a one-page letter — and the length budget in
``pipeline.py`` remains the dial that holds it, though it is explicitly not a
target to be reached by borrowing.

WHAT COVERAGE MEANS. The lesson's obligation is the CONCEPT LIST, not the source
text. Every prompt below that talks about completeness takes an explicit list of
the ideas the analyze phase indexed in these segments, and "complete" means every
one of them is taught. It does not mean every sentence has a counterpart.

That distinction is the difference between a course and a reading. Earlier, the
validator was asked to flag anything "the segments say that the lesson never
teaches", judged against raw text — so every clause was an obligation, the length
floor sat just under parity, and the retry loop restored anything dropped. The
result was a 1:1 re-voicing: a Bible came back as ~370 lessons and ~108 hours,
because 780,000 source words could not become fewer than about 700,000 narrated
ones. Teaching an idea once is not the same as reproducing every instance of it,
and only the concept list can express that difference.

So: forty genealogy entries are ONE concept and are taught as what they are, not
recited. A law restated in three places is taught once. The compression that
falls out of this is not summarizing — every indexed idea still gets taught in
full — it is the removal of repetition and enumeration that only exists because
the source is a written document being read end to end.

NOTE: prompt wording alone cannot enforce this. The validator is a language model
asked to compare two texts, and a lifted passage reads to it as perfectly
grounded — which is how 46% passed. A deterministic n-gram check belongs in the
regeneration loop; these prompts reduce the rate, they do not guarantee it.

Note on naming: the pipeline phase and the DB column are still called
``curriculum`` (changing them would break in-flight rows), but the job that
phase now performs is *segmentation* — deciding where to cut one continuous work
into parts — which is what ``SEGMENT_SYSTEM`` below asks for.
"""
from __future__ import annotations

import json

# Raised from 8 when the index became the coverage contract rather than a hint
# for boundary placement. At 8, a dense excerpt silently shed ideas that the
# lesson was then never obliged to teach — invisible content loss. Paired with
# the `truncated` flag so the remaining ceiling is at least observable.
MAX_INDEX_ITEMS_PER_ARRAY = 14

GROUNDING = (
    "You are part of Book Alchemy, which turns a written work into a spoken audio "
    "course. Imagine you are a professor who has read the source, closed the book, "
    "and is now teaching what you learned to a class.\n"
    "\n"
    "Follow three rules:\n"
    "\n"
    "1. CONTENT: Teach every distinct idea you are given accurately and completely. "
    "Do not omit an idea.\n"
    "\n"
    "2. EXPRESSION: Use the source only for information. Explain the material naturally "
    "in your own words and sentence structures. Do not copy, closely paraphrase, or "
    "rewrite the source sentence by sentence.\n"
    "\n"
    "3. FIDELITY: Do not add facts, explanations, conclusions, implications, examples, "
    "motives, or interpretations that the source does not provide.\n"
    "\n"
    "Teach the subject directly. Do not mention the source, book, text, passage, chapter, "
    "or author.\n"
    "\n"
    "Repetition and long enumerations may be compressed when they communicate the same "
    "idea, but distinct ideas must remain."
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
    + "\n\nTask: take complete meaning-only notes on ONE excerpt of the source. "
    "The lesson is later written from these notes with the source closed, so these "
    "notes must preserve all of the information the lesson needs to teach.\n"
    "\n"
    "Compress WORDING and REPETITION, not INFORMATION. You may combine closely related "
    "facts into one note, but combining them must not remove distinct facts, examples, "
    "qualifications, relationships, definitions, or conclusions. If the excerpt makes "
    "six different factual points about one concept, one note may contain all six — "
    "but all six must survive in the note.\n"
    "\n"
    'Return a JSON object with: teachable (bool), truncated (bool), and the arrays '
    "concepts, definitions, frameworks, examples, conclusions. Each array item is "
    '{"title": str, "body": str}.\n'
    "- `title`: a short label for the idea, in your own words.\n"
    "- `body`: a meaning-complete account of everything THIS excerpt says that the "
    "lesson needs in order to teach that item accurately. Use your own wording. Keep "
    "exact only proper nouns, numbers, dates, direct quotations that must remain exact, "
    "and established terms of art. Do not impose an artificial word limit on a body; "
    "use as much space as needed to preserve the information, while remaining concise.\n"
    "- `teachable`: false if this excerpt is apparatus rather than material — for "
    "example a table of contents, index, page-number run, copyright/permissions page, "
    "cross-reference table, or publisher note about the edition. Set true for actual "
    "content, including substantive prefaces and introductions. When genuinely unsure, "
    "set true.\n"
    "- `truncated`: true ONLY if any teachable information had to be left out because "
    "of output or item limits. Never silently omit information to keep notes short.\n"
    "\n"
    "Do not add knowledge, interpretation, or implications that the excerpt does not "
    "provide. Do not create one item per sentence merely to preserve wording. Repeated "
    "instances of the same point may be represented once, but distinct information "
    "must remain.\n"
    "\n"
    f"At most {MAX_INDEX_ITEMS_PER_ARRAY} items per array. This is an item-count limit, "
    "not an information limit: when related material belongs together, preserve its "
    "full meaning in the body rather than dropping details."
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
        f"source. That figure already accounts for the lesson being shorter than "
        f"the source it teaches from, so do not discount it again.\n\n"
        f"Source segments:\n" + "\n".join(lines)
    )


# ----------------------------------------------------------------- session script
SCRIPT_SYSTEM = (
    GROUNDING
    + "\n\nTask: teach one spoken lesson from the concept notes below.\n"
    "\n"
    "The book is closed: you are not shown the source text, only the notes taken "
    "from it. The notes are the whole of what this lesson may contain.\n"
    "\n"
    "- Teach every idea on the note list. That list is the coverage contract, and "
    "it is also the boundary: state no fact, name, figure, event, example, or "
    "analogy the notes do not support — not from your own knowledge of the "
    "subject, however sure you are. A gap in the notes stays a gap.\n"
    "- A note is compressed; teaching it is not expanding it word by word. Say "
    "what it means, connect it to what came before, let one point lead to the "
    "next the way a lecture does.\n"
    "- Where a note records a repeated pattern (a genealogy, a rule over many "
    "cases), teach the pattern and its range. Do not invent the instances back.\n"
    "- If material was already taught in an earlier lesson, do not repeat it unless "
    "these notes add something new.\n"
    "- Follow the notes' order unless a small rearrangement makes the lesson clearer.\n"
    "- Say each point once.\n"
    "- Open directly on the material. Do not announce the lesson.\n"
    "- Do not close with a recap, summary, or takeaways.\n"
    "- Write as a single instructor speaking naturally to a class.\n"
    "- No markdown, headings, bullets, speaker labels, or stage directions.\n"
    "\n"
    "Return plain text only."
)

def already_taught_block(previously_taught: list[dict] | None) -> str:
    """What earlier lessons in this course have already covered.

    Both the writer and the checker need this, and they must be given the same
    text: a lesson told to pass over settled material would otherwise be marked
    down for omitting it, and the regeneration loop would put the repetition
    straight back in.
    """
    if not previously_taught:
        return ""
    lines = []
    for lesson in previously_taught:
        lines.append(f"- Lesson {lesson.get('ordinal')}: {lesson.get('title') or 'untitled'}")
        for topic in lesson.get("topics") or []:
            lines.append(f"    · {topic}")
    return "Already taught in earlier lessons of this course:\n" + "\n".join(lines) + "\n"


def concept_checklist(concepts: list[dict] | None) -> list[str]:
    """The lesson's coverage contract as flat lines, deduplicated in order.

    Built from what the analyze phase indexed in this lesson's own segments, so
    it is the same list the validator judges omission against. Each line carries
    the body as well as the title: "Covenant" alone does not tell a writer what
    has to be taught, and does not tell a checker what to look for.
    """
    lines: list[str] = []
    seen: set[str] = set()
    for c in concepts or []:
        title = str(c.get("title") or "").strip()
        body = str(c.get("body") or "").strip()
        text = f"{title} — {body}" if title and body else (title or body)
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        lines.append(text)
    return lines


def script_user(
    session_title: str,
    outline: list[str],
    *,
    min_words: int,
    max_words: int,
    concepts: list[dict] | None = None,
    feedback: list[dict] | None = None,
    previously_taught: list[dict] | None = None,
) -> str:
    # Deliberately NO source excerpts. The writer teaching with the prose in front
    # of it is what produced the 82%-verbatim course (see the module docstring);
    # the notes are the only material, so source wording cannot be carried over.
    # The validator still reads the excerpts — see validation_user.
    checklist = concept_checklist(concepts) or list(outline or [])
    notes_txt = "\n".join(f"- {b}" for b in checklist) or "- (no notes were indexed for this lesson)"
    covered = already_taught_block(previously_taught)
    if covered:
        covered = (
            f"\n{covered}"
            "Anything on that list is settled. Teach it again only if these notes "
            "genuinely extend it, and then teach only what is new.\n"
        )

    fix = ""
    if feedback:
        problems = "\n".join(
            f"- {f.get('type', 'problem')}: {f.get('claim', '')}"
            + (f" ({f.get('reason')})" if f.get("reason") else "")
            for f in feedback[:12]
        )
        fix = (
            "\nA previous attempt was rejected. Write the lesson again from the "
            "notes, fixing these problems:\n"
            f"{problems}\n"
        )

    return (
        f"Lesson title: {session_title}\n\n"
        f"Concept notes, in source order — the complete material for this lesson. "
        f"Every idea on this list must be taught, and nothing beyond it may be "
        f"stated:\n"
        f"{notes_txt}\n"
        f"{covered}{fix}\n"
        f"Length: aim for {min_words} to {max_words} spoken words. Do not add "
        f"material just to reach the range — the notes decide the content; the "
        f"range only paces how fully each idea is unpacked."
    )


# ----------------------------------------------------------------- validation
VALIDATION_SYSTEM = (
    GROUNDING
    + "\n\nTask: check one lesson against its source and concept list. Judge three "
    "things independently: coverage, originality, and fidelity.\n"
    "\n"
    "COVERAGE: every distinct idea on the concept list must be taught. Do not require "
    "every source sentence, repeated instance, or enumerated example to appear.\n"
    "\n"
    "ORIGINALITY: the lesson must be written in genuinely new language. Flag copied "
    "wording when 8 or more consecutive words match the source, or when a sentence "
    "closely follows the source's sentence structure with substitutions. Proper nouns, "
    "numbers, dates, and established terms are not copying.\n"
    "\n"
    "FIDELITY: everything taught must be supported by the source. Flag added facts, "
    "unsupported explanations, implications, motives, interpretations, or examples.\n"
    "\n"
    "Also flag attribution if the lesson talks about the author, book, text, chapter, "
    "or source instead of teaching the subject directly. Flag filler if it adds empty "
    "introductions, recaps, or unnecessary repetition.\n"
    "\n"
    "Use these flag types:\n"
    '- "added": unsupported factual material or example.\n'
    '- "inferred": unsupported explanation, implication, motive, interpretation, or significance.\n'
    '- "omitted": a concept on the supplied list is not actually taught.\n'
    '- "copied": 8+ consecutive matching words or close structural paraphrase.\n'
    '- "filler": unnecessary intro, recap, padding, or repeated teaching.\n'
    '- "attributed": mentions the source or writer instead of teaching directly.\n'
    "\n"
    "If material was already taught in earlier lessons, do not mark it omitted when "
    "the current lesson correctly skips it.\n"
    "\n"
    'Return JSON: {"supported": bool, "flagged": [{"type": '
    '"added"|"inferred"|"omitted"|"copied"|"filler"|"attributed", "claim": str, '
    '"reason": str}], "notes": str}.\n'
    'Set supported to false for any copied item, and for any material non-copying problem.'
)

def validation_user(
    script: str,
    chunks: list[dict],
    *,
    concepts: list[dict] | None = None,
    previously_taught: list[dict] | None = None,
) -> str:
    excerpts = "\n\n".join(
        f"[segment {i + 1}{_ref(c)}]\n{c['text']}" for i, c in enumerate(chunks)
    )
    covered = already_taught_block(previously_taught)
    # The same list the writer was handed. If these two ever diverge, the lesson
    # is marked down for not doing something it was never asked to do, and the
    # retry loop chases a target that does not exist.
    checklist = concept_checklist(concepts)
    contract = (
        "Concepts this lesson was required to teach (the coverage contract):\n"
        + "\n".join(f"- {line}" for line in checklist)
        + "\n\n"
    ) if checklist else (
        "No concept list was indexed for these segments. Judge wording, added "
        "content and attribution as usual, but do not flag anything as "
        '"omitted" — there is no contract to check coverage against.\n\n'
    )
    return (
        f"{contract}"
        f"Source segments (for checking truth and wording, NOT a coverage "
        f"checklist):\n{excerpts}\n\n"
        + (f"{covered}\n" if covered else "")
        + f"Narration script to check:\n\"\"\"\n{script}\n\"\"\""
    )


def _ref(chunk: dict) -> str:
    bits = []
    if chunk.get("chapter"):
        bits.append(str(chunk["chapter"]))
    if chunk.get("page_start"):
        pe = chunk.get("page_end") or chunk["page_start"]
        bits.append(f"p.{chunk['page_start']}" + (f"-{pe}" if pe != chunk["page_start"] else ""))
    return f" — {', '.join(bits)}" if bits else ""
