"""Prompts for the writing model.

Each builder returns ``(system, user)`` for one request. They are written the way a brief
to a capable colleague would be: who the reader is, what the output is for, and what
shape it must take, rather than a list of rigid rules. Most of the wording comes from the
standalone Nebiru app this product replaces.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .schemas import (
    AnalyzeRequest,
    BackgroundChoicesRequest,
    Blueprint,
    BlueprintRequest,
    BlueprintSource,
    ChapterRequest,
    ChatTurn,
    IllustrationPromptsRequest,
    ImageBrief,
    ImageComposeRequest,
    ImageDesignsRequest,
    RemixQuestionsRequest,
    Rewrite,
    SourceChapterText,
)

SUMMARY_MARKER = "===SUMMARY==="
NOTES_MARKER = "===NOTES==="
# Where a picture goes, written by the model on a line of its own inside the chapter.
ILLUSTRATION_LINE = "[[ILLUSTRATION: what the picture shows | caption]]"

ROLE = (
    "You are the writing engine inside Talari Book Studio, where a person plans, writes and illustrates "
    "a book. What you return is placed straight into their book or their planning screen, and they read "
    "and edit it there, so write the finished thing itself: no greetings, and no notes about what you did "
    "unless you are asked for them."
)

ILLUSTRATION_BRIEF = (
    "Each picture is described in one or two sentences for an illustrator: a single concrete scene, with "
    "the subject, what they are doing, the setting, the light and the mood. Describe what is seen, not "
    "ideas; where the text is about an idea, show a situation, object or place that embodies it. Leave "
    "out any art style, medium or artist, because the book's illustration style is applied separately, "
    "and never ask for words, lettering, signs, labels or book titles in the picture."
)

MARKDOWN = (
    "Format it as light markdown: \"## \" before each section heading and \"### \" for a smaller heading "
    "inside a section; paragraphs separated by one blank line; *italics* and **bold** used sparingly; "
    "\"- \" or \"1. \" entries only where the material truly is a list; \"> \" before a quotation set apart "
    "from the text. No tables, links, images or code."
)

DEPTHS = {
    "light": (
        "Light: keep the source's structure and meaning, the same chapters in the same order saying the "
        "same things, in clearer, better language."
    ),
    "moderate": (
        "Moderate: keep the source's substance but reorganise it where that serves the reader (merge, split "
        "or reorder chapters) and expand explanations that are thin."
    ),
    "deep": (
        "Deep: a distinct new book on the same material, with its own structure, sequence and way of "
        "presenting things. Draw on the source for substance, not for shape."
    ),
    "full": (
        "Full: the source is broad inspiration only. Plan an original book that takes its subject or "
        "spirit in its own direction; it need not follow the source's content, structure or order."
    ),
}

# How each depth asks for a chapter to be written from its source.
DEPTH_WRITING = {
    "light": (
        "This chapter rewrites the source material below. Keep its structure and everything it says (every "
        "fact, idea, example, name and event, in the same order) in clearer, better language of your own. "
        "Build new sentences rather than swapping in synonyms, and keep proper names, numbers, dates and "
        "technical terms as they are."
    ),
    "moderate": (
        "This chapter is based on the source material below. Keep its substance, reorganise it where that "
        "helps the reader, and expand explanations that are thin. Write it in your own words throughout."
    ),
    "deep": (
        "This chapter draws on the source material below for substance, but presents it in its own way, "
        "with its own structure and examples. Write it in your own words throughout."
    ),
    "full": (
        "The source material below is inspiration only. Take from it whatever serves this chapter; the "
        "chapter itself is original."
    ),
}

OPTIONS = {
    "keep-tone": "Keep the source's tone and voice.",
    "simplify": "Make it simpler and more accessible than the source.",
    "expand": "Go further than the source: deeper explanations and more detail.",
    "modern-examples": "Replace dated examples and references with current ones.",
    "change-audience": "Write for the audience given in the brief, which is not the source's audience.",
    "illustrations": "The book will be illustrated: favour moments and ideas that lend themselves to pictures.",
}

MODES = {
    "idea": "The author has an idea for a book and wants a plan for it.",
    "rewrite": "The author is making a new book by rewriting existing material, the sources below.",
    "extract": (
        "The author wants a new book drawn out of existing material, the sources below: not the whole of "
        "it, but what serves the book described in the brief. Use the parts that serve it and leave out "
        "the rest."
    ),
    "documents": (
        "The author has a set of documents, the sources below, and wants them made into one coherent book: "
        "combine, order and connect what they hold, without covering the same material twice."
    ),
}

LENGTHS = {
    # chapters, words per chapter — for an idea; a book from sources is sized by them
    "short": (6, 1200),
    "medium": (10, 2000),
    "long": (16, 3000),
}
# The chapter length a book from sources is divided by, for the count it is planned at.
SOURCE_CHAPTER_WORDS = 2500


def source_chapters(target_words: int) -> int:
    """How many chapters a book of ``target_words`` from sources is planned at."""
    return max(1, round(target_words / SOURCE_CHAPTER_WORDS))

REVISIONS = {
    "rewrite": (
        "Rewrite the passage so it reads better: clearer, more vivid, better paced. Keep the meaning, the "
        "facts, the point of view and roughly the same length ({words} words)."
    ),
    "expand": (
        "The passage is {words} words. Expand it to about {double} words by developing what is already "
        "there: more concrete detail, fuller explanation, the steps or beats it currently skips. Do not add "
        "new plot turns or new factual claims."
    ),
    "shorten": (
        "The passage is {words} words. Cut it to about {half} words. Keep what matters most and the "
        "author's voice; remove repetition, throat-clearing and anything the passage does not need."
    ),
    "simplify": (
        "Make the passage simpler to read: plainer words, shorter sentences, one idea at a time, and any "
        "term a newcomer would not know explained where it first appears. Keep everything it says, and "
        "its voice."
    ),
    "tone": (
        "Rewrite the passage in this tone: {tone}. Keep the meaning, the facts and roughly the same length; "
        "change the voice, rhythm and word choice."
    ),
    "continue": (
        "Continue the text from where the passage ends: write what comes next, about {more} words, in the "
        "same voice, carrying it forward naturally{toward}. Return only the new text that follows the "
        "passage, not the passage itself."
    ),
    "flow": (
        "Improve the flow of the passage: smooth the transitions between its sentences and paragraphs and "
        "into the text around it, vary the rhythm, and remove abrupt jumps. Keep its content, its facts and "
        "its length."
    ),
    "examples": (
        "Add one or two concrete examples that make the passage's point clearer, woven in where they fit, "
        "and keep everything already there. Prefer examples that are plainly illustrative over specific "
        "factual claims; if you do include a specific fact, name it in a note so the author can check it."
    ),
    "custom": "Do what the author asks with the passage: {instruction}",
}


def _lines(*pairs: tuple[str, str]) -> str:
    return "\n".join(f"{label}: {value.strip()}" for label, value in pairs if value and value.strip())


def _clip(text: str, limit: int) -> str:
    """At most ``limit`` characters, cut at a word and marked when cut."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " […]"


def _tail(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return "[…] " + text[-limit:].split(" ", 1)[-1]


def sample(text: str, limit: int) -> str:
    """A chapter cut to ``limit`` characters: its opening, then a passage from its middle."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    head = int(limit * 0.65)
    middle = limit - head
    opening = text[:head].rsplit(" ", 1)[0]
    start = max(head, len(text) // 2 - middle // 2)
    passage = text[start : start + middle]
    passage = passage.split(" ", 1)[-1].rsplit(" ", 1)[0]
    return f"{opening} […]\n\n[…] {passage} […]"


def spread(text: str, limit: int, pieces: int = 4) -> str:
    """At most ``limit`` characters taken from evenly through a text, each piece verbatim."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    size = limit // pieces
    step = (len(text) - size) // max(1, pieces - 1)
    taken = []
    for n in range(pieces):
        piece = text[n * step : n * step + size]
        if n:
            piece = piece.split(" ", 1)[-1]
        taken.append(piece.rsplit(" ", 1)[0])
    return " […]\n\n[…] ".join(taken)


def _options(rewrite: Rewrite | None) -> str:
    if not rewrite or not rewrite.options:
        return ""
    wanted = [OPTIONS.get(option.strip().lower(), option.strip().replace("-", " ")) for option in rewrite.options]
    return "The author also asked for this: " + " ".join(w if w.endswith(".") else w + "." for w in wanted if w)


# --- reading sources ----------------------------------------------------------


# The titles sources.py gives text it could not find chapters in: "Part 3", "Opening".
_PLACEHOLDER_TITLE = re.compile(r"(?:part\s+\d+|opening|\(?untitled\)?)", re.IGNORECASE)


def placeholder_title(title: str | None) -> bool:
    """Whether a chapter's title is one Talari made up rather than one the source has."""
    return not (title or "").strip() or bool(_PLACEHOLDER_TITLE.fullmatch((title or "").strip()))


def analyze(
    request: AnalyzeRequest, chapters: Sequence[tuple[SourceChapterText, str]], *, overview: bool
) -> tuple[str, str]:
    """Read sampled chapters: with ``overview``, the whole analysis; without, chapter summaries only."""
    what = "a book" if request.kind == "book" else "a document, or a set of documents"
    titled = f", titled “{request.title.strip()}”" if request.title.strip() else ""
    body = "\n\n".join(f"=== [{chapter.index}] {chapter.title or '(untitled)'} ===\n{text}" for chapter, text in chapters)
    chapter_key = (
        '- "chapters": one object per chapter above, in the same order, each with "index" (the number in '
        'square brackets, copied exactly) and "summary" (one to three sentences saying concretely what that '
        "chapter covers or what happens in it)."
    )
    if any(placeholder_title(chapter.title) for chapter, _ in chapters):
        chapter_key += (
            ' A chapter whose title above is only a placeholder ("Part 3", "Opening" or "(untitled)") also '
            'gets "title": a real title for it, 2 to 8 words, made from the headings or lesson names its text '
            "shows where it has any, otherwise naming what it covers. No numbering, no \"Part\", no quotation marks. "
            "Its summary does not mention the placeholder."
        )
    if overview:
        keys = f"""- "summary": three to five sentences on what the material is, what it covers and how it is organised.
- "themes": 3 to 8 short phrases naming its main themes.
- "concepts": the 5 to 15 most important ideas, people, places, methods or events in it, each an object with "name", "description" (one sentence, in terms of this material) and "chapterIndexes" (the numbers in square brackets of the chapters where it matters most).
- "audience": one sentence on who it is written for.
- "tone": a few words describing its voice, such as "warm, conversational, practical".
{chapter_key}"""
    else:
        keys = chapter_key
    user = f"""The author has imported {what}{titled}, to make a new book from. Below is each chapter in order: its number in square brackets, its title, then its text. A long chapter is given as its opening and a passage from its middle, with […] where text was left out.

{body}

Return a JSON object with these keys:
{keys}

Base everything on the text given. Where a chapter is only sampled, summarise what the sample shows rather than guessing at what was left out."""
    return ROLE, user


def overview(request: AnalyzeRequest, summaries: Sequence[tuple[int, str, str]]) -> tuple[str, str]:
    """The whole-work analysis from chapter summaries, for material too long to read at once."""
    what = "a book" if request.kind == "book" else "a document, or a set of documents"
    titled = f", titled “{request.title.strip()}”" if request.title.strip() else ""
    body = "\n".join(f"[{index}] {title or '(untitled)'}: {summary}" for index, title, summary in summaries)
    user = f"""The author has imported {what}{titled}, to make a new book from. It is long, so here is a summary of each chapter in order, with the chapter's number in square brackets:

{body}

Return a JSON object with these keys:
- "summary": three to five sentences on what the material is, what it covers and how it is organised.
- "themes": 3 to 8 short phrases naming its main themes.
- "concepts": the 5 to 15 most important ideas, people, places, methods or events in it, each an object with "name", "description" (one sentence, in terms of this material) and "chapterIndexes" (the numbers in square brackets of the chapters where it matters most).
- "audience": one sentence on who it is written for.
- "tone": a few words describing its voice, such as "warm, conversational, practical"."""
    return ROLE, user


# --- planning -----------------------------------------------------------------


def _brief(request: BlueprintRequest) -> str:
    brief = request.brief
    return _lines(
        ("The idea", brief.idea),
        ("Genre", brief.genre),
        ("Written for", brief.audience),
        ("Tone", brief.tone),
        ("Approach", brief.approach),
        ("The author's instructions", brief.instructions),
    ) or "(The author gave no further details.)"


def source_index(sources: Sequence[BlueprintSource], labels: Sequence[str], limit: int) -> str:
    """The sources as the model plans from them. Each chapter carries a label such as
    [S1-3] (source 1, chapter index 3) for the plan to cite; ids from the app could be
    anything, and a model copies a short label far more reliably than a UUID."""
    blocks = []
    for label, source in zip(labels, sources):
        lines = [f"SOURCE {label}: “{source.title or 'Untitled'}”"]
        if source.summary:
            lines.append(f"Summary: {source.summary.strip()}")
        if source.themes:
            lines.append("Themes: " + "; ".join(t.strip() for t in source.themes if t.strip()))
        if source.concepts:
            lines.append(
                "Key concepts: "
                + "; ".join(f"{c.name.strip()} ({c.description.strip()})" if c.description else c.name.strip() for c in source.concepts)
            )
        if source.chapters:
            lines.append("Chapters:")
            lines += [
                f"[{label}-{c.index}] {c.title or '(untitled)'}{f' ({c.words:,} words)' if c.words else ''}: {c.summary.strip()}"
                for c in source.chapters
            ]
        blocks.append("\n".join(lines))
    text = "\n\n".join(blocks)
    if len(text) > limit:
        # Too long to send whole: keep every chapter line, each summary shortened to fit.
        share = max(80, limit // max(1, sum(len(s.chapters) + 3 for s in sources)))
        text = "\n".join(_clip(line, share) if line.startswith("[") else _clip(line, share * 4) for line in text.split("\n"))
    return text


def _planning_situation(request: BlueprintRequest, sources: str) -> str:
    parts = [MODES[request.mode], f"THE BRIEF\n{_brief(request)}"]
    if request.mode == "rewrite":
        rewrite = request.rewrite or Rewrite()
        parts.append(f"How far the new book may move from the source:\n{DEPTHS[rewrite.depth]}")
    options = _options(request.rewrite)
    if options:
        parts.append(options)
    if sources:
        parts.append(f"THE SOURCES\n{sources}")
    return "\n\n".join(parts)


def _size(request: BlueprintRequest) -> str:
    target = request.brief.target_words if request.mode != "idea" else 0
    if target:
        material = sum(c.words for s in request.sources for c in s.chapters)
        held = f"The sources hold about {material:,} words, so the" if material else "The"
        return (
            f"{held} book should come to about {target:,} words in all: the length its material supports, not "
            f"a length to fill. Give each chapter the room its material needs (each source chapter's length is "
            f"shown beside it), plus only the context a reader needs to follow it. Plan no chapter for which there "
            f"is not enough material, and leave out nothing the book needs."
        )
    chapters, words = LENGTHS[request.brief.length]
    if request.brief.chapter_count:
        return f"The book should have {request.brief.chapter_count} chapters of about {words} words each."
    if request.mode == "rewrite" and (request.rewrite or Rewrite()).depth == "light" and request.sources:
        count = sum(len(s.chapters) for s in request.sources)
        return f"Keep the source's chapters: {count} of them, each about as long as the source's."
    return f"Plan a {request.brief.length} book: about {chapters} chapters of about {words} words each, or what the material truly needs."


def directions(request: BlueprintRequest, sources: str, count: int) -> tuple[str, str]:
    """Distinct directions for the book, before any is planned in full."""
    user = f"""{_planning_situation(request, sources)}

The author will be shown {count} different versions of this book to choose from. Propose them. They must differ in substance, in angle, structure or approach (one chronological and one thematic, say, or one a practical guide and one a narrative), not merely in wording. Each must still be the book the brief asks for.{" With a light rewrite the chapters stay the source's, so let the versions differ in framing, emphasis and how each chapter is presented." if request.mode == "rewrite" and (request.rewrite or Rewrite()).depth == "light" else ""}

Return a JSON object with one key, "directions": an array of {count} objects, each with "title" (a working title) and "angle" (two or three sentences: what sets this version apart, how it is structured, and who it suits best)."""
    return ROLE, user


def blueprint(
    request: BlueprintRequest, sources: str, direction: str | None = None, others: Sequence[str] = ()
) -> tuple[str, str]:
    cites = (
        '  - "sources": the labels in square brackets (such as "S1-3") of the source chapters this chapter draws on; an empty array if none.\n'
        if request.mode != "idea"
        else ""
    )
    steer = ""
    if direction:
        steer = f"\n\nPlan this version of the book:\n{direction}"
        if others:
            steer += "\n\nThe author is also being offered these versions, so keep this one clearly distinct from them:\n" + "\n".join(f"- {o}" for o in others)
    user = f"""{_planning_situation(request, sources)}{steer}

{_size(request)}

Plan the book. Return a JSON object with these keys:
- "title": a title a reader would pick up, specific to this book rather than a restatement of the request.
- "subtitle": a subtitle that says what the book offers, or "" if it needs none.
- "description": two or three sentences in the voice of a back-cover blurb.
- "audience": one sentence on who the book is for.
- "tone": a few words describing its voice.
- "chapters": an array in reading order, each an object with:
  - "title": the chapter's name only, without "Chapter 1" numbering, which the layout adds.
  - "summary": one to three sentences saying concretely what the chapter covers or what happens in it, precise enough to write it from.
  - "sections": two to five objects, each with "title" and "topics" (two to five short phrases naming what that section covers).
  - "estimatedWords": the chapter's planned length in words.
{cites}
Together the chapters should build: an opening that earns attention, a middle that develops, an ending that lands."""
    return ROLE, user


def _plan_text(plan: Blueprint) -> str:
    lines = [
        _lines(
            ("Title", plan.title),
            ("Subtitle", plan.subtitle),
            ("Description", plan.description),
            ("Audience", plan.audience),
            ("Tone", plan.tone),
        ),
        "Chapters:",
    ]
    for number, chapter in enumerate(plan.chapters, start=1):
        words = f" (about {chapter.estimated_words} words)" if chapter.estimated_words else ""
        lines.append(f"{number}. {chapter.title}{words}")
        if chapter.summary:
            lines.append(f"   Summary: {chapter.summary}")
        if chapter.sections:
            lines.append(
                "   Sections: "
                + "; ".join(f"{s.title} ({', '.join(s.topics)})" if s.topics else s.title for s in chapter.sections)
            )
    return "\n".join(lines)


def refine(plan: Blueprint, message: str, history: Sequence[ChatTurn]) -> tuple[str, str]:
    conversation = ""
    if history:
        turns = "\n".join(f"{'Author' if t.role == 'user' else 'You'}: {t.content.strip()}" for t in history)
        conversation = f"\n\nTHE CONVERSATION SO FAR\n{turns}"
    user = f"""The author is shaping the plan for their book with you.

THE PLAN, AS IT STANDS
{_plan_text(plan)}{conversation}

The author now says:
{message.strip()}

Return a JSON object with these keys:
- "reply": one or two sentences to the author saying what you changed, or answering them if they asked a question.
- "blueprint": the complete revised plan, with "title", "subtitle", "description", "audience", "tone" and "chapters". "chapters" is an array in reading order, each an object with "from", "title", "summary", "sections" (each with "title" and "topics") and "estimatedWords". "from" is the number of the chapter above that this one continues, even if you retitle, rewrite or move it, a list of numbers if it merges several, or null for a brand-new chapter. If the author asked a question and nothing should change, make "blueprint" null.

When the author says "chapter 3" they mean the one numbered 3 above. Change what they asked for and keep everything else exactly as it is. Chapter titles carry no "Chapter 1" numbering."""
    return ROLE, user


# --- writing ------------------------------------------------------------------


def _book_card(request: ChapterRequest) -> str:
    book = request.book
    return _lines(
        ("Title", book.title or "Untitled"),
        ("Subtitle", book.subtitle),
        ("About the book", book.description),
        ("Written for", book.audience),
        ("Tone", book.tone),
        ("The author's instructions for the whole book", book.instructions),
    )


def _outline(request: ChapterRequest) -> str:
    lines = []
    for index, entry in enumerate(request.outline):
        marker = "   <- this chapter" if index == request.chapter_index else ""
        summary = f": {entry.summary.strip()}" if entry.summary.strip() else ""
        lines.append(f"{index + 1}. {entry.title}{summary}{marker}")
    return "\n".join(lines)


def _source_framing(request: ChapterRequest) -> str:
    rewrite = request.rewrite or Rewrite()
    if request.mode == "rewrite":
        framing = DEPTH_WRITING[rewrite.depth]
    elif request.mode == "extract":
        framing = "This chapter draws on the source material below. Use what serves this chapter, in your own words, and leave the rest."
    elif request.mode == "documents":
        framing = (
            "This chapter is built from the documents below. Bring together what they say on this chapter's "
            "subject into one connected text in your own words."
        )
    else:
        framing = "The author supplied the reference material below for this chapter. Use it where it helps."
    if not (request.mode == "rewrite" and rewrite.depth in ("deep", "full")) and request.mode != "idea":
        framing += " Do not add facts, figures or quotations the material does not support, and do not copy its sentences."
    options = _options(request.rewrite)
    return f"{framing} {options}".strip()


def chapter(request: ChapterRequest, excerpts: str, words: int, pictures: int) -> tuple[str, str]:
    plan = request.chapter
    number = request.chapter_index + 1
    parts = [f"THE BOOK\n{_book_card(request)}"]
    if request.outline:
        parts.append(f"THE TABLE OF CONTENTS\n{_outline(request)}")
    if request.previous_summary.strip() and request.chapter_index > 0:
        parts.append(f"WHERE THE BOOK STANDS (a recap of the chapter before this one)\n{request.previous_summary.strip()}")
    this = [f"Write chapter {number}, “{plan.title}”." + (f" It should cover: {plan.summary.strip()}" if plan.summary.strip() else "")]
    if plan.sections:
        this.append("Its sections, in order:")
        this += [f"- {s.title}" + (f": {'; '.join(s.topics)}" if s.topics else "") for s in plan.sections]
    if plan.instructions.strip():
        this.append(f"The author's direction for this chapter: {plan.instructions.strip()}")
    parts.append("THIS CHAPTER\n" + "\n".join(this))
    if excerpts:
        parts.append(f"SOURCE MATERIAL\n{_source_framing(request)}\n\n{excerpts}")

    continuity = (
        " Pick up where the previous chapter left off, so the book reads as one continuous whole, without retelling it."
        if request.previous_summary.strip() and request.chapter_index > 0
        else ""
    )
    sections = " Use the planned sections as the chapter's sections, in order, each under its own heading." if plan.sections else ""
    # From sources, the length is the material's: a guide, never a quota to pad toward.
    measure = (
        f"Aim for about {words} words, but let the material set the length: when it is covered, stop, rather "
        "than restating it to reach a number; when it needs more to be clear, give it more."
        if request.mode != "idea" and excerpts
        else f"Aim for about {words} words."
    )
    how = [
        f"{measure} Write it to be read: stay inside this chapter's part of the book without "
        f"running ahead into later chapters, and end where the chapter ends rather than summarising it.{continuity}{sections}",
        f"{MARKDOWN} Do not write the chapter title or a \"Chapter {number}\" heading; the layout adds it.",
    ]
    if pictures:
        where = (
            f"Mark {pictures} places for a picture, spread through the chapter, each right after the passage it illustrates"
            if pictures > 1
            else "Mark 1 place for a picture, right after the passage it illustrates"
        )
        how.append(
            f"{where}, with a line of its own in this form: {ILLUSTRATION_LINE}. "
            f"{ILLUSTRATION_BRIEF} The caption is a line for under the picture, at most 15 words."
        )
    how.append(
        f"After the chapter, on a line by itself, write {SUMMARY_MARKER}, then a recap of about 120 words of "
        "this chapter, written for whoever writes the next one: where things stand at its end, what was "
        "established or explained, and any thread left open. The recap is not part of the chapter."
    )
    parts.append("HOW TO WRITE IT\n" + "\n\n".join(how))
    return ROLE, "\n\n".join(parts)


def recap(title: str, text: str) -> tuple[str, str]:
    user = f"""Here is the chapter “{title}” of a book:

{sample(text, 30000)}

Write a recap of about 120 words of this chapter for whoever writes the next one: where things stand at its end, what was established or explained, and any thread left open. Plain prose, one paragraph, nothing else."""
    return ROLE, user


# --- revising -----------------------------------------------------------------


def revise(
    action: str,
    passage: str,
    *,
    instruction: str,
    tone: str,
    book_title: str,
    chapter_title: str,
    before: str,
    after: str,
) -> tuple[str, str]:
    words = len(passage.split())
    task = REVISIONS[action].format(
        words=words,
        half=max(1, words // 2),
        # A word target works where "half" or "twice" alone gets a timid edit.
        double=max(words * 2, words + 60),
        tone=tone or instruction,
        more=max(150, min(400, words)),
        toward=" toward what comes after it" if after else "",
        instruction=instruction,
    )
    if instruction and action not in ("custom", "tone"):
        task += f" The author adds: {instruction}"
    where = _lines(("Book", book_title), ("Chapter", chapter_title))
    parts = [where] if where else []
    if before:
        parts.append(f"WHAT COMES BEFORE THE PASSAGE (context only; do not rewrite it)\n{_tail(before, 3000)}")
    parts.append(f"THE PASSAGE\n{passage or '(empty: the text so far ends just before it)'}")
    if after:
        parts.append(f"WHAT COMES AFTER IT (context only)\n{_clip(after, 1500)}")
    parts.append(task)
    parts.append(
        "Return only the text, ready to go into the book: paragraphs separated by one blank line, "
        "*italics* and **bold** where the text calls for them, no title and no commentary. Then, only if "
        f"there is something the author should check or know, write {NOTES_MARKER} on a line by itself, "
        'followed by one short note per line, each beginning with "- ". A note is about the text — a fact '
        "to check, a name to confirm, a change of meaning the author should know about — never about these "
        "instructions or how closely you followed them (word counts, length, the request itself)."
    )
    return ROLE + " Here you are editing one passage of the author's book.", "\n\n".join(parts)


def factcheck(passage: str, book_title: str, chapter_title: str) -> tuple[str, str]:
    where = _lines(("Book", book_title), ("Chapter", chapter_title))
    user = f"""{where + chr(10) + chr(10) if where else ""}THE PASSAGE
{passage}

You cannot look anything up, and you must not judge whether anything in the passage is true. Your job is to point the author at what to check. List the specific factual claims in the passage that a careful author should check against a reliable source before publishing: names, dates, numbers and statistics, quotations and who said them, historical events, and scientific, medical, legal or technical statements. Each note is one sentence naming the claim and what to check, such as "Check the year given for the library's founding (1602) against a reliable source." Never say whether a claim is right or wrong, and never call anything verified, confirmed, accurate or correct. Opinions, the author's own experiences and plainly fictional events are not claims to check.

Return a JSON object with one key, "claims": an array of these notes, or an empty array if the passage makes no claims worth checking."""
    return ROLE, user


# --- illustration -------------------------------------------------------------


# --- a picture made by a conversation -----------------------------------------------------

# What each kind of picture is, to the model.
IMAGE_KINDS = {
    ("cover", "background"): "the art filling the book's front cover, behind its title and author's name, which are set in type over it separately",
    ("back-cover", "background"): "the art filling the book's back cover, behind the blurb, which is set in type over it separately",
    ("interior", "background"): "a picture filling a whole page of the book, behind the page's text",
    ("cover", "media"): "a picture placed on the book's front cover, over its background",
    ("back-cover", "media"): "a picture placed on the book's back cover",
    ("interior", "media"): "a picture placed on a page of the book, beside or over its text",
}

# The kinds of background design — design categories, not looks: what each one is, and what
# the follow-up questions for it are about.
DESIGN_KINDS = {
    "border": "a decorative border or frame around the edges, the middle left open",
    "corners": "ornaments in the corners only, the rest of the page open",
    "texture": "a surface or material across the whole page (paper, parchment, linen, stone, leather…), with no objects",
    "pattern": "a repeating pattern of small motifs across the page",
    "minimal": "a minimal ground: a soft gradient or a few flat fields of color",
    "scene": "a full illustrated scene, edge to edge",
    "emblem": "a single central emblem or medallion on a plain or textured ground",
    "abstract": "abstract shapes, strokes and color, nothing recognizable",
}
DESIGN_QUESTIONS = {
    "border": "the border's motifs, how wide and how ornate it is, what fills the middle",
    "corners": "the corner motifs, how far in they reach, all four corners or two",
    "texture": "the material, its age and wear, its tone, how strong the grain is",
    "pattern": "the motif, its scale and spacing, how much it stands out",
    "minimal": "the colors, the direction and softness of the blend, any single accent",
    "scene": "what it shows, the setting, the light and mood, the viewpoint",
    "emblem": "the emblem's motif, its size, its framing, the ground behind it",
    "abstract": "the shapes and their movement, how busy it is",
}

# What a background is (the app's model.js BACKGROUND_TYPES and BACKGROUND_TEXTURES, by
# id): its type, then its texture — set choices the author picks, never written here.
BACKGROUND_TYPES = {
    "atmospheric": "light and mood more than things: mist, haze, glow, smoke, soft depths of shadow",
    "scenic": "a place seen whole: a landscape, a sky, an interior",
    "abstract": "shapes, strokes and color, nothing recognizable",
    "geometric": "geometric shapes, lines and repeating forms",
    "decorative": "ornament: borders, flourishes, motifs, filigree",
    "material": "a surface or material across the whole page, with no objects",
}
BACKGROUND_TEXTURES = {
    "smooth": "smooth and even, with no grain",
    "paper": "paper and parchment",
    "canvas": "canvas and woven fabric",
    "leather": "leather",
    "wood": "wood and its grain",
    "stone": "stone and marble",
    "metallic": "metal with a metallic sheen",
    "painted": "a painted, brushed surface",
    "distressed": "distressed: aged, worn and scratched",
    "organic": "organic matter: leaves, moss, bark, petals",
}

GUIDE_ROLE = (
    "You help an author describe one picture for their book, by asking them questions one at a time, like "
    "a good art director: plain, friendly and brief. The author never sees an image prompt and never needs "
    "to know how images are made."
)


def _image_card(request: ImageBrief) -> str:
    book = request.book
    kind = IMAGE_KINDS[(request.purpose, request.mode)]
    parts = [f"THE PICTURE\nIt is {kind}." + (f" The page is {request.size}." if request.size else "")]
    # A background is made from what the author says of it, not from the book: the app sends
    # no book for one, and then there is none here to draw on.
    if book.title or book.description or book.tone:
        card = _lines(("Title", book.title or "Untitled"), ("About the book", book.description), ("Tone", book.tone))
        parts.insert(0, f"THE BOOK\n{card}")
    background = [
        f"Type: {BACKGROUND_TYPES[request.background_type]}." if request.background_type in BACKGROUND_TYPES else "",
        f"Texture: {BACKGROUND_TEXTURES[request.texture]}." if request.texture in BACKGROUND_TEXTURES else "",
    ]
    if request.mode == "background" and any(background):
        parts.append("THE BACKGROUND THE AUTHOR CHOSE\n" + "\n".join(x for x in background if x))
    if request.look:
        parts.append(f"It will be drawn as {request.look}. That is decided: never ask about art style, medium, technique or artist.")
    if request.designs:
        lines = "\n".join(
            f"- {d.title}: {d.description} ({', '.join(DESIGN_KINDS.get(k, k) for k in d.kinds)})" for d in request.designs
        )
        parts.append(f"THE BACKGROUND DESIGN THE AUTHOR CHOSE\n{lines}")
    if request.page_text.strip():
        parts.append(f"THE PAGE IT IS FOR\n{_clip(request.page_text, 3000)}")
    return "\n\n".join(parts)


def _fits(request: ImageBrief) -> str:
    return "this background" if request.mode == "background" and not (request.book.title or request.book.description) else "the book"


def _image_answers(request: ImageBrief) -> str:
    if not request.answers:
        return "Nothing asked yet."
    lines = []
    for a in request.answers:
        said = f"(Let Talari decide: choose what suits {_fits(request)} best.)" if a.decide else a.answer
        lines.append(f"- [{a.id}] {a.question}\n  {said}")
    return "\n".join(lines)


def _design_needs(request: ImageBrief) -> str:
    kinds = list(dict.fromkeys(k for d in request.designs for k in d.kinds if k in DESIGN_QUESTIONS))
    return "; ".join(DESIGN_QUESTIONS[k] for k in kinds)


def image_guide(request: ImageBrief, remaining: int) -> tuple[str, str]:
    cover = request.mode == "background" and request.purpose in ("cover", "back-cover")
    chosen = " Never ask about the type of background or its texture: the author chose them." if request.mode == "background" and (request.background_type or request.texture) else ""
    words = ", where the title and words will sit, so that part stays calm" if cover else ""
    if request.mode == "background" and request.designs:
        # A chosen design: the questions are about what it needs, not about a subject.
        order = f"what this design still needs: {_design_needs(request)}{words}, then the colors and the atmosphere"
    elif request.mode == "background" and (request.background_type or request.texture):
        # Its type and texture picked: the questions are about what they still leave open.
        order = (f"what this background still needs for its type and texture — for a scenic one what it shows, for a decorative one its "
                 f"motifs, for a material or a texture its wear and grain{words} — then the mood and light, then the colors")
    elif request.mode == "background":
        order = f"the kind of background (a decorative border, corner ornaments, a texture, a pattern, an emblem, a minimal ground or a full scene) and what it shows{words}, the mood and light, the colors"
    else:
        order = f"what the picture shows (its subject){words}, the setting, the mood and light, the composition or viewpoint, the colors"
    user = f"""{_image_card(request)}

ASKED SO FAR
{_image_answers(request)}

Decide whether one more question would make this picture clearly better. Up to {remaining} more may be asked, but ask fewer when the picture is clear enough: every question costs the author a moment.

If one is worth asking, ask the one that matters most of what is still open, usually in this order: {order}. Never ask again about anything already asked, decided or answered, nor about what the book or the page already makes plain, and never about the look, the size or shape of the page, or what the picture is for: those are decided.{chosen} Ask in under 15 words, in everyday language, with no jargon (never "prompt", "render", "aspect ratio", "composition", "seed", "style").

Give 3 or 4 suggested answers that fit {_fits(request)}, each a concrete choice in 2 to 8 words, different from one another.

Also write "summary": the picture as it now stands, in one or two plain sentences for the author, beginning "A" or "An" (e.g. "A misty forest at dawn, a small girl in a red coat at the edge of it."). Where the author let Talari decide, choose something fitting and include it. Never mention the art style, prompts or anything technical.

Return a JSON object: {{"question": {{"id": "a short-kebab-case name for what it asks about", "text": "the question", "suggestions": ["…", "…", "…"]}} or null when nothing more is worth asking, "summary": "…"}}"""
    return GUIDE_ROLE, user


def image_designs(request: ImageDesignsRequest) -> tuple[str, str]:
    cover = request.purpose in ("cover", "back-cover")
    kinds = "\n".join(f"- {k}: {v}" for k, v in DESIGN_KINDS.items())
    avoid = ("\n\nAlready suggested, so suggest none of these again or anything close to them: " + "; ".join(request.exclude)) if request.exclude else ""
    room = " Every design leaves a calm area for the title and the author's name." if cover else ""
    user = f"""{_image_card(request)}

Recommend {request.count} background designs for it, made for this book: its title and subject, and the look it is drawn in — an Art Nouveau watercolor and a minimalist design should get very different suggestions. A background is often a decorative design rather than a picture of the book's subject: borders, corner ornaments, textures, patterns, emblems. A full scene can be one of them.{room}

The kinds of design:
{kinds}

Make them varied: at least three different kinds among them, never two alike, and at least two that are not a picture of the subject. A design may combine kinds (a textured parchment with corner ornaments).{avoid}

Return a JSON object: {{"designs": [{{"title": "the design in 3 to 7 words", "description": "one plain sentence for the author: what it looks like", "kinds": ["one to three of: border, corners, texture, pattern, minimal, scene, emblem, abstract"], "palette": ["three colors as #rrggbb that suit it and the look"]}}]}}"""
    return ROLE, user


# A remix's picture, as far as anyone has said: its description, or nothing — and then it is
# never guessed: the image model starts from the picture itself. A guess pulls the drawing
# away from it (an undescribed photo of a dragon was remixed as "a figure in a field").
def _remix_subject(request: ImageComposeRequest | RemixQuestionsRequest) -> str:
    known = request.previous.strip()
    if known:
        return f"WHAT THE PICTURE SHOWS (as it was described)\n{known}"
    return "WHAT THE PICTURE SHOWS\nNot described. Never name or guess what it shows: the image model starts from the picture itself."


def _clarified(request: ImageComposeRequest) -> str:
    asked = "".join(
        f"\n- {a.question}\n  {a.answer if a.answer and not a.decide else '(Let Talari decide.)'}"
        for a in request.clarifications if a.question
    )
    return f"\n\nWHAT THEY SAID WHEN ASKED ABOUT IT{asked}" if asked else ""


def _remix_compose(request: ImageComposeRequest) -> tuple[str, str]:
    """A remix's prompt: the change, not the picture. The image model already has the
    picture, so the prompt says what it becomes, what of it is kept, and what the change needs,
    in the remix's look."""
    look = request.look or "the book's own art style"
    user = f"""THE PICTURE
It is redrawn from itself: the image model starts from the picture (image to image) and changes it as the author asks. It will be drawn as {look}.

{_remix_subject(request)}

THE AUTHOR'S CHANGE
{request.instruction.strip() or "Redraw it in its look, otherwise unchanged."}{_clarified(request)}

Write the prompt for that image model, in one paragraph of at most 90 words. It already has the picture, so write the change, not the picture:
- Begin with what the picture becomes: "Transform this image into …" or "Make …".
- Say to preserve the original's subject, pose, anatomy, proportions, composition, colors and distinctive features, except what the change alters.
- Then what the change needs, concretely: surfaces, textures, materials, light, detail.
- Name the subject only as the description above names it; never guess it.
- Make the change within {look}: "more real" means more lifelike forms, textures and light as that look can show them — never a photograph, a 3D render or another art style. Leave out art style, medium and artist names: the look's own words are added after the prompt.
- Where the author let Talari decide, choose what suits the change best.
- Never include words, letters, titles, signs or typography in the picture.

Return a JSON object with:
- "prompt": that prompt.
- "caption": a short name for the remix, at most 6 words, for the author's list of pictures: its subject where the description names it, else the change ("More lifelike").
- "summary": the change in one or two plain sentences for the author, with nothing technical.
- "shape": "square" (a remix keeps the picture's own shape)."""
    return ROLE, user


def image_compose(request: ImageComposeRequest) -> tuple[str, str]:
    if request.remix:
        return _remix_compose(request)
    cover = request.purpose in ("cover", "back-cover") and request.mode == "background"
    rules = [
        "Describe what is seen, concretely: the subject, what it is doing, the setting, the light, the mood, the viewpoint and the colors.",
        "Leave out any art style, medium or artist: the look is applied separately.",
        "Never include words, letters, titles, signs or typography in the picture.",
    ]
    if cover:
        rules.append("Keep the area where the words will sit calm and uncluttered, as the answers say (else the upper third).")
    kinds = {k for d in request.designs for k in d.kinds}
    if request.mode == "background":
        rules.append("It fills the whole page edge to edge, with nothing important at the very edges, which are trimmed.")
        if request.background_type in BACKGROUND_TYPES:
            rules.append(f"It is {BACKGROUND_TYPES[request.background_type]}.")
            if request.background_type in ("atmospheric", "abstract", "geometric", "material"):
                rules.append("No figures, creatures or story scenes in it.")
        if request.texture in BACKGROUND_TEXTURES:
            rules.append(f"Its surface is {BACKGROUND_TEXTURES[request.texture]}.")
        for kind in sorted(kinds):
            if kind in DESIGN_KINDS:
                rules.append(f"It is {DESIGN_KINDS[kind]}.")
        if kinds & {"border", "corners", "emblem"}:
            rules.append("Keep the open part of the page plain and calm, a flat or softly textured ground, so words can sit on it.")
        if kinds and not kinds & {"scene", "emblem"}:
            rules.append("No figures, creatures or scenery beyond the decoration itself.")
    change = ""
    if request.previous.strip():
        change = f"""

THE PICTURE AS IT WAS DESCRIBED
{request.previous.strip()}

THE AUTHOR'S CHANGE
{request.instruction.strip() or "Draw it again, the same picture."}{_clarified(request)}

Rewrite the description with this change made, keeping everything else as it was."""
        if request.look:
            rules.append(
                f"The picture stays drawn as {request.look}: make the change within that look. \"More real\" means more "
                "lifelike forms, anatomy, light and texture as that look can show them — never a photograph, a 3D render "
                "or another art style."
            )
    user = f"""{_image_card(request)}

THE AUTHOR'S ANSWERS
{_image_answers(request)}{change}

Write the description an illustrator (an image model) will draw from, in one paragraph of at most 90 words. Where the author let Talari decide, choose what suits {_fits(request)} best.
{chr(10).join("- " + r for r in rules)}

Return a JSON object with:
- "prompt": that description.
- "caption": a short name for the picture, at most 6 words, for the author's list of pictures.
- "summary": the picture in one or two plain sentences for the author, beginning "A" or "An", with nothing technical.
- "shape": "portrait", "square" or "landscape" — the shape that suits the picture best."""
    return ROLE, user


def remix_questions(request: RemixQuestionsRequest, most: int) -> tuple[str, str]:
    look = request.look or "the book's own art style"
    user = f"""{_image_card(request)}

{_remix_subject(request)}

THE AUTHOR'S CHANGE
{request.instruction.strip()}

The author is changing this picture. Ask only what their words leave open that would change the result a lot — how far to go, which part, what kind. For "make the dragon look more real": how realistic. At most {most} questions, the most important first; none when the change is already clear.
- Each question in plain words, at most 12 words.
- Each with three or four answers to pick, at most 6 words each, from the smallest change to the largest.
- The picture stays drawn as {look}: the answers are degrees and kinds of the change within that look — never a photograph, a 3D render or another art style.
- Never ask about the art style, medium, size or shape, or about anything the change already says. Nothing technical.
- Name what the picture shows only as the description above names it; otherwise ask in general terms ("How realistic should it look?").

Return a JSON object: {{"questions": [{{"id": "a-short-name", "text": "…", "suggestions": ["…", "…", "…"]}}]}}, with an empty list when nothing needs asking."""
    return ROLE, user


def background_choices(request: BackgroundChoicesRequest) -> tuple[str, str]:
    types = "\n".join(f"- {c.id}: {c.label}" for c in request.types)
    textures = "\n".join(f"- {c.id}: {c.label}" for c in request.textures)
    user = f"""An author described, in their own words, the background they want for a page:
"{request.words.strip()}"

Which of these types of background fit what they said? One or two, best first:
{types}

Which of these textures fit? Up to three, best first, or none when what they said needs no texture:
{textures}

Return a JSON object: {{"types": ["id", …], "textures": ["id", …]}}, using only the ids above."""
    return ROLE, user


def illustration_ideas(request: IllustrationPromptsRequest, style: str, text: str, count: int) -> tuple[str, str]:
    book = request.book
    card = _lines(("Title", book.title or "Untitled"), ("About the book", book.description), ("Tone", book.tone))
    plural = count > 1
    style_line = (
        f"The pictures will be drawn in this style: {style}. Do not describe the style, medium or artist "
        "in your descriptions; it is applied separately, but let the subjects suit it."
        if style
        else "Do not describe an art style, medium or artist; the book's style is applied separately."
    )
    if request.purpose == "cover":
        opening = f"\n\nFrom the book:\n{text}" if text else ""
        user = f"""THE BOOK
{card}{opening}

{style_line}

Propose {count} different concept{'s' if plural else ''} for the art on this book's cover. Each describes the picture only, in one or two sentences: the cover's title and the author's name are set in type over the art separately, so never mention the title, words, lettering or typography. Give it one strong focal point that reads at thumbnail size, and leave a calm, uncluttered area in the upper third for the title. Capture what the book is about and how it feels.

Return a JSON object with one key, "prompts": an array of {count} object{'s' if plural else ''}, each with "prompt" (the picture) and "caption" (a short name for the concept, at most 8 words)."""
        return ROLE, user
    title = f"“{request.chapter.title}”" if request.chapter.title else "of the book"
    user = f"""THE BOOK
{card}

{style_line}

THE CHAPTER {title}
{text}

Choose {count} moment{'s' if plural else ''} from this chapter that would make the strongest illustration{'s' if plural else ''}{', spread through the chapter rather than bunched together' if plural else ''}. If the chapter is a list or an index rather than prose, show instead an emblem or scene that stands for the book as a whole. {ILLUSTRATION_BRIEF}

Return a JSON object with one key, "prompts": an array of {count} object{'s' if plural else ''}, each with:
- "prompt": the picture, as above.
- "caption": a caption for under the picture, at most 15 words.
- "anchor": a quotation of 5 to 15 words, copied exactly from the chapter text above, from the passage the picture illustrates."""
    return ROLE, user
