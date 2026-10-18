"""The writing operations behind Talari's text endpoints.

Each one builds its prompt (prompts.py), asks DeepSeek (llm.py) and turns the reply into
the response the app expects. The model's output is never trusted for shape: every field
is checked and normalized here, indexes and references are matched against what the app
sent, and text bound for the page goes through markup.py so only escaped text, <em> and
<strong> reach it.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Iterable
from typing import Any, TypeVar

from . import llm, prompts
from .errors import ServiceError
from .markup import as_markdown, as_text, blocks_as_html, parse_blocks, plain
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    Block,
    Blueprint,
    BlueprintChapter,
    BlueprintRequest,
    BlueprintResponse,
    ChapterRequest,
    ChapterResponse,
    ChapterSummary,
    Concept,
    IllustrationIdea,
    IllustrationPromptsRequest,
    IllustrationPromptsResponse,
    PlacedIllustration,
    RefineRequest,
    RefineResponse,
    ReviseRequest,
    ReviseResponse,
    Rewrite,
    Section,
    SourceChapterText,
    SourceRef,
)
from .styles import style_text

logger = logging.getLogger("talari")

T = TypeVar("T")

# Calls made at once for one request (chapter batches, blueprint versions).
CONCURRENCY = 4

# Reading sources: ~80k characters of chapter text per call (about 20k tokens), shared out
# so short chapters go whole and long ones are sampled; at most this many chapters a call,
# since every chapter's summary has to fit in the reply.
ANALYZE_BUDGET = 80_000
ANALYZE_BATCH = 40
OVERVIEW_BUDGET = 80_000

SOURCES_BUDGET = 60_000
EXCERPTS_BUDGET = 60_000
MIN_CHAPTER_WORDS, MAX_CHAPTER_WORDS = 300, 9_000
# A chapter's reply: about 1.4 tokens a word of prose, plus the recap and picture lines.
TOKENS_PER_WORD = 1.6
ILLUSTRATION_TEXT_BUDGET = 14_000
REFINE_HISTORY_TURNS = 12

TEMPERATURE = {
    "analyze": 0.3,
    "directions": 1.2,
    "blueprint": 0.9,
    "refine": 0.7,
    "chapter": 1.2,
    "chapter_faithful": 0.8,
    "recap": 0.3,
    "illustrations": 1.0,
    "factcheck": 0.0,
}
REVISE_TEMPERATURE = {
    "rewrite": 1.0,
    "expand": 1.0,
    "shorten": 0.5,
    "simplify": 0.5,
    "tone": 1.0,
    "continue": 1.2,
    "flow": 0.6,
    "examples": 1.0,
    "custom": 0.8,
}


def _marker(marker: str) -> re.Pattern[str]:
    """Match a marker line even if the model varies its spacing or decoration. The "="s
    are required: a chapter may well have a section headed "## Summary" of its own."""
    words = marker.strip("=").strip()
    return re.compile(rf"^[ \t*_]*={{2,}}\s*{re.escape(words)}\s*={{2,}}[ \t*_:]*$", re.IGNORECASE | re.MULTILINE)


SUMMARY = _marker(prompts.SUMMARY_MARKER)
NOTES = _marker(prompts.NOTES_MARKER)
PICTURE = re.compile(
    r"^[ \t>*_]*\[{1,2}\s*(?:ILLUSTRATION|PICTURE|IMAGE)\s*[:|]\s*(.+?)\s*\]{1,2}[ \t*_]*$",
    re.IGNORECASE | re.MULTILINE,
)
# The same mark left at the end of a paragraph rather than on a line of its own.
PICTURE_INLINE = re.compile(r"\[\[\s*(?:ILLUSTRATION|PICTURE|IMAGE)\s*[:|]\s*(.+?)\s*\]\]", re.IGNORECASE)
PLACEHOLDER = "@@picture{}@@"
PLACEHOLDER_RE = re.compile(r"^@@picture(\d+)@@$")
# A recap or note that claims a check this service cannot make.
CLAIMS_VERIFIED = re.compile(r"\b(?:verified|confirmed|is (?:correct|accurate|true)|fact-checked)\b", re.IGNORECASE)
QUOTED = re.compile(r"[\"“”][^\"“”]{0,200}[\"“”]")
ENDS_SENTENCE = re.compile(r"[.!?…:;\"”’')\]]$")


def _text(value: Any) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _strings(value: Any, limit: int = 20) -> list[str]:
    return [s for s in (_text(v) for v in (value if isinstance(value, list) else [])) if s][:limit]


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


async def _gather(jobs: Iterable[Awaitable[T]]) -> list[T]:
    gate = asyncio.Semaphore(CONCURRENCY)

    async def run(job: Awaitable[T]) -> T:
        async with gate:
            return await job

    return await asyncio.gather(*(run(job) for job in jobs))


async def _gather_some(jobs: Iterable[Awaitable[T]]) -> list[T]:
    """Like _gather, keeping what succeeded; raises the first failure only if all failed."""
    results = await _gather(_settled(job) for job in jobs)
    kept = [r for r in results if not isinstance(r, ServiceError)]
    if not kept and results:
        raise results[0]
    return kept


async def _settled(job: Awaitable[T]) -> T | ServiceError:
    try:
        return await job
    except ServiceError as error:
        logger.warning("Talari: one of several calls failed: %s", error.message)
        return error


def _share(texts: list[str], budget: int) -> list[int]:
    """Split a character budget across texts: short ones whole, the rest evenly."""
    limits = [0] * len(texts)
    remaining, left = budget, len(texts)
    for i in sorted(range(len(texts)), key=lambda i: len(texts[i])):
        take = min(len(texts[i]), remaining // max(1, left))
        limits[i] = take
        remaining -= take
        left -= 1
    return limits


# --- reading sources ----------------------------------------------------------


def _chapter_summaries(data: dict[str, Any], indexes: set[int]) -> dict[int, str]:
    found: dict[int, str] = {}
    for entry in data.get("chapters") if isinstance(data.get("chapters"), list) else []:
        if isinstance(entry, dict):
            index = _int(entry.get("index"))
            summary = _text(entry.get("summary"))
            if index in indexes and summary:
                found[index] = summary
    return found


TITLE_MAX_CHARS = 80


def _chapter_titles(data: dict[str, Any], placeholders: set[int]) -> dict[int, str]:
    """The titles the model gave chapters that only had a placeholder ("Part 3")."""
    found: dict[int, str] = {}
    for entry in data.get("chapters") if isinstance(data.get("chapters"), list) else []:
        if not isinstance(entry, dict):
            continue
        index = _int(entry.get("index"))
        title = " ".join(_text(entry.get("title")).strip("\"'“”‘’ ").split())
        if index in placeholders and title and len(title) <= TITLE_MAX_CHARS and not prompts.placeholder_title(title):
            found[index] = title
    return found


async def _summarize(request: AnalyzeRequest, chapters: list[SourceChapterText], *, overview: bool) -> dict[str, Any]:
    texts = [c.text for c in chapters]
    sampled = [(c, prompts.sample(c.text, limit)) for c, limit in zip(chapters, _share(texts, ANALYZE_BUDGET))]
    system, user = prompts.analyze(request, sampled, overview=overview)
    budget = 120 * len(chapters) + (1800 if overview else 400)
    return await llm.ask_json(system, user, max_tokens=min(budget, 8000), temperature=TEMPERATURE["analyze"])


def _opening(text: str, words: int = 40) -> str:
    """A chapter's first words: the stand-in when no summary could be had for it."""
    taken = text.split()[:words]
    return " ".join(taken) + (" …" if len(text.split()) > words else "")


async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    chapters = request.chapters
    indexes = {c.index for c in chapters}
    # Text the source gave no chapters is cut into "Part 1", "Part 2"…; the model that
    # summarises each part names it too, so the overview lists real titles.
    placeholders = {c.index for c in chapters if prompts.placeholder_title(c.title)}
    titles: dict[int, str] = {}
    batches = [chapters[i : i + ANALYZE_BATCH] for i in range(0, len(chapters), ANALYZE_BATCH)]
    if len(batches) == 1:
        overview = await _summarize(request, chapters, overview=True)
        summaries = _chapter_summaries(overview, indexes)
        titles.update(_chapter_titles(overview, placeholders))
    else:
        summaries = {}
        for data in await _gather(_summarize(request, batch, overview=False) for batch in batches):
            summaries.update(_chapter_summaries(data, indexes))
            titles.update(_chapter_titles(data, placeholders))

    missing = [c for c in chapters if c.index not in summaries]
    if missing:
        # One more try for the chapters the reply skipped; a model asked for forty
        # summaries will sometimes stop at thirty-eight. Failing that, a chapter shows its
        # opening words rather than failing the whole analysis.
        try:
            for batch in [missing[i : i + ANALYZE_BATCH] for i in range(0, len(missing), ANALYZE_BATCH)]:
                data = await _summarize(request, batch, overview=False)
                summaries.update(_chapter_summaries(data, indexes))
                titles.update(_chapter_titles(data, placeholders))
        except ServiceError as error:
            logger.warning("Talari: %d chapters left unsummarized: %s", len(missing), error.message)

    if len(batches) > 1:
        lines = [(c.index, titles.get(c.index, c.title), summaries.get(c.index, "")) for c in chapters]
        limits = _share([s for _, _, s in lines], OVERVIEW_BUDGET)
        clipped = [(i, t, prompts.sample(s, limit)) for (i, t, s), limit in zip(lines, limits)]
        system, user = prompts.overview(request, clipped)
        overview = await llm.ask_json(system, user, max_tokens=3000, temperature=TEMPERATURE["analyze"])

    concepts = []
    for entry in overview.get("concepts") if isinstance(overview.get("concepts"), list) else []:
        if isinstance(entry, dict) and _text(entry.get("name")):
            where = entry.get("chapterIndexes")
            concepts.append(
                Concept(
                    name=_text(entry["name"]),
                    description=_text(entry.get("description")),
                    chapter_indexes=sorted({i for i in map(_int, where if isinstance(where, list) else []) if i in indexes}),
                )
            )
    summary = _text(overview.get("summary"))
    if not summary:
        raise ServiceError("DeepSeek returned an analysis with no summary.")
    return AnalyzeResponse(
        summary=summary,
        themes=_strings(overview.get("themes"), 12),
        concepts=concepts[:20],
        audience=_text(overview.get("audience")),
        tone=_text(overview.get("tone")),
        chapters=[
            ChapterSummary(
                index=c.index, title=titles.get(c.index, c.title), summary=summaries.get(c.index) or _opening(c.text)
            )
            for c in chapters
        ],
    )


# --- planning -----------------------------------------------------------------


def _labels(request_sources: list[Any]) -> list[str]:
    return [f"S{n}" for n in range(1, len(request_sources) + 1)]


def _refs(value: Any, known: dict[str, tuple[str, int]]) -> list[SourceRef]:
    """The model's citations ("S1-3", or objects) as references to chapters the app sent."""
    refs: list[SourceRef] = []
    seen: set[tuple[str, int]] = set()
    for entry in value if isinstance(value, list) else []:
        key: tuple[str, int] | None = None
        if isinstance(entry, str):
            key = known.get(entry.strip().strip("[]").upper().replace(" ", ""))
        elif isinstance(entry, dict):
            source, index = entry.get("sourceId"), _int(entry.get("chapterIndex"))
            label = f"{str(source).strip().upper()}-{index}"
            key = known.get(label) or next((k for k in known.values() if k == (source, index)), None)
        if key and key not in seen:
            seen.add(key)
            refs.append(SourceRef(source_id=key[0], chapter_index=key[1]))
    return refs


def _sections(value: Any) -> list[Section]:
    sections = []
    for entry in value if isinstance(value, list) else []:
        if isinstance(entry, dict) and _text(entry.get("title")):
            sections.append(Section(title=_text(entry["title"]), topics=_strings(entry.get("topics"), 10)))
        elif isinstance(entry, str) and _text(entry):
            sections.append(Section(title=_text(entry)))
    return sections[:12]


def _words(value: Any, default: int) -> int:
    words = _int(value)
    return max(MIN_CHAPTER_WORDS, min(MAX_CHAPTER_WORDS, words)) if words else default


def _plan(data: Any, *, default_words: int, refs: Any) -> Blueprint:
    """A plan from the model, checked. ``refs`` maps each chapter dict to its references."""
    if not isinstance(data, dict):
        raise ServiceError("DeepSeek returned a plan in the wrong shape.")
    chapters = []
    for entry in data.get("chapters") if isinstance(data.get("chapters"), list) else []:
        if not isinstance(entry, dict) or not _text(entry.get("title")):
            continue
        chapters.append(
            BlueprintChapter(
                title=re.sub(r"^chapter\s+[\divxlc]+\s*[:.\-–—]\s*", "", _text(entry["title"]), flags=re.IGNORECASE),
                summary=_text(entry.get("summary")),
                sections=_sections(entry.get("sections")),
                estimated_words=_words(entry.get("estimatedWords"), default_words),
                source_refs=refs(entry),
            )
        )
    if not chapters:
        raise ServiceError("DeepSeek returned a plan with no chapters.")
    return Blueprint(
        title=_text(data.get("title")) or "Untitled",
        subtitle=_text(data.get("subtitle")),
        description=_text(data.get("description")),
        audience=_text(data.get("audience")),
        tone=_text(data.get("tone")),
        chapters=chapters,
    )


async def blueprints(request: BlueprintRequest) -> BlueprintResponse:
    if request.mode != "idea" and not request.sources:
        raise ServiceError(f"A {request.mode} plan needs at least one source.", status=422)
    brief = request.brief
    if request.mode == "idea" and not any(v.strip() for v in (brief.idea, brief.instructions, brief.approach, brief.genre)):
        raise ServiceError("Describe the idea for the book first.", status=422)

    labels = _labels(request.sources)
    known = {f"{label}-{c.index}": (s.id, c.index) for label, s in zip(labels, request.sources) for c in s.chapters}
    sources = prompts.source_index(request.sources, labels, SOURCES_BUDGET) if request.sources else ""
    # An idea is planned to the length chosen for it; a book from sources to the length its
    # material supports, when the app says what that is.
    target = request.brief.target_words if request.mode != "idea" else 0
    planned = prompts.source_chapters(target) if target else prompts.LENGTHS[request.brief.length][0]
    default_words = (
        max(MIN_CHAPTER_WORDS, round(target / planned)) if target else prompts.LENGTHS[request.brief.length][1]
    )

    def refs(entry: dict[str, Any]) -> list[SourceRef]:
        return [] if request.mode == "idea" else _refs(entry.get("sources") or entry.get("sourceRefs"), known)

    expected = request.brief.chapter_count or max(planned, sum(len(s.chapters) for s in request.sources))
    tokens = min(llm.MAX_OUTPUT_TOKENS, 260 * expected + 1000)

    async def plan(direction: str | None = None, others: list[str] | None = None) -> Blueprint:
        system, user = prompts.blueprint(request, sources, direction, others or [])
        data = await llm.ask_json(system, user, max_tokens=tokens, temperature=TEMPERATURE["blueprint"])
        return _plan(data, default_words=default_words, refs=refs)

    if request.count == 1:
        return BlueprintResponse(proposals=[await plan()])

    # Several versions: first agree on how they differ, then plan each in full. Asked
    # for three plans in one go, a model writes one plan three times in different words.
    system, user = prompts.directions(request, sources, request.count)
    data = await llm.ask_json(system, user, max_tokens=1500, temperature=TEMPERATURE["directions"])
    found = [
        f"{_text(d.get('title'))}: {_text(d.get('angle'))}".strip(": ")
        for d in (data.get("directions") if isinstance(data.get("directions"), list) else [])
        if isinstance(d, dict) and (_text(d.get("title")) or _text(d.get("angle")))
    ][: request.count]
    if not found:
        return BlueprintResponse(proposals=[await plan()])
    # A version that fails is left out rather than failing the others with it.
    proposals = await _gather_some(plan(d, found[:n] + found[n + 1 :]) for n, d in enumerate(found))
    return BlueprintResponse(proposals=proposals)


async def refine(request: RefineRequest) -> RefineResponse:
    original = request.blueprint
    history = request.history[-REFINE_HISTORY_TURNS:]
    system, user = prompts.refine(original, request.message, history)
    tokens = min(llm.MAX_OUTPUT_TOKENS, 260 * (len(original.chapters) + 3) + 1000)
    data = await llm.ask_json(system, user, max_tokens=tokens, temperature=TEMPERATURE["refine"])
    reply = _text(data.get("reply")) or "Done."
    revised = data.get("blueprint")
    if not isinstance(revised, dict) or not isinstance(revised.get("chapters"), list) or not revised["chapters"]:
        return RefineResponse(reply=reply, blueprint=original)

    def origins(entry: dict[str, Any]) -> list[BlueprintChapter]:
        origin = entry.get("from")
        numbers = [_int(n) for n in (origin if isinstance(origin, list) else [origin])]
        return [original.chapters[n - 1] for n in numbers if n is not None and 1 <= n <= len(original.chapters)]

    def carried(entry: dict[str, Any]) -> list[SourceRef]:
        # A chapter that continues one of the originals keeps its source references;
        # the model is never asked to copy them, so it cannot garble them.
        refs: list[SourceRef] = []
        for chapter in origins(entry):
            refs += [r for r in chapter.source_refs if r not in refs]
        return refs

    for entry in revised["chapters"]:
        # What the model left off a chapter it kept is the chapter's as it was.
        sources = origins(entry) if isinstance(entry, dict) else []
        if len(sources) == 1:
            kept = sources[0]
            entry.setdefault("summary", kept.summary)
            entry.setdefault("sections", [s.model_dump() for s in kept.sections])
            if not _int(entry.get("estimatedWords")):
                entry["estimatedWords"] = kept.estimated_words

    plan = _plan(revised, default_words=1500, refs=carried)
    # A field the model left out stays as it was; one it emptied on purpose stays empty.
    for name in ("title", "subtitle", "description", "audience", "tone"):
        if name not in revised:
            setattr(plan, name, getattr(original, name))
    return RefineResponse(reply=reply, blueprint=plan)


# --- writing ------------------------------------------------------------------


def _excerpts(request: ChapterRequest) -> str:
    excerpts = [(e.title.strip(), as_markdown(e.text)) for e in request.source_excerpts]
    excerpts = [(title, text) for title, text in excerpts if text]
    limits = _share([text for _, text in excerpts], EXCERPTS_BUDGET)
    return "\n\n".join(
        f"--- {title or f'Excerpt {n}'} ---\n{prompts.sample(text, limit)}"
        for n, ((title, text), limit) in enumerate(zip(excerpts, limits), start=1)
    )


def _is_title(block: dict[str, Any], title: str) -> bool:
    """Is this block the chapter's title (or a bare "Chapter 3"), which the layout adds?"""
    if block["type"] not in ("heading", "paragraph"):
        return False

    def squash(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    text = plain(block["html"])
    stripped = re.sub(r"^chapter\s+[\w]+\s*[:.\-–—]?\s*", "", text, flags=re.IGNORECASE)
    if block["type"] == "heading" and re.fullmatch(r"chapter\s+[\w]+", text.strip(), flags=re.IGNORECASE):
        return True
    return bool(squash(title)) and squash(title) in {squash(text), squash(stripped)}


def _pick(items: list[T], count: int) -> list[T]:
    """``count`` of ``items``, spread evenly rather than the first few."""
    if len(items) <= count:
        return items
    step = len(items) / count
    return [items[int(n * step)] for n in range(count)]


def _scene(prompt: str) -> str:
    """An image prompt without quoted words, which an image model letters onto the picture."""
    return " ".join(QUOTED.sub(" ", prompt).split()).strip(" ,;")


def _anchor(anchor: str, text: str) -> str:
    """The quotation as it appears in the text, or the closest verbatim passage to it.

    The app finds the passage on the page by this quotation, so it must be the text's
    own words even when the model reworded or misquoted them.
    """
    words = re.findall(r"\w+", anchor or "")
    if not words or not text:
        return ""
    for size in range(min(len(words), 15), 3, -1):
        for start in range(0, len(words) - size + 1):
            pattern = r"\W+".join(re.escape(w) for w in words[start : start + size])
            found = re.search(pattern, text, flags=re.IGNORECASE)
            if found:
                return text[found.start() : found.end()]
    # Nothing matched word for word: the sentence sharing the most words, from its start.
    wanted = {w.lower() for w in words}
    sentences = re.findall(r"[^.!?\n]+[.!?]?", text)
    best = max(sentences, key=lambda s: len(wanted & {w.lower() for w in re.findall(r"\w+", s)}), default="")
    return " ".join(best.split()[:15]).strip()


async def illustration_prompts(request: IllustrationPromptsRequest) -> IllustrationPromptsResponse:
    text = as_text(request.chapter.text)
    if request.purpose == "interior" and not text:
        raise ServiceError("The chapter has no text to illustrate.", status=422)
    # Without a style (a chapter asking for pictures has none to give), none is named.
    style = (
        style_text(request.style_id, request.custom_style, request.art_style, request.medium, request.substyle)
        if request.style_id or request.art_style
        else ""
    )
    sampled = prompts.spread(text, ILLUSTRATION_TEXT_BUDGET) if request.purpose == "interior" else prompts.sample(text, 3000)
    system, user = prompts.illustration_ideas(request, style, sampled, request.count)
    data = await llm.ask_json(system, user, max_tokens=300 * request.count + 400, temperature=TEMPERATURE["illustrations"])
    ideas = []
    for entry in data.get("prompts") if isinstance(data.get("prompts"), list) else []:
        if isinstance(entry, str):
            entry = {"prompt": entry}
        if not isinstance(entry, dict) or not _scene(_text(entry.get("prompt"))):
            continue
        anchor = _anchor(_text(entry.get("anchor")), text) if request.purpose == "interior" else ""
        ideas.append(IllustrationIdea(prompt=_scene(_text(entry["prompt"])), caption=_text(entry.get("caption")), anchor=anchor))
    if not ideas:
        raise ServiceError("DeepSeek returned no illustration ideas.")
    return IllustrationPromptsResponse(prompts=ideas[: request.count])


def _block_of(anchor: str, blocks: list[dict[str, Any]]) -> int:
    """The index of the block holding a quotation (the last block when none does)."""
    wanted = " ".join(anchor.lower().split())
    for index, block in enumerate(blocks):
        pieces = block.get("items") or [block.get("html", "")]
        if any(wanted and wanted in " ".join(plain(p).lower().split()) for p in pieces):
            return index
    return len(blocks) - 1


async def write_chapter(request: ChapterRequest) -> ChapterResponse:
    plan = request.chapter
    words = max(MIN_CHAPTER_WORDS, min(MAX_CHAPTER_WORDS, plan.estimated_words or 1500))
    wanted = request.illustrations.per_chapter if request.illustrations.enabled else 0
    excerpts = _excerpts(request)
    system, user = prompts.chapter(request, excerpts, words, wanted)
    faithful = request.mode in ("extract", "documents") or (
        request.mode == "rewrite" and (request.rewrite or Rewrite()).depth in ("light", "moderate")
    )
    reply = await llm.ask(
        system,
        user,
        max_tokens=int(words * TOKENS_PER_WORD) + 1200 + 120 * wanted,
        temperature=TEMPERATURE["chapter_faithful" if faithful else "chapter"],
        long=True,
    )

    marked = SUMMARY.search(reply)
    body, recap = (reply[: marked.start()], reply[marked.end() :]) if marked else (reply, "")

    # Each picture line becomes a placeholder paragraph, so that once the text is in
    # blocks the picture's place is simply "after the block before the placeholder".
    pictures: list[tuple[str, str]] = []

    def hold(match: re.Match[str]) -> str:
        prompt, _, caption = match.group(1).partition("|")
        pictures.append((_scene(_text(prompt)), plain(caption)))
        return "\n\n" + PLACEHOLDER.format(len(pictures) - 1) + "\n\n"

    blocks = parse_blocks(PICTURE_INLINE.sub(hold, PICTURE.sub(hold, body)))
    while blocks and (PLACEHOLDER_RE.match(blocks[0].get("html", "")) is None) and _is_title(blocks[0], plan.title):
        blocks.pop(0)

    placed: list[PlacedIllustration] = []
    content: list[dict[str, Any]] = []
    for block in blocks:
        held = PLACEHOLDER_RE.match(block.get("html", "")) if block["type"] == "paragraph" else None
        if held:
            prompt, caption = pictures[int(held.group(1))]
            if prompt:
                placed.append(PlacedIllustration(after_block=max(0, len(content) - 1), prompt=prompt, caption=caption))
        else:
            content.append(block)

    if not marked and content and content[-1]["type"] == "paragraph" and not ENDS_SENTENCE.search(plain(content[-1]["html"])):
        # No recap means the reply was most likely cut off at the length limit, so its
        # last paragraph stops mid-sentence. Better to end on the one before.
        content.pop()
    if not content:
        raise ServiceError("DeepSeek returned a chapter with no text.")

    chapter_text = "\n\n".join(plain(b["html"]) if "html" in b else "\n".join(map(plain, b["items"])) for b in content)
    # Picture lines after the recap have no place in the chapter; the ones it lacks are
    # chosen from the written text below.
    summary = plain(PICTURE_INLINE.sub(" ", PICTURE.sub(" ", recap)))
    if not summary:
        try:
            system, user = prompts.recap(plan.title, chapter_text)
            summary = plain(await llm.ask(system, user, max_tokens=400, temperature=TEMPERATURE["recap"]))
        except ServiceError as error:
            # The chapter is written; losing its recap is no reason to lose it.
            logger.warning("Talari: no recap for chapter %r: %s", plan.title, error.message)

    placed = [p for p in placed if p.after_block < len(content)]
    if wanted:
        placed = _pick(placed, wanted)
        if len(placed) < wanted:
            placed = await _more_pictures(request, chapter_text, content, placed, wanted)
    else:
        placed = []

    return ChapterResponse(blocks=[Block(**b) for b in content], summary=summary, illustrations=placed)


async def _more_pictures(
    request: ChapterRequest,
    chapter_text: str,
    content: list[dict[str, Any]],
    placed: list[PlacedIllustration],
    wanted: int,
) -> list[PlacedIllustration]:
    """Pictures the chapter reply did not place, chosen from the written text instead."""
    try:
        ideas = await illustration_prompts(
            IllustrationPromptsRequest(
                book={"title": request.book.title, "description": request.book.description, "tone": request.book.tone},
                chapter={"title": request.chapter.title, "text": chapter_text},
                count=wanted - len(placed),
            )
        )
    except ServiceError as error:
        logger.warning("Talari: no extra pictures for chapter %r: %s", request.chapter.title, error.message)
        return placed
    taken = {p.after_block for p in placed}
    for idea in ideas.prompts:
        after = _block_of(idea.anchor, content)
        if after not in taken:
            taken.add(after)
            placed.append(PlacedIllustration(after_block=after, prompt=idea.prompt, caption=idea.caption))
    return sorted(placed, key=lambda p: p.after_block)


# --- revising -----------------------------------------------------------------


async def revise(request: ReviseRequest) -> ReviseResponse:
    passage = as_markdown(request.html)
    context = request.context
    wrap = bool(re.search(r"<p\b", request.html, flags=re.IGNORECASE))
    original = parse_blocks(request.html)

    if request.action == "factcheck":
        if not passage:
            raise ServiceError("There is no text to check.", status=422)
        system, user = prompts.factcheck(passage, context.book_title, context.chapter_title)
        data = await llm.ask_json(system, user, max_tokens=1500, temperature=TEMPERATURE["factcheck"])
        notes = [n for n in _strings(data.get("claims"), 30) if not CLAIMS_VERIFIED.search(n)]
        if not notes:
            notes = [
                "No specific factual claims stood out to flag in this passage. Nothing here has been verified, "
                "so check anything you are unsure of."
            ]
        return ReviseResponse(html=blocks_as_html(original, wrap=wrap or len(original) > 1), notes=notes)

    before = as_text(context.before)
    if not passage and not (request.action == "continue" and before):
        raise ServiceError("There is no text to revise.", status=422)
    if request.action == "tone" and not (request.tone.strip() or request.instruction.strip()):
        raise ServiceError("Say which tone to use.", status=422)
    if request.action == "custom" and not request.instruction.strip():
        raise ServiceError("Say what to do with the passage.", status=422)

    system, user = prompts.revise(
        request.action,
        passage,
        instruction=request.instruction.strip(),
        tone=request.tone.strip(),
        book_title=context.book_title,
        chapter_title=context.chapter_title,
        before=before,
        after=as_text(context.after),
    )
    words = len(passage.split())
    grow = {"expand": 2.6, "examples": 2.0, "continue": 0}.get(request.action, 1.5)
    tokens = int(words * grow * TOKENS_PER_WORD) + (900 if request.action == "continue" else 500)
    reply = await llm.ask(
        system,
        user,
        max_tokens=max(600, tokens),
        temperature=REVISE_TEMPERATURE[request.action],
        long=words > 2500,
    )

    marked = NOTES.search(reply)
    text, tail = (reply[: marked.start()], reply[marked.end() :]) if marked else (reply, "")
    notes = [plain(re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line)) for line in tail.splitlines()]
    notes = [n for n in notes if n and not CLAIMS_VERIFIED.search(n)]
    revised = parse_blocks(text)
    if not revised:
        raise ServiceError("DeepSeek returned no text.")
    if request.action == "continue":
        revised = original + revised
    return ReviseResponse(html=blocks_as_html(revised, wrap=wrap or len(revised) > 1), notes=notes)
