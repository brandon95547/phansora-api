"""The seven kinds of research the start screen offers, and the prompt each one sends.

Every kind runs the same pipeline as a trace: one grounded call in which the model runs
its own searches and answers with a JSON array, read by ``dated_list`` and laid on the
board. What differs is the question, and the question is the whole product: a
chronology asks for an ORDER, a genealogy for DESCENT, an interpretation for a MEANING, a
parallel for a COMPARISON, a translation for WORDING, an establishment for a HIERARCHY.
Changing the heading on the same trace prompt would have answered them all with the same
lineage.

Origins is the trace itself. ``RESEARCH_PROMPT`` is tuned by hand against the live model,
so its focus and perspectives are added after it, as the context line is, rather than
edited into it.

The subjects are deliberately unrestricted. These are written to read the same way for a
myth, a war, a word, a scientific idea, a law, a craft or a piece of software, because
the person asking may be researching any of them. Interpretive perspectives are reported
as their proponents hold them, never asserted and never dismissed.

Prompts are f-strings, so every literal brace in their JSON templates is doubled.
"""
from __future__ import annotations

from typing import List, Optional

from ..models import ResearchSpec

# --------------------------------------------------------------------- vocabulary
# What each choice means, in the words the model is given. The dashboard's labels and
# descriptions are written for a reader; these are written for a search.

CHRONOLOGY_PERSPECTIVES = {
    "narrative": (
        "Narrative",
        "the account as a selected text, tradition or source tells it: its events, in the "
        "words and order the account gives them",
    ),
    "historical": (
        "Evidence",
        "what archaeology, records and scholarship show actually happened: the events, "
        "people, places and dates behind the subject as historians reconstruct them, apart "
        "from what the account claims",
    ),
}

# Interpretations: what the writers may have encoded beneath the surface. One reading per
# request, chosen as its focus, and each asks whether the WRITERS meant the subject this
# way, so each entry is graded by intent.
INTERPRETATION_READINGS = {
    "coded_political": (
        "Political & Coded History",
        "readings that hold the writers disguised real rulers, nations and events of their "
        "own time in the account: coded commentary on the world they wrote in",
    ),
    "numerical": (
        "Symbolic & Numerical",
        "readings that hold the writers encoded meaning in numbers, names and symbols: "
        "gematria, number schemes, symbolic names and counts",
    ),
    "ritual": (
        "Seasonal & Ritual",
        "readings that hold the account began as the words or pattern of a seasonal rite: "
        "a festival, the harvest, the death and return of the year",
    ),
    "mystical": (
        "Spiritual & Mystical",
        "readings that hold the writers meant the account as a picture of the soul: its "
        "fall, trials, awakening or return",
    ),
    "solar_astronomical": (
        "Solar & Astronomical",
        "readings that hold the writers encoded the sky in the account: the sun's year, the "
        "moon, the planets, the constellations and zodiac, equinoxes, solstices and the "
        "calendar",
    ),
    "initiatory": (
        "Bodily & Initiatory",
        "readings that hold the account encodes teachings for initiates: rites of passage, "
        "the body as a temple, the stages of an initiation",
    ),
}

# An interpretation answers one question here: did the people who made the subject mean
# it this way? A lesson a preacher draws, or an archetype a psychologist sees,
# is a reading made later, and saying so is the point. Where nobody has proposed a reading
# for the subject, the honest answer is one entry that says so, not an invented reading.
_INTENT_RULE = (
    "THIS IS AN INTERPRETATION ({name}): the question is whether the people who made "
    "the subject meant it this way, not what later readers drew from it. Grade every "
    "entry in \"intent\": \"intended\" where evidence shows the writers meant it, "
    "\"debated\" where specialists disagree, or \"later reading\" where it was read into the "
    "subject after it was made, naming in \"authorship\" who first proposed it. Where "
    "no one has proposed this reading for this subject, return a single entry titled "
    "\"No recorded reading\" with \"intent\": \"no tradition\", and invent nothing."
)

GENEALOGY_FOCUS = {
    "families": (
        "people and families",
        "parents, children, spouses and siblings, as recorded in vital records, census and "
        "parish registers, wills, family papers, peerages and published genealogies",
        "parent",
    ),
    "dynasties": (
        "dynasties and rulers",
        "royal, imperial and noble houses: who descended from whom, who succeeded whom, and "
        "how the title passed, including adoptions, regencies and disputed successions",
        "parent or predecessor",
    ),
    "deities": (
        "gods and mythic figures",
        "divine, heroic and legendary parentage as myths, scriptures, epics and traditions "
        "give it, including where traditions give a figure different parents",
        "parent",
    ),
    "schools": (
        "ideas and schools",
        "teachers and students, founders and followers, and the schools, movements and "
        "traditions that grew from one another",
        "teacher or forerunner",
    ),
    "languages": (
        "languages",
        "parent languages, proto-languages, branches and daughter languages, as historical "
        "linguists reconstruct them",
        "parent language",
    ),
}

GENEALOGY_PERSPECTIVES = {
    "historical": ("Historical", "documented records: births, marriages, deaths and succession"),
    "traditional": ("Traditional & Mythic", "lines of descent as myths, scriptures and traditions give them"),
    "textual": ("Textual", "genealogies as written in particular texts, and where those texts differ"),
}

_DIRECTIONS = {
    "ancestors": (
        "ANCESTORS. Start from the subject and follow the line BACK, generation by "
        "generation, through every {parent} the research names, as far as the evidence "
        "or tradition reaches."
    ),
    "descendants": (
        "DESCENDANTS. Start from the subject and follow the line FORWARD, generation by "
        "generation, through every child, heir or successor the research names, as far as "
        "the evidence or tradition reaches."
    ),
    "both": (
        "ANCESTORS AND DESCENDANTS. Follow the line BACK from the subject through every "
        "{parent} the research names, and FORWARD through every child, heir or successor, "
        "with the subject itself between them."
    ),
}

ORIGINS_PERSPECTIVES = {
    "historical": ("Evidence", "documented events, dates and people, as records and scholarship establish them"),
    "textual": ("Textual", "the earliest writings, manuscripts and editions"),
    "archaeological": ("Archaeological & Material", "physical remains, artifacts and sites"),
    "linguistic": ("Linguistic", "words, names and their roots"),
}

PARALLEL_PERSPECTIVES = {
    "historical": ("Historical", "documented contact, borrowing and influence"),
    "literary": ("Literary", "plot, structure, motifs and narrative patterns"),
    "symbolic": ("Symbolic", "shared symbols, imagery and meanings"),
    "cultural": ("Cultural", "social role, function and setting"),
    "natural": ("Natural & Astronomical", "seasonal, natural or celestial cycles"),
}

ORIGINS_FOCUS = {
    "inventions": (
        "Focus on the invention or discovery: who first made or found it, where and when, "
        "the need it answered, the earlier techniques, materials and knowledge that made it "
        "possible, and how it spread and was improved."
    ),
    "customs": (
        "Focus on the custom or tradition: when and where it is first recorded, who first "
        "kept it, the occasion, need or belief it grew out of, and how it spread and changed."
    ),
    "words": (
        "Focus on the word or name: its etymology, its earliest attestations, the older "
        "words and roots it descends from, and how its form and meaning shifted. "
        "Inscriptions, glossaries and dictionaries are primary material here."
    ),
    "ideas": (
        "Focus on the idea or belief: who first stated it, where and when, the earlier "
        "ideas and questions it developed from, and how its meaning shifted as others took "
        "it up."
    ),
    "texts": (
        "Focus on the text or story as a written work: who first wrote it down, when, where "
        "and why; the sources, events and circumstances its authors drew on; its earliest "
        "surviving copies; and how it was edited, collected and received."
    ),
}

PARALLEL_FOCUS = {
    "stories": ("stories and events", "plots, episodes, narrative motifs and events"),
    "people": ("people and figures", "figures, roles and life patterns: birth, calling, "
               "trials, death, return"),
    "symbols": ("symbols and concepts", "symbols, imagery, ideas and concepts"),
    "practices": ("practices and rituals", "practices, rituals, customs, techniques and "
                  "institutions"),
}

TRANSLATION_FOCUS = {
    "wording": (
        "WORDING AND MEANING: what the words are in each version, what they meant in their "
        "own language and time, and how the meaning shifts from version to version."
    ),
    "choices": (
        "TRANSLATION CHOICES: the decisions translators made, which word each chose and "
        "what it chose it over, why (the source text it worked from, philology, doctrine, "
        "law, audience, style), and who disputes the choice."
    ),
    "manuscripts": (
        "MANUSCRIPT DIFFERENCES: how the text differs between manuscripts, recensions, "
        "drafts or early witnesses: additions, omissions and variant readings, and which "
        "later versions follow which witness."
    ),
}

# What kind of body an Establishment maps: (name, its parts, what its top is called). The
# words its offices and units go by differ by kind, and a prompt that only said "leader"
# and "division" would map a church as if it were a company.
ESTABLISHMENT_FOCUS = {
    "religions": (
        "a religion, church or denomination",
        "its supreme authority, governing councils and synods, the ranks of its clergy, its "
        "provinces, dioceses and congregations, its rites and member churches, and its "
        "religious orders, institutes and societies",
        "supreme authority",
    ),
    "societies": (
        "an order, society or fraternity",
        "its grand or supreme bodies, its officers, its lodges, chapters and degrees, its "
        "jurisdictions, its appendant and concordant bodies, and its auxiliary orders",
        "grand or supreme body",
    ),
    "governments": (
        "a government, state or public body",
        "its head of state and government, its executive, legislature and judiciary, its "
        "ministries, departments and agencies, its armed forces, and its regional and local "
        "governments",
        "head of state or highest authority",
    ),
    "companies": (
        "a company or institution",
        "its owners and board, its executive officers, its divisions, business units and "
        "subsidiaries, its regional operations, and its foundations and affiliates",
        "owners, board or chief executive",
    ),
    "movements": (
        "a movement, party or cause",
        "its leadership and founders, its governing committees, its wings, chapters and "
        "branches, its youth, women's and affiliated organizations, and its allied groups",
        "leadership",
    ),
}


# --------------------------------------------------------------------------- shared
_JSON_RULES = """\
Every object must have "title" and "date". Leave any other TEXT field as an empty string when the research does not support it. Do not guess a value to fill a field.

"is_collection" is true when the item is a set of separately made works gathered under one name (an anthology, a canon, a series, a catalogue, a product line, a standards family) and false for a single work, object, person, place or event. It must be present on every object.

Return as many valid items as the research supports. Do not shorten the list.

No prose before or after the array. No code fences.
"""


def _subject(title: str, context: Optional[str]) -> str:
    title = title.strip()
    return f'"{title}" ({context.strip()})' if context and context.strip() else f'"{title}"'


def _listed(names: List[str]) -> str:
    names = [n for n in names if n]
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _clean(values: List[str], limit: int = 120) -> List[str]:
    out: List[str] = []
    for v in values or []:
        v = " ".join(str(v).split())[:limit]
        if v and v.casefold() not in {o.casefold() for o in out}:
            out.append(v)
    return out


# ------------------------------------------------------------------------- origins
# Origins answers one question, "how did this come to be?", and leaves the others to the
# kinds that own them: an older story of another culture that resembles this one is
# Parallels, a family tree is Genealogy.
# The trace prompt it follows asks for predecessors in general, so this says which
# predecessors count. Sent with every Origins request; a plain trace (no research at all)
# is left as it always was.
_ORIGINS_QUESTION = """\
THE QUESTION FOR THIS TRACE: how did "{title}" come to be? Include, each as entries of their own:

1. Its point of origin: the first known appearance of "{title}", where and when, and who made, wrote, said or found it.
2. What made it possible: the need or occasion it answered, and the earlier techniques, materials, ideas, words or events it was built from.
3. How it spread and changed, step by step, from that first appearance to its present-day form.
4. Origin myths: popular claims about where "{title}" came from that the evidence does not support. Give each its own entry, dated when the claim first circulated, and begin its "significance" with "Origin myth:" followed by what the evidence shows instead.

A predecessor here is something "{title}" was built from or grew out of. Older versions of the same story, myth or motif told by other cultures or traditions are not its origin: leave them out, even where scholars think this one drew on them. They are researched as Parallels."""


def origins_addendum(title: str, spec: Optional[ResearchSpec]) -> str:
    """What Origins adds after the hand-tuned trace prompt. Empty for a plain trace."""
    if spec is None:
        return ""
    parts: List[str] = [_ORIGINS_QUESTION.replace("{title}", title)]
    if spec.focus in ORIGINS_FOCUS:
        parts.append(ORIGINS_FOCUS[spec.focus])
    chosen = [ORIGINS_PERSPECTIVES[p] for p in spec.perspectives if p in ORIGINS_PERSPECTIVES]
    if chosen:
        lines = "\n".join(f"* {label}: {desc}" for label, desc in chosen)
        parts.append(
            "Search the origin through these perspectives, giving each its own searches, "
            f"and include what each one finds:\n\n{lines}"
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------- chronology
_SEQUENCE_RULES = {
    "narrative": (
        "NARRATIVE ORDER. Order the entries by the order of events the account itself "
        "implies, not by when each source was written. Where the account is told across "
        "several books, texts, episodes, sources or tellings, bring the related passages "
        "together at the point in the sequence where they belong, and name every passage "
        "an entry draws on in \"passage\"."
    ),
    "historical": (
        "BY DATE. Order the entries by historical date, oldest first. Give each "
        "date as the evidence supports it and mark its uncertainty in \"certainty\": "
        "established, approximate, disputed, or traditional (held by a tradition but not "
        "established by evidence)."
    ),
}


def chronology_prompt(title: str, context: Optional[str], spec: ResearchSpec) -> str:
    # One timeline, read through one lens (ResearchSpec allows a chronology one).
    lens = next((p for p in spec.perspectives if p in CHRONOLOGY_PERSPECTIVES), "historical")
    label, description = CHRONOLOGY_PERSPECTIVES[lens]
    sequence = spec.sequence if spec.sequence in _SEQUENCE_RULES else "historical"
    subject = _subject(title, context)

    return f"""\
Using live web search, build the chronology of {subject} as read through this perspective:

* {label}: {description}

{_SEQUENCE_RULES[sequence]}

Report each interpretation as the people who hold it hold it, naming whose reading it is: do not present an interpretation as historical fact, and do not dismiss it. Where the historical record contradicts a reading, say so in that entry.

The goal is a large, complete, ordered list with rich detail, not a brief summary. Include every stage or event the research supports.

Return the result as a JSON array and nothing else, one object per entry, in the order described above:

[
  {{
    "title": "What happens at this point, named briefly",
    "stage": "Its place in the sequence, e.g. 1, Stage 3, Winter solstice, Exodus 14",
    "date": "Historical date or range where one is known, e.g. c. 1200 BCE, 1066 CE; otherwise empty",
    "certainty": "established, approximate, disputed or traditional",
    "passage": "The passages, chapters or sources this entry draws on",
    "origin": "Where this account or reading comes from",
    "authorship": "Whose account or reading this is",
    "significance": "What happens, or what this stage means in this perspective",
    "is_collection": false
  }}
]

{_JSON_RULES}"""


# ------------------------------------------------------------------- interpretations
# What the subject was meant to encode, through one reading. Its own kind, not a lens on a
# timeline: a number, a symbol or a building has no order of events, so the entries follow
# the subject's own order where it has one (a story's passages, a rite's stages, the
# year's seasons) and go element by element where it does not.
def interpretations_prompt(title: str, context: Optional[str], spec: ResearchSpec) -> str:
    reading = spec.focus if spec.focus in INTERPRETATION_READINGS else "numerical"
    label, description = INTERPRETATION_READINGS[reading]
    subject = _subject(title, context)

    return f"""\
Using live web search, find how {subject} has been read through this interpretation:

* {label}: {description}

{_INTENT_RULE.format(name=label)}

Give each element the reading decodes its own entry: a passage, figure, number, name, symbol, motif or feature of the subject, and what this reading takes it to mean. Order the entries by the subject's own sequence where it has one (the order of a story's passages, a rite's stages, the year's seasons, an initiation's degrees), and element by element where it has none. Report each reading as the people who hold it hold it, naming whose reading it is: do not present it as fact, and do not dismiss it. Where the evidence contradicts a reading, say so in that entry.

The goal is a large, complete list with rich detail, not a brief summary. Include every element the research supports.

Return the result as a JSON array and nothing else, one object per entry, in the order described above:

[
  {{
    "title": "The element and what it is read as, named briefly, e.g. 666 as Nero Caesar",
    "stage": "Its place in the subject's sequence, e.g. Exodus 14, Stage 3, Winter solstice; empty where there is none",
    "element": "The passage, figure, number, symbol or feature being read",
    "reading": "What this interpretation takes it to mean",
    "date": "When this reading is first attested, e.g. c. 180 CE, 1890s",
    "intent": "intended, debated, later reading or no tradition",
    "authorship": "Who first proposed this reading, and who holds it now",
    "origin": "The tradition, school or text the reading comes from",
    "significance": "Why the reading matters, and what the evidence says of it",
    "is_collection": false
  }}
]

{_JSON_RULES}"""


# ----------------------------------------------------------------------- genealogy
def genealogy_prompt(title: str, context: Optional[str], spec: ResearchSpec) -> str:
    name, scope, parent = GENEALOGY_FOCUS.get(spec.focus or "", GENEALOGY_FOCUS["families"])
    direction = _DIRECTIONS.get(spec.direction or "", _DIRECTIONS["ancestors"]).format(parent=parent)
    chosen = [GENEALOGY_PERSPECTIVES[p] for p in spec.perspectives if p in GENEALOGY_PERSPECTIVES]
    lens = ""
    if chosen:
        lens = "\n\nSearch the line through these perspectives, giving each its own searches:\n\n" + "\n".join(
            f"* {label}: {desc}" for label, desc in chosen
        )

    return f"""\
Using live web search, trace the genealogy of {_subject(title, context)} among {name}: {scope}.{lens}

{direction}

Include the subject itself as its own entry. Give each member of the line its own entry, and give "generation" relative to the subject: 0 for the subject, -1 for a {parent}, -2 for the generation before that, 1 for a child or successor, and so on.

Where sources or traditions disagree about a link (two different fathers, a disputed succession, a contested language family), include each version and say whose it is in "line". Report traditional and mythic descent as the tradition gives it, naming the tradition: do not present it as historical fact, and do not dismiss it. Never invent a link to complete a line: where it breaks, end it.

The goal is a large, complete line with rich detail, not a brief summary. Include every member the research supports, not only the famous ones.

Return the result as a JSON array and nothing else, one object per member of the line, earliest generation first:

[
  {{
    "title": "The person, figure, house, school or language, by its usual name",
    "date": "Birth to death, reign, founding or period of use, e.g. 742-814 CE, c. 3500 BCE; empty where none is known",
    "generation": "Its generation relative to the subject, e.g. -2, 0, 3",
    "relationship": "How it is related to the subject, e.g. paternal grandfather, son, teacher, parent language",
    "line": "Which line or version of the genealogy it belongs to, and whose account that is",
    "origin": "Where it lived, ruled, taught or was spoken",
    "authorship": "The record, text or tradition that attests the link",
    "significance": "Who it was, and why it matters to the line",
    "is_collection": false
  }}
]

{_JSON_RULES}"""


# ----------------------------------------------------------------------- parallels
def parallels_prompt(title: str, context: Optional[str], spec: ResearchSpec) -> str:
    focus_name, focus_desc = PARALLEL_FOCUS.get(spec.focus or "", PARALLEL_FOCUS["stories"])
    subjects = _clean(spec.subjects, 200) or [title.strip()]
    chosen = [PARALLEL_PERSPECTIVES[p] for p in spec.perspectives if p in PARALLEL_PERSPECTIVES]
    lens = ""
    if chosen:
        lens = "\n\nCompare through these perspectives, searching each separately:\n\n" + "\n".join(
            f"* {label}: {desc}" for label, desc in chosen
        )
    context_line = f"\n\nThe subject is understood in this context: {context.strip()}" if context and context.strip() else ""

    if len(subjects) > 1:
        quoted = _listed([f'"{s}"' for s in subjects])
        task = f"""\
Using live web search, compare {quoted}, looking at their {focus_name}: {focus_desc}.

Include:

1. EACH SUBJECT ITSELF, as its own entry, dated to its earliest attestation.
2. EVERY SPECIFIC POINT OF COMPARISON: each feature two or more of them share, and each place where they clearly differ. Name the feature itself, not merely that they are similar. Say in "subject" which of the subjects the entry involves.

Do not reduce a relational or narrative parallel to one participant: name the whole motif, scene, relationship or pattern. Include well-known and disputed comparisons, and say plainly where a proposed parallel is weak, late or rejected by specialists."""
    else:
        quoted = f'"{subjects[0]}"'
        task = f"""\
Using live web search, find the parallels to {quoted} among {focus_name}: {focus_desc}, from other cultures, traditions, periods or fields.

Include earlier, contemporary and later parallels that share SPECIFIC features with {quoted}, including well-known and disputed comparisons, even where no contact or influence has been shown. Name the shared feature itself and say in "subject" what it parallels in {quoted}.

Do not reduce a relational or narrative parallel to one participant: name the whole motif, scene, relationship or pattern. Say plainly where a proposed parallel is weak, late or rejected by specialists."""

    return f"""\
{task}{lens}{context_line}

Before returning, search separately for each kind of parallel above. Finding one result does not complete a kind.

Return the result as a JSON array and nothing else, one object per entry, oldest first:

[
  {{
    "title": "The parallel, or the shared feature, named specifically",
    "date": "Earliest defensible attestation, e.g. c. 2100 BCE, 5th century CE",
    "origin": "Culture, tradition, place or field it comes from",
    "subject": "What it parallels, or which of the compared subjects it involves",
    "similarities": "The specific features shared",
    "differences": "The specific ways it differs",
    "relation": "direct_source, probable_influence, possible_influence, independent_parallel or disputed_parallel",
    "authorship": "Source, author or community that attests it",
    "significance": "Why the comparison matters, and who makes it",
    "is_collection": false
  }}
]

{_JSON_RULES}"""


# -------------------------------------------------------------------- translations
def translations_prompt(title: str, context: Optional[str], spec: ResearchSpec) -> str:
    focus = TRANSLATION_FOCUS.get(spec.focus or "", TRANSLATION_FOCUS["wording"])
    languages = _clean(spec.languages, 40)
    versions = _clean(spec.versions)
    scope: List[str] = []
    if languages:
        scope.append(f"Cover these languages: {_listed(languages)}.")
    if versions:
        scope.append(f"Include these versions, editions or translations by name: {_listed(versions)}.")
    if not scope:
        scope.append("Cover the original language and the most important translations and versions.")
    scope_text = " ".join(scope)

    return f"""\
Using live web search, compare how {_subject(title, context)} is worded across languages and versions. It may be a passage, a line, a saying, a law, a term or a single word, from any field.

Focus on {focus}

{scope_text}

Include the earliest form you can find and every version the research shows, each as its own entry: originals, manuscripts and early witnesses, translations, revisions and modern editions. Search separately for each language and each named version. A version not found is left out, not invented.

Quote briefly. "wording" is at most about 25 words. For a modern translation still under copyright, give only the key words or a close paraphrase, marked as a paraphrase.

Return the result as a JSON array and nothing else, one object per version, oldest first:

[
  {{
    "title": "The version, translation, edition or manuscript, by its usual name",
    "date": "When this version was made, e.g. 3rd century BCE, 1611 CE",
    "language": "Its language",
    "version": "Translator, editor, tradition or manuscript family",
    "wording": "How it words the text",
    "meaning": "What that wording means, and how it differs from the others",
    "translation_choice": "The choice made here and why, where the research says",
    "manuscript_basis": "The source text or manuscripts it follows",
    "origin": "Where it was made",
    "significance": "Why this version matters to the comparison",
    "is_collection": false
  }}
]

{_JSON_RULES}"""


# ------------------------------------------------------------------- establishment
def establishment_prompt(title: str, context: Optional[str], spec: ResearchSpec) -> str:
    name, parts, top = ESTABLISHMENT_FOCUS.get(spec.focus or "", ESTABLISHMENT_FOCUS["companies"])

    return f"""\
Using live web search, map the power structure of {_subject(title, context)}, which is {name}: {parts}.

Include:

1. THE OFFICIAL STRUCTURE, from the top down: its {top}, then every level of leadership and authority beneath it, and its major divisions and branches. Each office, body or unit is its own entry.
2. EVERYTHING OUTSIDE THE OFFICIAL STRUCTURE: splinter groups, breakaway factions, schisms, rival claimants, bodies whose standing is disputed, and bodies the official structure does not recognize. Search for these separately, as hard as for the official structure; finding one does not complete them.

Map the structure as it stands now. Include a former body only where a group outside the structure still descends from or claims it.

Map kinds of unit, not every instance: "Dioceses" is one entry, not one per diocese, unless there are only a few of them and each matters to the structure (each rite, each branch of the armed forces, each major subsidiary).

Give every entry a title no other entry has. Name in "parent_body" the entry it sits directly under, written EXACTLY as that entry's title. Leave "parent_body" empty only for the top of the structure. For a body outside the official structure, "parent_body" is the official body it broke from, rivals or claims.

Classify each entry in "standing":

* official: part of the official structure
* breakaway: a splinter group, breakaway faction or schism that left it
* disputed: claims a place in it, but that claim is contested or irregular
* unrecognized: a body the official structure does not recognize

Report each body's standing as the sources give it, and where the official structure and the group disagree about it, give both views in "relationship". Do not take a side.

The goal is a large, complete map with rich detail, not a brief summary. Include every body the research supports.

Return the result as a JSON array and nothing else, one object per entry, the top of the structure first and every entry after the one it sits under:

[
  {{
    "title": "The office, body or group, by its usual name",
    "date": "When it was founded, established or split off, e.g. 1054 CE, 1970; empty where none is known",
    "parent_body": "The title of the entry it sits directly under",
    "standing": "official, breakaway, disputed or unrecognized",
    "level": "Its place in the structure, e.g. supreme authority, governing council, division, branch, splinter group",
    "head": "Who leads it: the office, and the current holder where known",
    "authority": "What it governs, controls or answers for",
    "relationship": "How it relates to the body above it; for a body outside the structure, how and why it split, and how each side regards the other",
    "origin": "Where it is based",
    "significance": "Its role in the structure, and why it matters",
    "is_collection": false
  }}
]

{_JSON_RULES}"""


# ------------------------------------------------------------------------ dispatch
_BUILDERS = {
    "chronology": chronology_prompt,
    "interpretations": interpretations_prompt,
    "genealogy": genealogy_prompt,
    "parallels": parallels_prompt,
    "translations": translations_prompt,
    "establishment": establishment_prompt,
}


def prompt_for(title: str, context: Optional[str], spec: Optional[ResearchSpec]) -> Optional[str]:
    """The prompt for a non-Origins request, or None when the trace prompt applies."""
    if spec is None or spec.kind == "origins":
        return None
    return _BUILDERS[spec.kind](title, context, spec)


def ordering_for(spec: Optional[ResearchSpec]) -> str:
    """"given" when the research's own order is the finding and a date sort would undo it."""
    if spec is not None and spec.kind == "chronology" and spec.sequence == "narrative":
        return "given"
    # An interpretation follows the subject's own sequence, or goes element by element.
    if spec is not None and spec.kind == "interpretations":
        return "given"
    # A line of descent is ordered by generation, and a mythic one has no dates to sort by.
    if spec is not None and spec.kind == "genealogy":
        return "given"
    # A structure is read from the top down, and its order is the hierarchy.
    if spec is not None and spec.kind == "establishment":
        return "given"
    return "date"
