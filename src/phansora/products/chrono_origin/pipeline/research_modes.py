"""The five kinds of research the start screen offers, and the prompt each one sends.

Every kind runs the same pipeline as a trace: one grounded call in which the model runs
its own searches and answers with a JSON array, read by ``dated_list`` and laid on the
board. What differs is the question, and the question is the whole product: a
chronology asks for an ORDER, a parallel for a COMPARISON, a translation for WORDING, a
piece of evidence for a STANCE. Changing the heading on the same trace prompt would have
answered all five with the same lineage.

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
    "bodily_spiritual": (
        "Bodily & Spiritual",
        "readings that take the account as symbolism of the body, the mind or the inner "
        "life: allegory, initiation, psychological or spiritual development",
    ),
    "solar_astronomical": (
        "Solar & Astronomical",
        "readings that tie the account to celestial cycles: the sun's year, the moon, the "
        "planets, constellations, equinoxes, solstices and the calendar",
    ),
    "historical": (
        "Historical",
        "the historical events, people, places and dates behind the subject, as historians "
        "and archaeologists reconstruct them",
    ),
    "comparative": (
        "Comparative",
        "related accounts in other traditions, cultures, periods or sources, set beside "
        "each point in the sequence",
    ),
}

ORIGINS_PERSPECTIVES = {
    "historical": ("Historical", "documented events, dates and people"),
    "textual": ("Textual", "the earliest writings, manuscripts and editions"),
    "archaeological": ("Archaeological & Material", "physical remains, artifacts and sites"),
    "linguistic": ("Linguistic", "words, names and their roots"),
    "comparative": ("Comparative", "related developments in other cultures"),
}

PARALLEL_PERSPECTIVES = {
    "historical": ("Historical", "documented contact, borrowing and influence"),
    "literary": ("Literary", "plot, structure, motifs and narrative patterns"),
    "symbolic": ("Symbolic", "shared symbols, imagery and meanings"),
    "cultural": ("Cultural", "social role, function and setting"),
    "natural": ("Natural & Astronomical", "seasonal, natural or celestial cycles"),
}

ORIGINS_FOCUS = {
    "stories": (
        "Focus on the story or tradition itself: its earliest tellings, the older stories, "
        "accounts and motifs it grew from, and how it was retold and changed."
    ),
    "beliefs": (
        "Focus on the belief, idea or interpretation: where it was first stated, the "
        "earlier ideas it developed from, who held it, and how its meaning changed."
    ),
    "words": (
        "Focus on the word or name: its etymology, its earliest attestations, the older "
        "words and roots it descends from, its cognates in related languages, and how its "
        "form and meaning shifted. Inscriptions, glossaries and dictionaries are primary "
        "material here."
    ),
    "practices": (
        "Focus on the practice or object: its earliest physical and written evidence, the "
        "earlier techniques, tools, customs and materials it came from, and how it spread "
        "and changed."
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

EVIDENCE_SOURCE_TYPES = {
    "texts": "texts and manuscripts: literary, religious, scientific or legal writings and "
             "their surviving copies",
    "records": "historical records: administrative, legal, financial and census records, "
               "correspondence, chronicles, newspapers and official reports",
    "inscriptions": "inscriptions and artifacts: inscriptions, coins, seals, monuments, "
                    "excavated objects and other physical remains",
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
def origins_addendum(spec: Optional[ResearchSpec]) -> str:
    """What Origins adds after the hand-tuned trace prompt. Empty with nothing chosen."""
    if spec is None:
        return ""
    parts: List[str] = []
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
        "HISTORICAL ORDER. Order the entries by historical date, oldest first. Give each "
        "date as the evidence supports it and mark its uncertainty in \"certainty\": "
        "established, approximate, disputed, or traditional (held by a tradition but not "
        "established by evidence)."
    ),
    "symbolic": (
        "SYMBOLIC ORDER. Order the entries by the stages or cycles of the interpretation "
        "itself: the phases, stations, initiations, seasons or cycles each perspective "
        "describes, in the order it describes them, not by historical date."
    ),
}


def chronology_prompt(title: str, context: Optional[str], spec: ResearchSpec) -> str:
    perspectives = [p for p in spec.perspectives if p in CHRONOLOGY_PERSPECTIVES] or ["historical"]
    sequence = spec.sequence if spec.sequence in _SEQUENCE_RULES else "historical"
    grouped = spec.layout == "grouped" and len(perspectives) > 1
    labels = [CHRONOLOGY_PERSPECTIVES[p][0] for p in perspectives]
    lines = "\n".join(
        f"* {CHRONOLOGY_PERSPECTIVES[p][0]}: {CHRONOLOGY_PERSPECTIVES[p][1]}"
        for p in perspectives
    )
    subject = _subject(title, context)

    if grouped:
        shape_rule = (
            "ONE ENTRY PER STAGE of the sequence. Each entry gives, in \"readings\", what "
            "each perspective says about that stage, keyed by the perspective's name "
            f"exactly as written here: {_listed(labels)}. Leave a perspective out of a "
            "stage where it says nothing about it, rather than inventing a reading."
        )
        readings_field = ',\n    "readings": {"' + labels[0] + '": "What this perspective says about the stage"}'
        perspective_field = ""
    else:
        shape_rule = (
            "ONE ENTRY PER EVENT PER PERSPECTIVE. Name the perspective each entry belongs "
            f"to in \"perspective\", exactly as written here: {_listed(labels)}. Each "
            "perspective's entries must make a complete sequence on their own."
        )
        readings_field = ""
        perspective_field = f',\n    "perspective": "{labels[0]}"'

    return f"""\
Using live web search, build the chronology of {subject} from these perspectives:

{lines}

{_SEQUENCE_RULES[sequence]}

{shape_rule}

Search separately for each perspective. Report each interpretation as the people who hold it hold it, naming whose reading it is: do not present an interpretation as historical fact, and do not dismiss it. Where the historical record contradicts a reading, say so in that entry.

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
    "significance": "What happens, or what this stage means in this perspective"{perspective_field}{readings_field},
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


# ------------------------------------------------------------------------ evidence
def evidence_prompt(title: str, context: Optional[str], spec: ResearchSpec) -> str:
    types = [t for t in spec.source_types if t in EVIDENCE_SOURCE_TYPES]
    if types:
        kinds = "\n".join(f"* {EVIDENCE_SOURCE_TYPES[t]}" for t in types)
        scope = f"Look in these kinds of source:\n\n{kinds}"
    else:
        kinds = "\n".join(f"* {d}" for d in EVIDENCE_SOURCE_TYPES.values())
        scope = f"Look in every kind of source:\n\n{kinds}\n* modern scholarship that weighs them"

    return f"""\
Using live web search, find the evidence bearing on {_subject(title, context)}. It may be a subject or a specific claim, from any field or period.

{scope}

Search for sources that CHALLENGE it as hard as for sources that SUPPORT it. Then search for its LIMITATIONS: late or single copies, unknown provenance, suspected forgery or interpolation, dating disputes, silence where evidence would be expected, and disagreement among specialists.

Do not decide the question. Report what each source shows, when it was made, and how reliable it is, and classify each in "stance":

* supports: the source is evidence for it
* challenges: the source is evidence against it, or for an alternative
* limitation: a gap, dispute or weakness in the evidence itself

Date each source to when the SOURCE was made, not to what it describes.

The goal is a large, complete list with rich detail, not a brief summary. Include every qualifying source found, rather than only the strongest.

Return the result as a JSON array and nothing else, one object per source, oldest first:

[
  {{
    "title": "The source, by its usual name",
    "date": "When the source was made, e.g. c. 700 BCE, 1947 CE",
    "stance": "supports, challenges or limitation",
    "source_type": "text, manuscript, record, inscription, artifact or scholarship",
    "origin": "Where it was made, found or is now held",
    "shows": "What it shows about the subject or claim",
    "reliability": "Its provenance, dating and how far it can be trusted",
    "authorship": "Author, issuer or finder",
    "significance": "Why it matters to the question",
    "is_collection": false
  }}
]

{_JSON_RULES}"""


# ------------------------------------------------------------------------ dispatch
_BUILDERS = {
    "chronology": chronology_prompt,
    "parallels": parallels_prompt,
    "translations": translations_prompt,
    "evidence": evidence_prompt,
}


def prompt_for(title: str, context: Optional[str], spec: Optional[ResearchSpec]) -> Optional[str]:
    """The prompt for a non-Origins request, or None when the trace prompt applies."""
    if spec is None or spec.kind == "origins":
        return None
    return _BUILDERS[spec.kind](title, context, spec)


def ordering_for(spec: Optional[ResearchSpec]) -> str:
    """"given" when the research's own order is the finding and a date sort would undo it."""
    if spec is not None and spec.kind == "chronology" and spec.sequence in ("narrative", "symbolic"):
        return "given"
    return "date"


def perspective_label(spec: Optional[ResearchSpec], value: str) -> str:
    """A perspective as the model wrote it, matched back to the label the request used."""
    value = (value or "").strip()
    if not value or spec is None:
        return value
    table = CHRONOLOGY_PERSPECTIVES if spec.kind == "chronology" else {}
    for pid, (label, _) in table.items():
        if value.casefold() in (label.casefold(), pid.casefold(), pid.replace("_", " ").casefold()):
            return label
    return value
