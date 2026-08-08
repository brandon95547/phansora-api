"""The Research Atlas report: merged record in, two-part dossier out.

Deterministic. Every heading below is always present and always in the same order, and
the body of each is assembled from the extracted record by code -- the model's prose
appears only in the connective sections (Overview, Observations, Threads, Conflicts,
Gaps, Synthesis), and even there it has been through the neutrality check twice before
it arrives.

The document has two parts, and the split is the whole point:

  PART I -- RESEARCH REPORT is what a person reads. It is organized by subject and by
  connection, and it carries only the material that the collection itself carries most
  heavily. Every list in it is capped, and every cap states how many further items are
  in Part II, so a short section is never mistaken for a complete one.

  PART II -- RESEARCH INDEX is what a person looks things up in. Nothing extracted is
  dropped to make Part I readable; it all lands here, exhaustively, in directories that
  can be scanned by name, by date, by source, or by document.

Two consequences worth stating, because they are the point rather than side effects:

  No AI table of contents. The predecessor generated its own chapter structure, which
  meant the shape of the document was itself a model output and varied run to run. The
  sections here are the product's definition of a research report, so they are fixed,
  and a reader who has seen one Atlas report can navigate the next one.

  Empty sections are RENDERED, not dropped. "No documents were recorded in the supplied
  material" is a finding about the collection. Silently omitting the heading would let
  absence look like an oversight, and a researcher cannot tell the difference between
  "nothing was found" and "nobody looked".

IMPORTANCE TAGS (Core / Supporting / Peripheral / Reference) are computed, never judged.
They answer "how much of this collection carries this item" -- a countable property --
and they are defined in one place, `_tier`, so the ladder printed for the reader in
"How to Read This Report" is literally the code that assigned the tags. They say nothing
about whether the material is correct; see neutrality.py.

Source numbering happens here and only here. Extraction carries raw labels (filenames,
URLs) because those are what the pipeline can prove; the report maps them to Source 1..N
for readability, and the Source Index and Appendix I print the mapping both ways so an
item can always be walked back to the file it came from.

ASCII only. The Node PDF renderer downstream strips non-ASCII glyphs, so an arrow here
is "->" and a dash is "--".
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import neutrality
from .extraction import date_sort_key, norm_key

# Part I section titles, in order. Numbered 1..14 in the rendered document.
SECTIONS: Sequence[str] = (
    "Research Overview",
    "Research Map",
    "Major Research Threads",
    "Key People",
    "Key Organizations",
    "Core Timeline",
    "Connection Analysis",
    "Major Claims by Topic",
    "Where the Sources Overlap",
    "Where the Sources Differ",
    "Evidence and Referenced Materials",
    "Unresolved Research Questions",
    "Research Gaps",
    "Research Synthesis",
)

# Part II, lettered. Between them these hold every item in the extracted record.
APPENDICES: Sequence[Tuple[str, str]] = (
    ("A", "Complete People Directory"),
    ("B", "Complete Organizations Directory"),
    ("C", "Complete Places Directory"),
    ("D", "Complete Timeline"),
    ("E", "Complete Events Record"),
    ("F", "Complete Claims Index"),
    ("G", "Complete Document and Evidence Index"),
    ("H", "Complete Connection Index"),
    ("I", "Source-by-Source Extraction"),
    ("J", "Research Leads and Unresolved Items"),
)

TIERS: Sequence[str] = ("Core", "Supporting", "Peripheral", "Reference")

# How much of the record Part I is allowed to show. Every one of these is paired with a
# "N further ... in Appendix X" line at the point it bites, so a cap is visible to the
# reader rather than being a silent truncation.
MAX_THREADS = 6
MAX_KEY_PEOPLE = 12
MAX_KEY_ORGS = 10
MAX_KEY_DESCRIPTIONS = 3
MAX_CONNECTIONS_PER_ENTITY = 6
MAX_CORE_TIMELINE = 45
MAX_CLUSTERS = 8
MAX_CLAIMS_PER_TOPIC = 8
MAX_OVERLAP_ITEMS = 25
MAX_KEY_DOCUMENTS = 25
MAX_QUESTIONS = 15

_EMPTY = "No entries of this kind were recorded in the supplied material."


class SourceMap:
    """Two-way map between a source's real label and its "Source N" shorthand."""

    def __init__(self, labels: Sequence[str]):
        self._order: List[str] = []
        for label in labels:
            label = str(label or "").strip()
            if label and label not in self._order:
                self._order.append(label)

    def num(self, label: str) -> str:
        """"Source 3" for a known label; the raw label for anything unrecognized.

        Falling back to the label rather than dropping it matters: an unmapped item is
        still attributed, just less tidily, and silently emitting an unattributed
        sentence is the one outcome this product cannot have.
        """
        label = str(label or "").strip()
        try:
            return f"Source {self._order.index(label) + 1}"
        except ValueError:
            return label or "unattributed"

    def index(self, label: str) -> int:
        """1-based position, or 0 for a label this map does not know."""
        label = str(label or "").strip()
        try:
            return self._order.index(label) + 1
        except ValueError:
            return 0

    def cite(self, labels: Sequence[str]) -> str:
        """"Sources 1, 3 and 6" / "Source 2" -- the citation as it reads in a sentence.

        Numbered sources collapse to their numbers and sort ascending, so a seven-source
        item cites as "Sources 1, 2, 3, 4, 5, 6 and 7" rather than repeating the word
        seven times. Anything unmapped keeps its raw label, because a clumsy attribution
        beats a missing one.
        """
        seen: List[str] = []
        for l in labels or []:
            l = str(l or "").strip()
            if l and l not in seen:
                seen.append(l)
        if not seen:
            return "unattributed"

        numbered = sorted({self.index(l) for l in seen if self.index(l)})
        unmapped = [l for l in seen if not self.index(l)]
        parts: List[str] = []
        if numbered:
            if len(numbered) == 1:
                parts.append(f"Source {numbered[0]}")
            else:
                head = ", ".join(str(n) for n in numbered[:-1])
                parts.append(f"Sources {head} and {numbered[-1]}")
        parts.extend(unmapped)
        if len(parts) == 1:
            return parts[0]
        return f"{', '.join(parts[:-1])} and {parts[-1]}"

    @property
    def labels(self) -> List[str]:
        return list(self._order)


def _ascii(text: Any) -> str:
    """Fold the characters a model reaches for onto the ASCII the PDF renderer keeps."""
    s = str(text if text is not None else "")
    for bad, good in (
        ("—", "--"), ("–", "-"), ("‘", "'"), ("’", "'"),
        ("“", '"'), ("”", '"'), ("…", "..."), ("→", "->"),
        (" ", " "), ("•", "-"),
    ):
        s = s.replace(bad, good)
    return s.encode("ascii", "ignore").decode("ascii").strip()


def _h1(title: str) -> str:
    return f"\n# {_ascii(title)}\n"


def _h(title: str, index: int) -> str:
    return f"\n## {index}. {_ascii(title)}\n"


def _hA(letter: str, title: str) -> str:
    return f"\n## Appendix {letter} -- {_ascii(title)}\n"


def _h3(title: str) -> str:
    return f"\n### {_ascii(title)}\n"


def _bullets(lines: Sequence[str], empty: str = _EMPTY) -> str:
    lines = [l for l in (_ascii(x) for x in lines) if l]
    return "\n".join(f"- {l}" for l in lines) if lines else empty


def _corroboration(sources: Sequence[str]) -> str:
    """How many sources carry an item, as a count. Deliberately not a rating.

    "Carried by 3 of 5 sources" is a countable property of the collection. "High
    confidence" is a verdict about whether it is correct, which is exactly the call
    this product does not make -- see neutrality.py.
    """
    n = len({s for s in sources or [] if str(s or "").strip()})
    return f"carried by {n} source{'s' if n != 1 else ''}"


def _tier(n_sources: int, mentions: int, has_detail: bool) -> str:
    """The importance ladder, in one place.

    Every input is a count taken from the record:

      n_sources  how many of the supplied sources carry the item
      mentions   how many OTHER places in the record name it -- a relationship, an
                 event, a claim, a timeline entry
      has_detail whether any source described it rather than merely naming it

    Nothing here looks at what the item says, so a tag can never amount to an opinion
    about the material. It is a measure of how much of the collection is built on the
    item, and that is exactly how "How to Read This Report" explains it.
    """
    if n_sources >= 3 or (n_sources >= 2 and mentions >= 1):
        return "Core"
    if n_sources >= 2 or mentions >= 3:
        return "Supporting"
    if has_detail or mentions >= 1:
        return "Peripheral"
    return "Reference"


def _document_tier(n_sources: int) -> str:
    """Documents run on their own ladder, because Reference is what most of them are.

    A record, filing or recording that one source cites is reference material for this
    collection -- it is the thing being pointed at, not a subject the collection is
    built on. Only a document that more than one source reaches for has been leaned on
    by the collection itself, and that is what moves it up.
    """
    if n_sources >= 3:
        return "Core"
    if n_sources == 2:
        return "Supporting"
    return "Reference"


def _more(count: int, noun: str, appendix: str) -> str:
    """The line that keeps a cap honest."""
    if count <= 0:
        return ""
    return (f"\n\n{count} further {noun} {'is' if count == 1 else 'are'} recorded in "
            f"Appendix {appendix}.")


# ---------------------------------------------------------------------------
# Derived indexes over the merged record -- counting only, no judgment
# ---------------------------------------------------------------------------

class Corpus:
    """Cross-references the merged record against itself.

    Everything this class produces is a count of co-occurrence inside the supplied
    material: which entities a claim names, which relationships touch an entity, how
    many sources carry a document. Part I's selection and the importance tags both run
    off these counts, which is what keeps "most important" meaning "most carried by
    this collection" rather than "most interesting to the reader".
    """

    #: Names shorter than this are not matched inside free text -- "Q" or "Uber" as a
    #: substring produces junk hits that would inflate every count built on them.
    MIN_MATCH_LEN = 4

    def __init__(self, record: Dict):
        self.record = record
        self.people = record.get("people") or []
        self.organizations = record.get("organizations") or []
        self.places = record.get("places") or []
        self.timeline = record.get("timeline") or []
        self.events = record.get("events") or []
        self.claims = record.get("claims") or []
        self.documents = record.get("documents") or []
        self.relationships = record.get("relationships") or []
        self.gaps = record.get("gaps") or []

        # key -> every spelling of the entity that is safe to search for in free text
        self._needles: Dict[str, List[str]] = {}
        self.display: Dict[str, str] = {}
        for items in (self.people, self.organizations, self.places):
            for it in items:
                key = norm_key(it.get("name"))
                if not key:
                    continue
                self.display.setdefault(key, str(it.get("name")))
                bucket = self._needles.setdefault(key, [])
                for spelling in [it.get("name")] + list(it.get("aliases") or []):
                    s = str(spelling or "").strip().lower()
                    if len(s) >= self.MIN_MATCH_LEN and s not in bucket:
                        bucket.append(s)

        self.links_by_entity: Dict[str, List[Dict]] = defaultdict(list)
        for link in self.relationships:
            for side in ("from", "to"):
                # A relationship can name an entity the extraction never listed as a
                # person/organization/place. Register its spelling so the connection
                # sections print it as the source wrote it rather than as a lowercased
                # lookup key.
                self.display.setdefault(norm_key(link.get(side)), str(link.get(side) or ""))
                self.links_by_entity[norm_key(link.get(side))].append(link)

        self.events_by_entity: Dict[str, List[Dict]] = defaultdict(list)
        for ev in self.events:
            for field in ("people", "organizations", "places"):
                for name in ev.get(field) or []:
                    self.events_by_entity[norm_key(name)].append(ev)

        self.claims_by_entity: Dict[str, List[int]] = defaultdict(list)
        self.entities_by_claim: Dict[int, List[str]] = defaultdict(list)
        for i, claim in enumerate(self.claims):
            for key in self._match(claim.get("statement")):
                self.claims_by_entity[key].append(i)
                self.entities_by_claim[i].append(key)

        self.timeline_by_entity: Dict[str, List[int]] = defaultdict(list)
        self.entities_by_timeline: Dict[int, List[str]] = defaultdict(list)
        for i, entry in enumerate(self.timeline):
            for key in self._match(entry.get("event")):
                self.timeline_by_entity[key].append(i)
                self.entities_by_timeline[i].append(key)

        self.docs_by_entity: Dict[str, List[Dict]] = defaultdict(list)
        for doc in self.documents:
            text = " ".join([str(doc.get("title") or "")]
                            + [str(d.get("text") or "") for d in doc.get("descriptions") or []])
            for key in self._match(text):
                self.docs_by_entity[key].append(doc)

        # Tags, computed once so Part I's selection and Part II's directories agree.
        #
        # Two maps, deliberately. A tag printed next to an ENTRY has to match the
        # citation printed beside it: Comet Ping Pong is carried by four sources as an
        # organization and by one as a place, and "[Core] (Source 6)" on the place entry
        # reads as a mistake. So the printed tag is per entry. Centrality, on the other
        # hand, is a property of the SUBJECT however it was filed, and that is what
        # decides whether a claim naming it belongs in Part I -- so core_keys is by name.
        self.tier_by_entry: Dict[int, str] = {}
        self.entity_tier: Dict[str, str] = {}
        self.entity_mentions: Dict[str, int] = {}
        for items in (self.people, self.organizations, self.places):
            for it in items:
                key = norm_key(it.get("name"))
                mentions = (len(self.links_by_entity.get(key, []))
                            + len(self.events_by_entity.get(key, []))
                            + len(self.claims_by_entity.get(key, []))
                            + len(self.timeline_by_entity.get(key, [])))
                self.entity_mentions[key] = mentions
                tier = _tier(len(set(it.get("sources") or [])), mentions,
                             bool(it.get("descriptions")))
                self.tier_by_entry[id(it)] = tier
                if TIERS.index(tier) < TIERS.index(self.entity_tier.get(key, "Reference")):
                    self.entity_tier[key] = tier
                else:
                    self.entity_tier.setdefault(key, tier)

        self.core_keys = {k for k, t in self.entity_tier.items() if t == "Core"}

    def _match(self, text: Any) -> List[str]:
        """Which recorded entities are named in this piece of text."""
        t = str(text or "").lower()
        if not t:
            return []
        return [key for key, needles in self._needles.items()
                if any(n in t for n in needles)]

    # -- tags -------------------------------------------------------------

    def tier_of(self, name: Any) -> str:
        """The strongest tag any entry for this NAME earned -- centrality of a subject."""
        return self.entity_tier.get(norm_key(name), "Reference")

    def tier_of_entry(self, entity: Dict) -> str:
        """The tag for THIS entry, matching the sources cited next to it."""
        return self.tier_by_entry.get(id(entity)) or self.tier_of(entity.get("name"))

    def claim_tier(self, index: int, claim: Dict) -> str:
        core_named = sum(1 for k in self.entities_by_claim.get(index, []) if k in self.core_keys)
        return _tier(len(set(claim.get("sources") or [])), core_named,
                     bool(claim.get("attributed_to")))

    def event_tier(self, event: Dict) -> str:
        named = len(event.get("people") or []) + len(event.get("organizations") or []) \
            + len(event.get("places") or [])
        return _tier(len(set(event.get("sources") or [])), named, bool(event.get("accounts")))

    def document_tier(self, doc: Dict) -> str:
        return _document_tier(len(set(doc.get("sources") or [])))

    # -- ordering ---------------------------------------------------------

    def rank(self, entity: Dict) -> Tuple[int, int, str]:
        """Sort key: most-carried first, then most cross-referenced, then by name."""
        key = norm_key(entity.get("name"))
        return (-len(set(entity.get("sources") or [])),
                -self.entity_mentions.get(key, 0),
                str(entity.get("name") or "").lower())

    def top(self, items: Sequence[Dict], tier: Optional[str] = None,
            limit: Optional[int] = None) -> List[Dict]:
        picked = [i for i in items if tier is None or self.tier_of_entry(i) == tier]
        picked.sort(key=self.rank)
        return picked[:limit] if limit else picked

    def related(self, name: Any, limit: int = MAX_CONNECTIONS_PER_ENTITY) -> List[Dict]:
        """The recorded links that touch this entity, most-carried first."""
        links = self.links_by_entity.get(norm_key(name), [])
        return sorted(links, key=lambda l: -len(set(l.get("sources") or [])))[:limit]

    def neighbours(self, name: Any, limit: int = 8) -> List[str]:
        """Entity names on the other end of a recorded link from this one."""
        key = norm_key(name)
        out: List[str] = []
        for link in self.links_by_entity.get(key, []):
            other = link["to"] if norm_key(link["from"]) == key else link["from"]
            if other not in out:
                out.append(other)
        return out[:limit]


# ---------------------------------------------------------------------------
# Threads: the recurring subject areas Part I is organized around
# ---------------------------------------------------------------------------

def _resolve_threads(corpus: Corpus, connective: Dict, sm: SourceMap) -> List[Dict]:
    """The subjects the collection keeps returning to.

    Preferred shape comes from the connective pass, which can name a thread the way a
    reader would ("The Podesta Emails") rather than by its busiest entity. Whatever it
    names, the MEMBERS are resolved back to entities that exist in the record -- a
    thread cannot introduce a subject the extraction never recorded.

    With no usable model output, threads fall back to the Core entities themselves,
    most-carried first, each carrying the entities it is linked to. That is the same
    organizing principle, just labelled by entity name.
    """
    threads: List[Dict] = []
    used: set = set()

    for t in (connective or {}).get("threads") or []:
        name = _ascii(t.get("name"))
        if not name:
            continue
        keys: List[str] = []
        for raw in t.get("entities") or []:
            key = norm_key(raw)
            if key and key in corpus.entity_tier and key not in keys:
                keys.append(key)
        if not keys:
            continue
        threads.append({
            "name": name,
            "summary": _ascii(t.get("summary")),
            "keys": keys,
            # What a claim has to name to be filed under this thread. For a model
            # thread that is its whole membership, because the membership IS the
            # thread's definition.
            "claim_keys": keys,
            "sources": list(t.get("sources") or []),
        })
        used.update(keys)
        if len(threads) >= MAX_THREADS:
            break

    if threads:
        return threads

    for entity in corpus.top(list(corpus.people) + list(corpus.organizations)
                             + list(corpus.places), tier="Core", limit=MAX_THREADS * 3):
        key = norm_key(entity.get("name"))
        if key in used:
            continue
        keys = [key] + [norm_key(n) for n in corpus.neighbours(entity.get("name"), 6)]
        threads.append({
            "name": str(entity.get("name")),
            "summary": "",
            "keys": [k for k in dict.fromkeys(keys) if k in corpus.entity_tier],
            # A fallback thread is named for ONE subject, so only that subject may
            # pull a claim into it. Matching on the linked subjects too would file a
            # claim about a neighbour under a heading that does not name it.
            "claim_keys": [key],
            "sources": list(entity.get("sources") or []),
            "anchor": key,
        })
        used.add(key)
        if len(threads) >= MAX_THREADS:
            break
    return threads


def _thread_sources(thread: Dict, corpus: Corpus) -> List[str]:
    """Which sources carry the thread itself.

    Unioning every member's sources was the obvious implementation and it was wrong:
    a thread whose members include one entity from each source cites all of them, and
    "carried by all seven sources" then means nothing. The citation is the sources that
    carry the thread's OWN subjects -- the ones named in the heading.
    """
    labels: List[str] = list(thread.get("sources") or [])
    for items in (corpus.people, corpus.organizations, corpus.places):
        for it in items:
            if norm_key(it.get("name")) in (thread.get("claim_keys") or thread["keys"]):
                for s in it.get("sources") or []:
                    if s not in labels:
                        labels.append(s)
    return labels


# ---------------------------------------------------------------------------
# Part I
# ---------------------------------------------------------------------------

def _front_matter(record: Dict, corpus: Corpus, sm: SourceMap,
                  profiles_by_label: Dict[str, Any], title: str) -> str:
    forms: Dict[str, int] = defaultdict(int)
    for label in sm.labels:
        form = _ascii(getattr(profiles_by_label.get(label), "source_type", "") or "")
        if form and form.lower() != "unknown":
            forms[form.lower()] += 1
    # Only worth printing when the forms were actually profiled -- "7 unspecified form"
    # is noise dressed as a fact.
    form_summary = ", ".join(f"{n} {name}" for name, n in sorted(forms.items(), key=lambda kv: -kv[1])) \
        if sum(forms.values()) == len(sm.labels) else ""

    counts = ", ".join(f"{len(record.get(k) or [])} {n}" for k, n in (
        ("people", "people"), ("organizations", "organizations"), ("places", "places"),
        ("timeline", "timeline entries"), ("events", "events"), ("claims", "claims"),
        ("documents", "documents"), ("relationships", "connections")))

    return "\n".join([
        f"# {_ascii(title)}",
        "",
        "_Structured research report_",
        "",
        f"- Sources analyzed: {len(sm.labels)}" + (f" ({form_summary})" if form_summary else ""),
        f"- Extracted record: {counts}",
        "- Report purpose: organize the supplied material into a readable map of people, "
        "events, claims, documents, connections and unresolved questions, with every item "
        "attributed to the source that carried it.",
    ])


def _how_to_read() -> str:
    return "\n".join([
        _h1("How to Read This Report"),
        "This document is in two parts.",
        "",
        "**Part I -- Research Report** is organized by subject and by connection. It carries "
        "the material that the supplied collection itself carries most heavily. Every list in "
        "Part I is capped, and each cap states how many further items are held in Part II, so "
        "a short section here never means a short record.",
        "",
        "**Part II -- Research Index** holds the complete extracted material: every person, "
        "organization, place, dated entry, event, claim, document, connection and research "
        "lead, together with a source-by-source breakdown. Nothing extracted was dropped to "
        "make Part I readable.",
        "",
        _h3("Attribution"),
        "Sources are numbered in the order they were supplied, and the numbering is printed "
        "both ways in the Research Overview and in Appendix I so any item can be walked back "
        "to the file it came from. Every statement in this report belongs to a source; where "
        "two sources describe the same subject differently, both descriptions are kept and "
        "neither is preferred.",
        "",
        "This report does not assess whether the supplied material is accurate. It records "
        "what the sources say, who said it, and how the parts relate to each other.",
        "",
        _h3("Importance tags"),
        "Items are tagged by how much of the collection carries them. The tags are counted, "
        "not assessed, and say nothing about the accuracy of the material:",
        "",
        "- **Core** -- carried by three or more sources, or carried by two or more sources and "
        "cross-referenced elsewhere in the record (in a connection, event, claim or dated entry).",
        "- **Supporting** -- carried by two or more sources, or carried by one source and "
        "cross-referenced three or more times elsewhere in the record.",
        "- **Peripheral** -- carried by one source, with a description or a single "
        "cross-reference.",
        "- **Reference** -- recorded by name only, with nothing further attached to it in the "
        "supplied material.",
        "",
        "Documents and referred-to material run on a shorter ladder, because a record that one "
        "source cites is reference material for this collection rather than a subject it is "
        "built on: **Core** at three or more sources, **Supporting** at two, **Reference** at "
        "one.",
        "",
        _h3("Corroboration"),
        "Where an item shows \"carried by 3 sources\", that is a count of the supplied sources "
        "that contain it. It is not a rating.",
    ])


def _overview_section(record: Dict, corpus: Corpus, connective: Dict, sm: SourceMap,
                      profiles_by_label: Dict[str, Any]) -> str:
    parts = [
        _ascii((connective or {}).get("overview"))
        or "The supplied material has been organized below. No overview text was produced.",
        _h3("Sources in this collection"),
        _source_index(record, sm, profiles_by_label),
    ]
    observations = [f"{_ascii(o.get('observation'))} ({sm.cite(o.get('sources') or [])})"
                    for o in (connective or {}).get("observations") or []]
    parts.append(_h3("Key research observations"))
    parts.append(_bullets(observations,
                          empty="No cross-source observations were produced for this collection."))
    return "\n".join(parts)


def _research_map(corpus: Corpus, threads: Sequence[Dict], sm: SourceMap) -> str:
    """The shape of the collection: what sits at the centre and what runs off it."""
    central = corpus.top(list(corpus.people) + list(corpus.organizations) + list(corpus.places),
                         tier="Core", limit=8)
    parts = [_h3("Central subjects"),
             "The subjects the largest number of supplied sources carry:",
             "",
             _bullets([f"**{_ascii(e['name'])}** ({sm.cite(e['sources'])}, "
                       f"{_corroboration(e['sources'])})" for e in central])]

    parts.append(_h3("Research threads"))
    if threads:
        parts.append("The collection runs along the following threads. Not every source "
                     "covers every thread.")
        parts.append("")
        chain = []
        for i, t in enumerate(threads, start=1):
            labels = _thread_sources(t, corpus)
            chain.append(f"{i}. {_ascii(t['name'])} ({sm.cite(labels)})")
        parts.append("\n".join(chain))
        parts.append("")
        parts.append("Each thread is set out in section 3, and the connections between them "
                     "in section 7.")
    else:
        parts.append("No recurring subject drew together enough of the supplied material to "
                     "form a thread.")

    everything = list(corpus.people) + list(corpus.organizations) + list(corpus.places)
    unlinked = sum(1 for it in everything if not corpus.links_by_entity.get(norm_key(it["name"])))
    total = len(everything)
    # Counted per ENTRY, the same way section 13 counts them -- two different totals for
    # "how many are Core" in one document is a question the reader should not have to ask.
    core_entries = sum(1 for it in everything if corpus.tier_of_entry(it) == "Core")
    parts.append(_h3("Density of the record"))
    parts.append(_bullets([
        f"{core_entries} of {total} recorded subjects are tagged Core.",
        f"{unlinked} of {total} recorded subjects carry no recorded connection to another "
        f"subject in the supplied material.",
        f"{len(corpus.relationships)} connections were extracted between named subjects.",
    ]))
    return "\n".join(parts)


def _threads_section(corpus: Corpus, threads: Sequence[Dict], sm: SourceMap) -> str:
    if not threads:
        return ("No recurring subject drew together enough of the supplied material to form "
                "a thread. The complete record is in Part II.")

    out: List[str] = []
    for i, thread in enumerate(threads, start=1):
        labels = _thread_sources(thread, corpus)
        out.append(_h3(f"3.{i} {thread['name']}"))
        if thread.get("summary"):
            out.append(thread["summary"])
            out.append("")

        members = [(items_name, it)
                   for items_name, items in (("person", corpus.people),
                                             ("organization", corpus.organizations),
                                             ("place", corpus.places))
                   for it in items if norm_key(it["name"]) in thread["keys"]]
        members.sort(key=lambda kv: corpus.rank(kv[1]))

        anchor = thread.get("anchor")
        if anchor and not thread.get("summary"):
            entity = next((it for _, it in members if norm_key(it["name"]) == anchor), None)
            if entity and entity.get("descriptions"):
                out.append("How the sources describe this subject:")
                out.append("")
                out.append(_bullets([f"{sm.num(d['source'])} describes: {_ascii(d['text'])}"
                                     for d in entity["descriptions"][:MAX_KEY_DESCRIPTIONS]]))
                out.append("")

        if members:
            out.append("**Subjects in this thread**")
            out.append("")
            out.append(_bullets([
                f"**{_ascii(it['name'])}** -- {kind}, {corpus.tier_of_entry(it)} "
                f"({sm.cite(it['sources'])})" for kind, it in members[:10]]))
            out.append("")

        links = [l for l in corpus.relationships
                 if norm_key(l["from"]) in thread["keys"] and norm_key(l["to"]) in thread["keys"]]
        if links:
            out.append("**Connections recorded within this thread**")
            out.append("")
            out.append(_bullets([
                f"{_ascii(l['from'])} -> {_ascii(l['relation'])} -> {_ascii(l['to'])} "
                f"({sm.cite(l['sources'])})"
                for l in sorted(links, key=lambda l: -len(set(l["sources"])))[:8]]))
            out.append("")

        docs: List[Dict] = []
        for key in thread["keys"]:
            for d in corpus.docs_by_entity.get(key, []):
                if d not in docs:
                    docs.append(d)
        if docs:
            out.append("**Material referred to in this thread**")
            out.append("")
            out.append(_bullets([f"{_ascii(d['title'])} ({sm.cite(d['sources'])})"
                                 for d in docs[:8]]))
            out.append("")

        claim_count = len({i for key in (thread.get("claim_keys") or thread["keys"])
                           for i in corpus.claims_by_entity.get(key, [])})
        out.append(f"Sources contributing to this thread: {sm.cite(labels)}. "
                   f"{claim_count} recorded claim{'s' if claim_count != 1 else ''} "
                   f"name{'' if claim_count != 1 else 's'} a subject in this thread; "
                   f"they are grouped in section 8 and listed in full in Appendix F.")
    return "\n".join(out)


def _key_people_section(corpus: Corpus, sm: SourceMap) -> str:
    people = corpus.top(corpus.people, tier="Core", limit=MAX_KEY_PEOPLE)
    if not people:
        people = corpus.top(corpus.people, limit=MAX_KEY_PEOPLE)
    if not people:
        return _EMPTY

    out = ["The people the supplied collection carries most heavily. The complete directory "
           "of every person extracted is Appendix A.", ""]
    for p in people:
        out.append(_h3(_ascii(p["name"])))
        out.append(_bullets([
            f"Importance: {corpus.tier_of_entry(p)}",
            f"Referenced by: {sm.cite(p['sources'])} ({_corroboration(p['sources'])})",
        ] + ([f"Also referred to as: {_ascii(', '.join(p['aliases']))}"] if p.get("aliases") else [])))
        descs = p.get("descriptions") or []
        if descs:
            out.append("")
            out.append("How the sources describe this person:")
            out.append("")
            out.append(_bullets([f"{sm.num(d['source'])} describes: {_ascii(d['text'])}"
                                 for d in descs[:MAX_KEY_DESCRIPTIONS]]))
            if len(descs) > MAX_KEY_DESCRIPTIONS:
                out.append("")
                out.append(f"{len(descs) - MAX_KEY_DESCRIPTIONS} further description"
                           f"{'s' if len(descs) - MAX_KEY_DESCRIPTIONS != 1 else ''} "
                           f"in Appendix A.")
        links = corpus.related(p["name"])
        if links:
            out.append("")
            out.append("Recorded connections:")
            out.append("")
            out.append(_bullets([
                f"{_ascii(l['from'])} -> {_ascii(l['relation'])} -> {_ascii(l['to'])} "
                f"({sm.cite(l['sources'])})" for l in links]))
        out.append("")
    remainder = len(corpus.people) - len(people)
    out.append(_more(remainder, "people", "A").strip())
    return "\n".join(out)


def _key_orgs_section(corpus: Corpus, sm: SourceMap) -> str:
    orgs = corpus.top(corpus.organizations, tier="Core", limit=MAX_KEY_ORGS)
    if not orgs:
        orgs = corpus.top(corpus.organizations, limit=MAX_KEY_ORGS)
    if not orgs:
        return _EMPTY

    out = ["The organizations and groups the supplied collection carries most heavily. The "
           "complete directory is Appendix B.", ""]
    for o in orgs:
        out.append(_h3(_ascii(o["name"])))
        lines = [
            f"Importance: {corpus.tier_of_entry(o)}",
            f"Referenced by: {sm.cite(o['sources'])} ({_corroboration(o['sources'])})",
        ]
        if o.get("aliases"):
            lines.append(f"Also referred to as: {_ascii(', '.join(o['aliases']))}")
        for d in (o.get("descriptions") or [])[:2]:
            lines.append(f"{sm.num(d['source'])} describes: {_ascii(d['text'])}")
        neighbours = corpus.neighbours(o["name"], 6)
        if neighbours:
            lines.append(f"Connected in the record to: {_ascii(', '.join(neighbours))}")
        out.append(_bullets(lines))
        out.append("")
    remainder = len(corpus.organizations) - len(orgs)
    out.append(_more(remainder, "organizations", "B").strip())
    return "\n".join(out)


def _core_timeline_section(corpus: Corpus, sm: SourceMap) -> str:
    """Dated entries that touch a Core subject, in one chronological run.

    Selection is by connection to the rest of the record, not by what the entry says.
    Everything else -- including every undated entry -- is preserved in Appendix D.
    """
    if not corpus.timeline:
        return _EMPTY

    scored: List[Tuple[int, int, Dict]] = []
    for i, entry in enumerate(corpus.timeline):
        core_named = sum(1 for k in corpus.entities_by_timeline.get(i, []) if k in corpus.core_keys)
        if core_named:
            scored.append((core_named, i, entry))
    if not scored:
        scored = [(0, i, e) for i, e in enumerate(corpus.timeline)]

    scored.sort(key=lambda t: (-t[0], date_sort_key(t[2].get("date", ""))))
    kept = scored[:MAX_CORE_TIMELINE]
    kept.sort(key=lambda t: date_sort_key(t[2].get("date", "")))

    out = ["The dated entries that connect to the subjects in section 3. The complete "
           "timeline, including undated entries, is Appendix D.", ""]
    current_year: Optional[str] = None
    for _, _, e in kept:
        parsed = date_sort_key(e.get("date", ""))
        year = str(parsed[1]) if parsed[0] == 0 and parsed[1] else "Undated"
        if year != current_year:
            out.append(_h3(year))
            current_year = year
        date = _ascii(e.get("date")) or "(undated)"
        time = f" {_ascii(e.get('time'))}" if _ascii(e.get("time")) else ""
        out.append(f"- **{date}{time}** -- {_ascii(e['event'])} ({sm.num(e['source'])})")

    remainder = len(corpus.timeline) - len(kept)
    out.append(_more(remainder, "timeline entries", "D").strip())
    return "\n".join(out)


def _connection_analysis(corpus: Corpus, sm: SourceMap) -> str:
    if not corpus.relationships:
        return ("No connections between named subjects were recorded in the supplied "
                "material.")

    degree = sorted(
        ((len(links), corpus.display.get(key, key)) for key, links in corpus.links_by_entity.items() if key),
        key=lambda t: (-t[0], t[1].lower()))[:12]

    out = [_h3("Most connected subjects"),
           _bullets([f"**{_ascii(name)}** -- {n} recorded connection{'s' if n != 1 else ''}"
                     for n, name in degree])]

    chains = _build_chains(corpus.relationships, limit=MAX_CLUSTERS)
    out.append(_h3("Connection clusters"))
    if chains:
        out.append("Each cluster below is a run of separately recorded links that join up. "
                   "No link was added to complete a chain.")
        out.append("")
        for i, chain in enumerate(chains):
            letter = chr(ord("A") + i)
            steps = [_ascii(chain[0]["from"])]
            labels: List[str] = []
            for link in chain:
                steps.append(f"-> {_ascii(link['relation'])} -> {_ascii(link['to'])}")
                for s in link.get("sources") or []:
                    if s not in labels:
                        labels.append(s)
            out.append(f"**Cluster {letter}**")
            out.append("")
            out.append(f"- {' '.join(steps)}")
            out.append(f"- Carried by: {sm.cite(labels)}")
            out.append("")
    else:
        out.append("No recorded link joined onto another to form a chain.")

    out.append(_h3("Where the threads meet"))
    cross = [l for l in corpus.relationships
             if norm_key(l["from"]) in corpus.core_keys and norm_key(l["to"]) in corpus.core_keys]
    out.append(_bullets(
        [f"{_ascii(l['from'])} -> {_ascii(l['relation'])} -> {_ascii(l['to'])} "
         f"({sm.cite(l['sources'])})"
         for l in sorted(cross, key=lambda l: -len(set(l["sources"])))[:15]],
        empty="No recorded connection ran between two Core subjects."))
    out.append(_more(max(0, len(corpus.relationships) - len(cross)), "connections", "H").strip())
    return "\n".join(out)


def _claims_by_topic(corpus: Corpus, threads: Sequence[Dict], sm: SourceMap) -> str:
    """Claims grouped by the subject they name, rather than listed end to end.

    A claim is filed under the first thread whose subjects it names, so it appears once.
    Claims that name no thread subject are counted and left to Appendix F -- inventing a
    topic for them would be organizing by interpretation rather than by what they say.
    """
    if not corpus.claims:
        return _EMPTY

    assigned: Dict[int, int] = {}
    for ti, thread in enumerate(threads):
        for key in thread.get("claim_keys") or thread["keys"]:
            for ci in corpus.claims_by_entity.get(key, []):
                assigned.setdefault(ci, ti)

    out = ["Claims are grouped by the subject they name and are recorded as the assertions "
           "of the sources that carry them. The complete claims index is Appendix F.", ""]

    for ti, thread in enumerate(threads):
        indexes = sorted((ci for ci, t in assigned.items() if t == ti),
                         key=lambda ci: -len(set(corpus.claims[ci].get("sources") or [])))
        out.append(_h3(f"Claims naming {thread['name']}"))
        if not indexes:
            out.append("No claim in the supplied material named a subject in this thread.")
            out.append("")
            continue
        out.append(_bullets([_claim_line(corpus, sm, ci) for ci in indexes[:MAX_CLAIMS_PER_TOPIC]]))
        if len(indexes) > MAX_CLAIMS_PER_TOPIC:
            out.append("")
            out.append(f"{len(indexes) - MAX_CLAIMS_PER_TOPIC} further claims naming this "
                       f"subject are listed in Appendix F.")
        out.append("")

    unassigned = [i for i in range(len(corpus.claims)) if i not in assigned]
    out.append(_h3("Claims recorded outside these subjects"))
    if unassigned:
        top = sorted(unassigned, key=lambda ci: -len(set(corpus.claims[ci].get("sources") or [])))
        multi = [ci for ci in top if len(set(corpus.claims[ci].get("sources") or [])) > 1]
        out.append(f"{len(unassigned)} recorded claims name none of the subjects above. "
                   f"{len(multi)} of them are carried by more than one source:")
        out.append("")
        out.append(_bullets([_claim_line(corpus, sm, ci) for ci in (multi or top)[:MAX_CLAIMS_PER_TOPIC]],
                            empty="All of them are carried by a single source."))
        out.append("")
        out.append("The full list is in Appendix F.")
    else:
        out.append("Every recorded claim names a subject listed above.")
    return "\n".join(out)


def _claim_line(corpus: Corpus, sm: SourceMap, index: int) -> str:
    c = corpus.claims[index]
    who = _ascii(c.get("attributed_to"))
    lead = f"{sm.cite(c['sources'])} " + ("records that " if not who
                                          else f"records {who} as stating that ")
    return (f"{lead}{_ascii(c['statement'])} [{corpus.claim_tier(index, c)}, "
            f"{_corroboration(c['sources'])}]")


def _overlap_section(corpus: Corpus, connective: Dict, sm: SourceMap) -> str:
    everything = list(corpus.people) + list(corpus.organizations) + list(corpus.places)
    strong = [e for e in everything if len(set(e["sources"])) >= 3]
    secondary = [e for e in everything if len(set(e["sources"])) == 2]
    strong.sort(key=corpus.rank)
    secondary.sort(key=corpus.rank)

    out = [_h3("Subjects carried by three or more sources"),
           _bullets([f"**{_ascii(e['name'])}** ({sm.cite(e['sources'])})"
                     for e in strong[:MAX_OVERLAP_ITEMS]],
                    empty="No subject in the supplied material was carried by three or more sources.")]
    if len(strong) > MAX_OVERLAP_ITEMS:
        out.append("")
        out.append(f"{len(strong) - MAX_OVERLAP_ITEMS} further subjects in this band are in "
                   f"Appendices A to C.")

    out.append(_h3("Subjects carried by exactly two sources"))
    out.append(_bullets([f"**{_ascii(e['name'])}** ({sm.cite(e['sources'])})"
                         for e in secondary[:MAX_OVERLAP_ITEMS]],
                        empty="No subject in the supplied material was carried by exactly two sources."))
    if len(secondary) > MAX_OVERLAP_ITEMS:
        out.append("")
        out.append(f"{len(secondary) - MAX_OVERLAP_ITEMS} further subjects in this band are in "
                   f"Appendices A to C.")

    cs = (connective or {}).get("cross_source") or {}
    out.append(_h3("Statements carried by more than one source"))
    out.append(_bullets([f"{_ascii(c.get('statement'))} ({sm.cite(c.get('sources') or [])})"
                         for c in cs.get("corroborated") or []],
                        empty="No cross-source statement overlap was recorded."))

    out.append(_h3("How the sources relate to each other"))
    out.append(_bullets([f"{_ascii(r.get('observation'))} ({sm.cite(r.get('sources') or [])})"
                         for r in cs.get("source_relationships") or []],
                        empty="No relationship between the supplied sources was recorded."))

    out.append(_h3("Material concentrated in a single source"))
    per_source: List[str] = []
    for label in sm.labels:
        only = [e for e in everything if set(e["sources"]) == {label}]
        if not only:
            continue
        only.sort(key=corpus.rank)
        examples = ", ".join(_ascii(e["name"]) for e in only[:6])
        per_source.append(f"{sm.num(label)} is the only source carrying {len(only)} recorded "
                          f"subjects, including: {examples}")
    out.append(_bullets(per_source,
                        empty="Every recorded subject is carried by more than one source."))
    single = cs.get("single_source") or []
    if single:
        out.append("")
        out.append(_bullets([f"{_ascii(c.get('statement'))} ({sm.num(c.get('source'))})"
                             for c in single]))
    return "\n".join(out)


def _differ_section(connective: Dict, record: Dict, sm: SourceMap) -> str:
    """Disagreements, preserved. Nothing here picks a side.

    Two origins, deliberately kept together: date conflicts derived mechanically by
    comparing what each source recorded, and substantive conflicts the cross-source
    pass identified. The mechanical ones cannot be talked out of existence, which is
    why they are computed rather than requested.
    """
    rows: List[str] = []
    for c in record.get("date_conflicts") or []:
        versions = "; ".join(f"{sm.num(v['source'])} gives {_ascii(v['date']) or '(no date)'}"
                             for v in c["versions"])
        rows.append(f"**Date of: {_ascii(c['event'])}** -- {versions}. "
                    "The supplied material does not resolve this difference.")

    for c in (connective or {}).get("conflicts") or []:
        versions = "\n".join(f"  - {sm.num(v.get('source'))}: {_ascii(v.get('account'))}"
                             for v in c.get("versions") or [])
        note = _ascii(c.get("note")) or "The supplied material does not resolve this difference."
        rows.append(f"**{_ascii(c.get('topic'))}**\n{versions}\n  - {note}")

    return "\n".join(f"- {r}" for r in rows) if rows else \
        "No differences between the supplied sources were recorded."


def _evidence_section(corpus: Corpus, sm: SourceMap) -> str:
    # Documents attached to a Core subject come first. Without that the section is
    # alphabetical, and in a collection where every document is single-sourced that
    # means Part I's evidence is whatever happens to start with "A".
    core_docs = {id(d) for key in corpus.core_keys for d in corpus.docs_by_entity.get(key, [])}
    docs = [d for d in corpus.documents if corpus.document_tier(d) in ("Core", "Supporting")]
    if len(docs) < MAX_KEY_DOCUMENTS:
        docs += [d for d in corpus.documents
                 if d not in docs and (id(d) in core_docs or d.get("descriptions"))]
    docs.sort(key=lambda d: (-len(set(d.get("sources") or [])),
                             0 if id(d) in core_docs else 1,
                             str(d.get("title", "")).lower()))
    shown = docs[:MAX_KEY_DOCUMENTS]
    if not shown:
        return _EMPTY

    out = ["Documents, records, recordings and other material the sources refer to. The "
           "complete index is Appendix G.", ""]
    rows = []
    for d in shown:
        line = (f"**{_ascii(d['title'])}** [{corpus.document_tier(d)}] "
                f"({sm.cite(d['sources'])})")
        for desc in (d.get("descriptions") or [])[:2]:
            line += f"\n  - {sm.num(desc['source'])} describes: {_ascii(desc['text'])}"
        rows.append(line)
    out.append("\n".join(f"- {r}" for r in rows))
    out.append(_more(len(corpus.documents) - len(shown), "documents", "G").strip())
    return "\n".join(out)


def _questions_section(connective: Dict, corpus: Corpus, sm: SourceMap) -> str:
    """The unanswered questions worth carrying to the front of the report.

    Collection-level questions from the connective pass come first because they are
    about the record as a whole; per-source questions follow, most-connected first.
    Everything is repeated in full in Appendix J.
    """
    rows: List[str] = []
    for g in (connective or {}).get("gaps") or []:
        line = _ascii(g.get("gap"))
        if not line:
            continue
        if g.get("lead"):
            line += f"\n  - Needed research: {_ascii(g['lead'])}"
        rows.append(line)

    with_leads = [g for g in corpus.gaps if g.get("lead")]
    without = [g for g in corpus.gaps if not g.get("lead")]
    for g in with_leads + without:
        if len(rows) >= MAX_QUESTIONS:
            break
        line = f"{_ascii(g['gap'])} ({sm.num(g.get('source'))})"
        if g.get("lead"):
            line += f"\n  - Needed research: {_ascii(g['lead'])}"
        rows.append(line)

    if not rows:
        return "No unanswered questions were recorded from the supplied material."
    out = "\n".join(f"- {r}" for r in rows[:MAX_QUESTIONS])
    remaining = len(corpus.gaps) + len((connective or {}).get("gaps") or []) - min(len(rows), MAX_QUESTIONS)
    return out + _more(max(0, remaining), "recorded questions and leads", "J")


def _research_gaps_section(corpus: Corpus, sm: SourceMap) -> str:
    """What the SHAPE of the collection leaves open, as counts.

    Distinct from section 12: that section carries questions the sources themselves
    raise. This one is about the record's own weighting -- which source the material
    rests on, how much of it stands on a single source, what is undated. Those are
    properties a researcher needs before reading anything else, and every one of them
    is arithmetic over the record.
    """
    everything = list(corpus.people) + list(corpus.organizations) + list(corpus.places)
    total = len(everything)
    rows: List[str] = []

    contributions: List[Tuple[int, str]] = []
    for label in sm.labels:
        n = (sum(1 for e in everything if label in (e.get("sources") or []))
             + sum(1 for t in corpus.timeline if t.get("source") == label)
             + sum(1 for c in corpus.claims if label in (c.get("sources") or []))
             + sum(1 for d in corpus.documents if label in (d.get("sources") or [])))
        contributions.append((n, label))
    grand = sum(n for n, _ in contributions) or 1
    contributions.sort(key=lambda t: -t[0])
    rows.append("Share of the extracted record contributed by each source: " + "; ".join(
        f"{sm.num(label)} {round(100 * n / grand)}%" for n, label in contributions))

    if len(contributions) > 1 and contributions[0][0] > 0:
        top_n, top_label = contributions[0]
        rows.append(f"{sm.num(top_label)} contributes the largest share of the record "
                    f"({round(100 * top_n / grand)}%). Subjects that appear only through it "
                    f"are in the record because one source introduced them, not because the "
                    f"collection converges on them.")

    single = sum(1 for e in everything if len(set(e.get("sources") or [])) == 1)
    if total:
        rows.append(f"{single} of {total} recorded subjects ({round(100 * single / total)}%) are "
                    f"carried by a single source.")

    undated = sum(1 for t in corpus.timeline if date_sort_key(t.get("date", ""))[0] != 0)
    if corpus.timeline:
        rows.append(f"{undated} of {len(corpus.timeline)} timeline entries carry no date that "
                    f"could be placed in sequence; they are held at the end of Appendix D.")

    unattributed = sum(1 for c in corpus.claims if not str(c.get("attributed_to") or "").strip())
    if corpus.claims:
        rows.append(f"{unattributed} of {len(corpus.claims)} recorded claims are asserted by "
                    f"the source directly rather than attributed by it to a named person, body "
                    f"or document.")

    unlinked = [e for e in everything if not corpus.links_by_entity.get(norm_key(e["name"]))]
    if total:
        rows.append(f"{len(unlinked)} of {total} recorded subjects carry no recorded connection "
                    f"to another subject. A connection missing here is material the supplied "
                    f"sources did not state, not a connection that was ruled out.")

    tier_counts = defaultdict(int)
    for e in everything:
        tier_counts[corpus.tier_of_entry(e)] += 1
    rows.append("Recorded subjects by importance tag: " + ", ".join(
        f"{tier_counts.get(t, 0)} {t}" for t in TIERS))

    rows.append("These counts describe how the supplied collection is weighted. They are not "
                "an assessment of the material.")
    return _bullets(rows)


# ---------------------------------------------------------------------------
# Part II
# ---------------------------------------------------------------------------

def _people_directory(corpus: Corpus, sm: SourceMap) -> str:
    rows = []
    for p in corpus.people:
        head = (f"**{_ascii(p['name'])}** [{corpus.tier_of_entry(p)}] "
                f"({sm.cite(p['sources'])}, {_corroboration(p['sources'])})")
        if p.get("aliases"):
            head += f"\n  - Also referred to as: {_ascii(', '.join(p['aliases']))}"
        for d in p.get("descriptions") or []:
            head += f"\n  - {sm.num(d['source'])} describes: {_ascii(d['text'])}"
        rows.append(head)
    return "\n".join(f"- {r}" for r in rows) if rows else _EMPTY


def _orgs_directory(corpus: Corpus, sm: SourceMap) -> str:
    rows = []
    for o in corpus.organizations:
        head = (f"**{_ascii(o['name'])}** [{corpus.tier_of_entry(o)}] "
                f"({sm.cite(o['sources'])}, {_corroboration(o['sources'])})")
        if o.get("aliases"):
            head += f"\n  - Also referred to as: {_ascii(', '.join(o['aliases']))}"
        for d in o.get("descriptions") or []:
            head += f"\n  - {sm.num(d['source'])} describes: {_ascii(d['text'])}"
        rows.append(head)
    return "\n".join(f"- {r}" for r in rows) if rows else _EMPTY


def _places_directory(corpus: Corpus, sm: SourceMap) -> str:
    rows = []
    for p in corpus.places:
        head = (f"**{_ascii(p['name'])}** [{corpus.tier_of_entry(p)}] "
                f"({sm.cite(p['sources'])})")
        for d in p.get("descriptions") or []:
            head += f"\n  - {sm.num(d['source'])} records: {_ascii(d['text'])}"
        rows.append(head)
    return "\n".join(f"- {r}" for r in rows) if rows else _EMPTY


def _timeline_appendix(record: Dict, sm: SourceMap) -> str:
    """One chronological sequence across every source.

    Entries whose date could not be parsed sort to the end and are labelled as
    undated rather than being given a guessed position -- placing them would be an
    assertion the sources did not make.
    """
    entries = record.get("timeline") or []
    if not entries:
        return _EMPTY
    lines = ["| Date | Time | Event | Source |", "| --- | --- | --- | --- |"]
    for e in entries:
        date = _ascii(e.get("date")) or "(undated)"
        time = _ascii(e.get("time")) or "-"
        lines.append(f"| {date} | {time} | {_ascii(e['event'])} | {sm.num(e['source'])} |")
    out = "\n".join(lines)

    conflicts = record.get("date_conflicts") or []
    if conflicts:
        out += ("\n\nThe sources give differing dates for the following. The supplied "
                "material does not resolve these differences.\n")
        for c in conflicts:
            versions = "; ".join(
                f"{sm.num(v['source'])} gives {_ascii(v['date']) or '(no date)'}" for v in c["versions"]
            )
            out += f"\n- {_ascii(c['event'])}: {versions}"
    return out


def _events_appendix(corpus: Corpus, sm: SourceMap) -> str:
    rows = []
    for e in corpus.events:
        head = f"**{_ascii(e['title'])}**"
        if e.get("date"):
            head += f" -- {_ascii(e['date'])}"
        head += f" [{corpus.event_tier(e)}] ({sm.cite(e['sources'])})"
        for a in e.get("accounts") or []:
            head += f"\n  - {sm.num(a['source'])} records: {_ascii(a['text'])}"
        for field, tag in (("people", "People"), ("organizations", "Organizations"),
                           ("places", "Places")):
            if e.get(field):
                head += f"\n  - {tag} named: {_ascii(', '.join(e[field]))}"
        rows.append(head)
    return "\n".join(f"- {r}" for r in rows) if rows else _EMPTY


def _claims_appendix(corpus: Corpus, sm: SourceMap) -> str:
    if not corpus.claims:
        return _EMPTY
    order = sorted(range(len(corpus.claims)),
                   key=lambda i: -len(set(corpus.claims[i].get("sources") or [])))
    return "\n".join(f"- {_claim_line(corpus, sm, i)}" for i in order)


def _documents_appendix(corpus: Corpus, sm: SourceMap) -> str:
    rows = []
    for d in corpus.documents:
        head = f"**{_ascii(d['title'])}** [{corpus.document_tier(d)}] ({sm.cite(d['sources'])})"
        for desc in d.get("descriptions") or []:
            head += f"\n  - {sm.num(desc['source'])} describes: {_ascii(desc['text'])}"
        rows.append(head)
    return "\n".join(f"- {r}" for r in rows) if rows else _EMPTY


def _connections_appendix(corpus: Corpus, sm: SourceMap) -> str:
    """Every connection the sources support, plus the chains they form.

    The chain view is the one the product brief asks for -- Person A -> worked for
    Organization B -> which operated from Location C. It is assembled ONLY from links
    that were separately extracted and attributed; nothing is bridged because it would
    read well. A missing hop stays missing and turns up under Research Leads.
    """
    links = corpus.relationships
    if not links:
        return _EMPTY
    rows = []
    for r in links:
        line = (f"**{_ascii(r['from'])}** -> {_ascii(r['relation'])} -> **{_ascii(r['to'])}** "
                f"({sm.cite(r['sources'])})")
        for b in r.get("basis") or []:
            line += f"\n  - {sm.num(b['source'])} basis: {_ascii(b['text'])}"
        rows.append(line)
    out = "\n".join(f"- {r}" for r in rows)

    chains = _build_chains(links)
    if chains:
        out += "\n\n**Chains formed by the links above**\n"
        for chain in chains:
            steps = [f"{_ascii(chain[0]['from'])}"]
            for link in chain:
                steps.append(f"-> {_ascii(link['relation'])} -> {_ascii(link['to'])}")
            out += "\n- " + " ".join(steps)
    return out


def _build_chains(links: Sequence[Dict], max_len: int = 4, limit: int = 12) -> List[List[Dict]]:
    """Walk extracted links into multi-hop paths.

    Purely mechanical: a chain exists when the "to" of one attributed link is the
    "from" of another. No link is created here, so a chain can never assert a
    connection the sources did not carry -- it only shows that several of them join up.

    Selection is deliberately spread out. The busiest entity in a research record is
    joined to everything, so a straight "longest first" pick returned twelve chains
    that all began with the same hop -- twelve restatements of one connection, printed
    as twelve findings. A chain is therefore only kept if its opening link has not
    already opened a kept chain, which surfaces twelve different parts of the graph.
    """
    by_from: Dict[str, List[Dict]] = {}
    for l in links:
        by_from.setdefault(norm_key(l["from"]), []).append(l)

    # Walked per STARTING LINK with its own small quota, so a hub entity cannot fill the
    # candidate pool before the rest of the graph has been looked at.
    pool_size = max(limit * 20, 200)
    per_start = 4
    chains: List[List[Dict]] = []
    for start in links:
        if len(chains) >= pool_size:
            break
        found = 0
        stack = [[start]]
        while stack and found < per_start:
            path = stack.pop()
            if len(path) >= max_len:
                continue
            tail = norm_key(path[-1]["to"])
            visited = {norm_key(p["from"]) for p in path} | {norm_key(p["to"]) for p in path[:-1]}
            for nxt in by_from.get(tail, []):
                if norm_key(nxt["to"]) in visited:
                    continue  # a cycle is not a chain
                extended = path + [nxt]
                if len(extended) >= 2:
                    chains.append(extended)
                    found += 1
                stack.append(extended)

    def sig(chain: Sequence[Dict]) -> Tuple:
        return tuple((norm_key(l["from"]), norm_key(l["to"])) for l in chain)

    # Longest first, and only the distinct ones -- a 3-hop chain contains a 2-hop one.
    chains.sort(key=len, reverse=True)
    kept: List[List[Dict]] = []
    openings: set = set()

    def _take(require_new_opening: bool) -> None:
        for c in chains:
            if len(kept) >= limit:
                return
            s = sig(c)
            if any(s == sig(k)[:len(s)] for k in kept) or c in kept:
                continue
            if require_new_opening and s[0] in openings:
                continue
            kept.append(c)
            openings.add(s[0])

    _take(require_new_opening=True)
    # A sparse graph may not have `limit` distinct openings. Rather than print three
    # clusters when eight distinct chains exist, fill the remainder without the
    # opening constraint -- prefix dedup still keeps them from being restatements.
    _take(require_new_opening=False)
    return kept


def _source_by_source(corpus: Corpus, sm: SourceMap, profiles_by_label: Dict[str, Any]) -> str:
    """What each source contributed, source by source, in full.

    The same items appear in the directories above; here they are cut the other way, so
    a reader can ask "what did this one file actually carry" without filtering a
    thousand-line index by eye.
    """
    if not sm.labels:
        return _EMPTY
    out: List[str] = []
    for i, label in enumerate(sm.labels, start=1):
        out.append(_h3(f"Source {i} -- {_ascii(label)}"))
        prof = profiles_by_label.get(label)
        form = _ascii(getattr(prof, "source_type", "") or "")
        header = []
        if form and form.lower() != "unknown":
            header.append(f"Form: {form}")
        counts = _counts_for_label(corpus.record, label)
        if counts:
            header.append(f"Contributes: {counts}")
        if header:
            out.append(_bullets(header))
            out.append("")

        for title, items in (
            ("People named", [p for p in corpus.people if label in (p.get("sources") or [])]),
            ("Organizations named", [o for o in corpus.organizations if label in (o.get("sources") or [])]),
            ("Places named", [p for p in corpus.places if label in (p.get("sources") or [])]),
        ):
            out.append(f"**{title}** ({len(items)})")
            out.append("")
            out.append(_ascii(", ".join(x["name"] for x in items)) or
                       "None recorded from this source.")
            out.append("")

        entries = [t for t in corpus.timeline if t.get("source") == label]
        out.append(f"**Timeline entries** ({len(entries)})")
        out.append("")
        out.append(_bullets([f"{_ascii(t.get('date')) or '(undated)'} -- {_ascii(t['event'])}"
                             for t in entries], empty="None recorded from this source."))
        out.append("")

        events = [e for e in corpus.events if label in (e.get("sources") or [])]
        out.append(f"**Events** ({len(events)})")
        out.append("")
        out.append(_bullets([f"{_ascii(e['title'])}"
                             + (f" -- {_ascii(e['date'])}" if e.get("date") else "")
                             for e in events], empty="None recorded from this source."))
        out.append("")

        claims = [c for c in corpus.claims if label in (c.get("sources") or [])]
        out.append(f"**Claims recorded** ({len(claims)})")
        out.append("")
        out.append(_bullets([
            (f"{_ascii(c['statement'])}"
             + (f" -- attributed by this source to {_ascii(c['attributed_to'])}"
                if c.get("attributed_to") else ""))
            for c in claims], empty="None recorded from this source."))
        out.append("")

        docs = [d for d in corpus.documents if label in (d.get("sources") or [])]
        out.append(f"**Documents and material referred to** ({len(docs)})")
        out.append("")
        out.append(_bullets([_ascii(d["title"]) for d in docs],
                            empty="None recorded from this source."))
        out.append("")

        gaps = [g for g in corpus.gaps if g.get("source") == label]
        out.append(f"**Questions this source leaves open** ({len(gaps)})")
        out.append("")
        out.append(_bullets([_ascii(g["gap"]) for g in gaps],
                            empty="None recorded from this source."))
        out.append("")
    return "\n".join(out)


def _leads_appendix(connective: Dict, corpus: Corpus, sm: SourceMap) -> str:
    rows: List[str] = []
    for g in corpus.gaps:
        line = f"{_ascii(g['gap'])} ({sm.num(g.get('source'))})"
        if g.get("lead"):
            line += f"\n  - Possible next step: {_ascii(g['lead'])}"
        rows.append(line)
    for g in (connective or {}).get("gaps") or []:
        line = _ascii(g.get("gap"))
        if not line:
            continue
        if g.get("lead"):
            line += f"\n  - Possible next step: {_ascii(g['lead'])}"
        rows.append(line)
    return "\n".join(f"- {r}" for r in rows) if rows else \
        "No unanswered questions were recorded from the supplied material."


def _source_index(record: Dict, sm: SourceMap, profiles_by_label: Dict[str, Any]) -> str:
    rows = []
    for i, label in enumerate(sm.labels, start=1):
        line = f"**Source {i}** -- {_ascii(label)}"
        prof = profiles_by_label.get(label)
        form = _ascii(getattr(prof, "source_type", "") or "")
        if form and form.lower() != "unknown":
            # The document's FORM (transcript, article, filing), which is descriptive.
            # No assessment of the source accompanies it, by design.
            line += f"\n  - Form: {form}"
        counts = _counts_for_label(record, label)
        if counts:
            line += f"\n  - Contributes: {counts}"
        rows.append(line)
    return "\n".join(f"- {r}" for r in rows) if rows else _EMPTY


def _counts_for_label(record: Dict, label: str) -> str:
    """What this source contributed, as counts -- a map of where material came from."""
    def n_named(key):
        return sum(1 for x in record.get(key) or [] if label in (x.get("sources") or []))
    bits = [
        (n_named("people"), "people"),
        (n_named("organizations"), "organizations"),
        (n_named("places"), "places"),
        (sum(1 for t in record.get("timeline") or [] if t.get("source") == label), "timeline entries"),
        (n_named("claims"), "claims"),
        (n_named("documents"), "documents"),
    ]
    return ", ".join(f"{c} {name}" for c, name in bits if c)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def render_report(
    record: Dict,
    connective: Dict,
    source_labels: Sequence[str],
    title: str = "Research Atlas Report",
    profiles: Sequence[Any] = (),
) -> str:
    """The two-part dossier, as ASCII Markdown."""
    sm = SourceMap(source_labels)
    profiles_by_label = {getattr(p, "source_label", ""): p for p in (profiles or [])}
    c = connective or {}
    corpus = Corpus(record)
    threads = _resolve_threads(corpus, c, sm)

    body: List[str] = [_front_matter(record, corpus, sm, profiles_by_label, title)]
    body.append(_how_to_read())

    body.append(_h1("PART I -- RESEARCH REPORT"))
    body.append("The readable report. Every list below is capped and says where the rest of "
                "the material is held in Part II.")

    body.append(_h(SECTIONS[0], 1))
    body.append(_overview_section(record, corpus, c, sm, profiles_by_label))

    body.append(_h(SECTIONS[1], 2));  body.append(_research_map(corpus, threads, sm))
    body.append(_h(SECTIONS[2], 3));  body.append(_threads_section(corpus, threads, sm))
    body.append(_h(SECTIONS[3], 4));  body.append(_key_people_section(corpus, sm))
    body.append(_h(SECTIONS[4], 5));  body.append(_key_orgs_section(corpus, sm))
    body.append(_h(SECTIONS[5], 6));  body.append(_core_timeline_section(corpus, sm))
    body.append(_h(SECTIONS[6], 7));  body.append(_connection_analysis(corpus, sm))
    body.append(_h(SECTIONS[7], 8));  body.append(_claims_by_topic(corpus, threads, sm))
    body.append(_h(SECTIONS[8], 9));  body.append(_overlap_section(corpus, c, sm))
    body.append(_h(SECTIONS[9], 10)); body.append(_differ_section(c, record, sm))
    body.append(_h(SECTIONS[10], 11)); body.append(_evidence_section(corpus, sm))
    body.append(_h(SECTIONS[11], 12)); body.append(_questions_section(c, corpus, sm))
    body.append(_h(SECTIONS[12], 13)); body.append(_research_gaps_section(corpus, sm))

    body.append(_h(SECTIONS[13], 14))
    body.append(_ascii(c.get("synthesis")) or
                "No synthesis text was produced for the supplied material.")

    body.append(_h1("PART II -- RESEARCH INDEX"))
    body.append("The complete extracted material, for lookup and reference. Items carry the "
                "importance tag defined in How to Read This Report; the tag is a count of how "
                "much of the collection carries the item.")

    renderers = {
        "A": lambda: _people_directory(corpus, sm),
        "B": lambda: _orgs_directory(corpus, sm),
        "C": lambda: _places_directory(corpus, sm),
        "D": lambda: _timeline_appendix(record, sm),
        "E": lambda: _events_appendix(corpus, sm),
        "F": lambda: _claims_appendix(corpus, sm),
        "G": lambda: _documents_appendix(corpus, sm),
        "H": lambda: _connections_appendix(corpus, sm),
        "I": lambda: _source_by_source(corpus, sm, profiles_by_label),
        "J": lambda: _leads_appendix(c, corpus, sm),
    }
    for letter, name in APPENDICES:
        body.append(_hA(letter, name))
        body.append(renderers[letter]())

    return "\n".join(body).strip() + "\n"


def audit_report(markdown: str, source_labels: Sequence[str]) -> List[neutrality.Violation]:
    """Final gate: scan the assembled document for anything judging in its own voice.

    Runs on the rendered text rather than the record, so it also catches phrasing the
    renderer itself introduces. A clean pass here is the claim the product makes.
    """
    return neutrality.scan(markdown, source_labels, path="report")
