"""Neutral extraction: source material in, structured research record out.

Shape of the thing, and why it is this shape:

  MAP     one call per chunk, and each call sees exactly ONE source. The model is
          never asked which source something came from, because it is never shown
          more than one -- the label is attached here, by code, from the chunk's own
          provenance. Misattribution is therefore not a thing the model can get wrong,
          which is what makes "every item traceable to its source" a property of the
          pipeline rather than a hope about the prompt.

  REDUCE  merge the per-chunk records deterministically. Entities join on a normalized
          key and union their source lists; timeline entries are kept whole and sorted;
          conflicts are DERIVED by comparing what different sources said about the same
          event, not asked for.

  Why map-reduce at all: the previous pipeline ran ONE call capped at 8000 characters
  per source (SYNTHESIS_SAMPLE_CHARS). That is a summary budget. A four-hour transcript is
  hundreds of thousands of characters, so 8k meant most of the material was never read
  -- fine for a blurb, incompatible with a research report that promises completeness.

Nothing here decides whether a source is correct. Corroboration is reported as a COUNT
of which sources carry an item; the reader weighs it. See neutrality.py.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from phansora.shared.ai.deepseek import chat_model

from . import neutrality

# ---------------------------------------------------------------------------
# The per-chunk extraction call
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = neutrality.NEUTRALITY_RULES + """

TASK

You are reading ONE excerpt from ONE source in a research collection. Extract what is
present in this excerpt. Do not speculate, do not fill gaps from your own knowledge, and
do not carry in anything the excerpt does not contain.

If the excerpt contains nothing for a category, return an empty array for it. An empty
array is a correct answer. Inventing entries to fill the shape is not.

Record a person, organization or place if the excerpt NAMES it. Record an event if the
excerpt says something happened. Record a claim as the source's own assertion, in the
source's terms. Record a relationship ONLY where the excerpt states or directly shows it
-- if you find yourself reasoning "they were probably connected because...", that is a
gap, not a relationship, and belongs in gaps.

Return ONLY valid JSON in exactly this shape, no markdown fence and no commentary:
{
  "people": [
    {"name": "Full Name as written",
     "aliases": ["other names/spellings used for this same person in the excerpt"],
     "described_as": "how the excerpt describes them, in its terms, one line"}
  ],
  "organizations": [
    {"name": "...", "aliases": ["..."], "described_as": "how the excerpt describes it"}
  ],
  "places": [
    {"name": "...", "described_as": "what the excerpt says happened or existed there"}
  ],
  "timeline": [
    {"date": "as written, e.g. 'March 12, 1998' or '1998-03-12' or 'spring 1998'",
     "time": "if given, else empty",
     "event": "what the excerpt says occurred"}
  ],
  "events": [
    {"title": "short name for the occurrence",
     "summary": "what the excerpt says happened",
     "date": "as written if given, else empty",
     "people": ["names involved, as written"],
     "organizations": ["..."],
     "places": ["..."]}
  ],
  "claims": [
    {"statement": "an assertion the excerpt makes or reports, in its own terms",
     "attributed_to": "who the EXCERPT says is making it (a person, body, or document); empty if the excerpt asserts it directly"}
  ],
  "documents": [
    {"title": "document, record, exhibit or recording referred to",
     "described_as": "what the excerpt says it is or contains"}
  ],
  "relationships": [
    {"from": "entity name", "relation": "short verb phrase, e.g. 'worked for', 'met with', 'owned'",
     "to": "entity name",
     "basis": "the wording in the excerpt that shows this"}
  ],
  "gaps": [
    {"gap": "something the excerpt raises but does not answer",
     "lead": "what a researcher would need to look at next; empty if unclear"}
  ]
}

Use plain ASCII. Keep every string short and factual."""

# The second, cross-source pass. Runs over the MERGED skeleton rather than raw text --
# by then the per-source facts exist and what is left is the connective work.
_SYNTHESIS_SYSTEM = neutrality.NEUTRALITY_RULES + """

TASK

You are given a structured record already extracted from a research collection: its
entities, timeline, claims and relationships, each carrying the sources it came from.

Write the connective sections. You are describing what THIS COLLECTION contains and how
its parts relate to each other. You are not assessing the subject matter.

Every statement you write must be traceable: name the sources ("Source 2", "Sources 1
and 4"). A sentence you cannot attribute is a sentence to delete.

Return ONLY valid JSON in exactly this shape:
{
  "overview": "3-6 sentences: what this collection consists of and what subject matter it addresses, in the sources' own terms",
  "observations": [
    {"observation": "something notable about the RECORD -- a recurring name, an overlap between sources, a dense cluster of events", "sources": ["Source 1", "..."]}
  ],
  "threads": [
    {"name": "short label for a subject area the collection keeps returning to, in the sources' own terms",
     "summary": "2-4 sentences on what the collection carries under this subject, attributed throughout",
     "entities": ["names copied EXACTLY as they appear in the record above, for the people, organizations and places belonging to this thread"],
     "sources": ["Source 1", "..."]}
  ],
  "cross_source": {
    "corroborated": [{"statement": "an item more than one source carries", "sources": ["Source 1", "Source 3"]}],
    "single_source": [{"statement": "an item only one source carries", "source": "Source 2"}],
    "source_relationships": [{"observation": "how two sources relate -- one cites the other, they cover the same period, they share a subject", "sources": ["Source 1", "Source 2"]}]
  },
  "conflicts": [
    {"topic": "what the sources disagree about",
     "versions": [{"source": "Source 3", "account": "what this source says"}],
     "note": "state that the supplied material does not resolve it; do not resolve it yourself"}
  ],
  "gaps": [
    {"gap": "what the collection does not establish", "lead": "what would need to be consulted"}
  ],
  "synthesis": "4-10 sentences connecting the collection into a coherent account of what it describes, attributed throughout. Do not conclude, do not advise, do not tell the reader what it means."
}

Give 3 to 6 threads, ordered by how much of the collection they carry. A thread's
"entities" must be names that appear in the record above, copied character for character
-- the report resolves them back to the extracted entries, and a name that does not match
one is dropped. Do not invent a thread to round the number up; if the collection only
sustains two, return two.

Use plain ASCII."""


def _extract_json(raw: str) -> Optional[Dict]:
    """Parse JSON from model output, tolerating fences and stray prose."""
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.splitlines() if not l.strip().startswith("```")).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None


def call_neutral(
    client,
    system: str,
    user: str,
    source_labels: Sequence[str] = (),
    temperature: float = 0.1,
    model_var: str = "RESEARCH_ATLAS_MODEL",
) -> Tuple[Optional[Dict], List[neutrality.Violation]]:
    """One model call, with the neutrality rule enforced rather than merely requested.

    Ladder: ask -> check -> ask again pointing at the violations -> redact what survives.
    The retry is worth its cost because a model that judged once usually attributes
    correctly when shown the specific sentence; redaction is the floor that guarantees
    the promise holds even if it does not.

    Returns (parsed, violations_that_survived). Surviving violations have already been
    redacted out of the returned object -- they are handed back so the job can record
    that redaction happened rather than silently losing text.
    """
    def _ask(messages) -> Optional[Dict]:
        try:
            resp = client.chat.completions.create(
                model=chat_model(model_var), messages=messages, temperature=temperature,
            )
            return _extract_json((resp.choices[0].message.content or "").strip())
        except Exception as e:  # noqa: BLE001 -- a failed call must not kill the run
            print(f"[ATLAS] extraction call failed: {e}")
            return None

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    parsed = _ask(messages)
    if parsed is None:
        return None, []

    violations = neutrality.scan_fields(parsed, source_labels)
    if not violations:
        return parsed, []

    print(f"[ATLAS] neutrality: {len(violations)} unattributed characterization(s); retrying")
    retry = _ask(messages + [
        {"role": "assistant", "content": json.dumps(parsed)},
        {"role": "user", "content": neutrality.REWRITE_INSTRUCTION + "\n\nSpecifically:\n"
         + neutrality.summarize(violations)},
    ])
    if retry is not None:
        parsed = retry
        violations = neutrality.scan_fields(parsed, source_labels)
        if not violations:
            return parsed, []

    print(f"[ATLAS] neutrality: {len(violations)} survived the retry; redacting")
    return _redact_fields(parsed, source_labels), violations


def _redact_fields(obj: Any, source_labels: Sequence[str]) -> Any:
    """neutrality.redact() over every string in a nested structure."""
    if isinstance(obj, str):
        return neutrality.redact(obj, source_labels)
    if isinstance(obj, dict):
        return {k: _redact_fields(v, source_labels) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact_fields(v, source_labels) for v in obj]
    return obj


def extract_from_chunk(client, chunk_text: str, source_label: str) -> Optional[Dict]:
    """Extract the structured record from one excerpt of one source."""
    if not (chunk_text or "").strip():
        return None
    user = (
        "This excerpt comes from a single source in the collection.\n\n"
        "--- BEGIN EXCERPT ---\n" + chunk_text.strip() + "\n--- END EXCERPT ---\n\n"
        "Extract the structured record for this excerpt as specified."
    )
    parsed, _ = call_neutral(client, _EXTRACT_SYSTEM, user, (source_label,))
    return parsed


# ---------------------------------------------------------------------------
# Reduce: merge per-chunk records into one traceable record
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def norm_key(value: str) -> str:
    """Join key for entity names. Case, punctuation and spacing are noise here."""
    return _WS.sub(" ", _PUNCT.sub(" ", str(value or "").lower())).strip()


def _tokens(text: str) -> set:
    return {t for t in norm_key(text).split() if len(t) > 3}


def _similar(a: str, b: str, threshold: float = 0.6) -> bool:
    """Jaccard overlap on content words -- enough to spot two sources describing the
    same occurrence in different wording, without pretending to be a real matcher."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= threshold


def _merge_named(bucket: Dict[str, Dict], items: Iterable[Dict], label: str, alias_key: bool) -> None:
    """Fold one chunk's named entities into the running bucket."""
    for item in items or []:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        key = norm_key(name)
        rec = bucket.setdefault(key, {
            "name": name, "aliases": [], "descriptions": [], "sources": [],
        })
        if label not in rec["sources"]:
            rec["sources"].append(label)
        if alias_key:
            for a in item.get("aliases") or []:
                a = str(a).strip()
                if a and norm_key(a) != key and a not in rec["aliases"]:
                    rec["aliases"].append(a)
        desc = str(item.get("described_as") or "").strip()
        if desc and not any(d["text"] == desc for d in rec["descriptions"]):
            # Descriptions stay PER SOURCE rather than being merged into one line.
            # Two sources describing the same person differently is a fact about the
            # record, and flattening it would quietly pick a winner.
            rec["descriptions"].append({"text": desc, "source": label})


def merge_extractions(per_chunk: Sequence[Tuple[str, Dict]]) -> Dict[str, Any]:
    """Combine (source_label, chunk_record) pairs into one research record.

    Everything that comes out of here carries the sources it came from. Corroboration
    is the LENGTH of that source list -- a count, not a rating.
    """
    people: Dict[str, Dict] = {}
    orgs: Dict[str, Dict] = {}
    places: Dict[str, Dict] = {}
    timeline: List[Dict] = []
    events: List[Dict] = []
    claims: List[Dict] = []
    documents: Dict[str, Dict] = {}
    relationships: Dict[Tuple[str, str, str], Dict] = {}
    gaps: List[Dict] = []

    for label, rec in per_chunk:
        if not rec:
            continue
        _merge_named(people, rec.get("people"), label, alias_key=True)
        _merge_named(orgs, rec.get("organizations"), label, alias_key=True)
        _merge_named(places, rec.get("places"), label, alias_key=False)

        for t in rec.get("timeline") or []:
            event = str(t.get("event") or "").strip()
            if not event:
                continue
            timeline.append({
                "date": str(t.get("date") or "").strip(),
                "time": str(t.get("time") or "").strip(),
                "event": event,
                "source": label,
            })

        for e in rec.get("events") or []:
            title = str(e.get("title") or "").strip()
            if not title:
                continue
            existing = next((x for x in events if _similar(x["title"], title)), None)
            if existing:
                if label not in existing["sources"]:
                    existing["sources"].append(label)
                summary = str(e.get("summary") or "").strip()
                if summary and not any(a["text"] == summary for a in existing["accounts"]):
                    existing["accounts"].append({"text": summary, "source": label})
                for field in ("people", "organizations", "places"):
                    for v in e.get(field) or []:
                        if v and v not in existing[field]:
                            existing[field].append(v)
                continue
            events.append({
                "title": title,
                "date": str(e.get("date") or "").strip(),
                "accounts": ([{"text": str(e.get("summary") or "").strip(), "source": label}]
                             if str(e.get("summary") or "").strip() else []),
                "people": list(e.get("people") or []),
                "organizations": list(e.get("organizations") or []),
                "places": list(e.get("places") or []),
                "sources": [label],
            })

        for c in rec.get("claims") or []:
            statement = str(c.get("statement") or "").strip()
            if not statement:
                continue
            existing = next((x for x in claims if _similar(x["statement"], statement, 0.75)), None)
            if existing:
                if label not in existing["sources"]:
                    existing["sources"].append(label)
                continue
            claims.append({
                "statement": statement,
                "attributed_to": str(c.get("attributed_to") or "").strip(),
                "sources": [label],
            })

        for d in rec.get("documents") or []:
            title = str(d.get("title") or "").strip()
            if not title:
                continue
            key = norm_key(title)
            doc = documents.setdefault(key, {"title": title, "descriptions": [], "sources": []})
            if label not in doc["sources"]:
                doc["sources"].append(label)
            desc = str(d.get("described_as") or "").strip()
            if desc and not any(x["text"] == desc for x in doc["descriptions"]):
                doc["descriptions"].append({"text": desc, "source": label})

        for r in rec.get("relationships") or []:
            a, rel, b = (str(r.get(k) or "").strip() for k in ("from", "relation", "to"))
            if not (a and rel and b):
                continue
            key = (norm_key(a), norm_key(rel), norm_key(b))
            link = relationships.setdefault(key, {
                "from": a, "relation": rel, "to": b, "basis": [], "sources": [],
            })
            if label not in link["sources"]:
                link["sources"].append(label)
            basis = str(r.get("basis") or "").strip()
            if basis and not any(x["text"] == basis for x in link["basis"]):
                link["basis"].append({"text": basis, "source": label})

        for g in rec.get("gaps") or []:
            gap = str(g.get("gap") or "").strip()
            if not gap or any(_similar(x["gap"], gap, 0.7) for x in gaps):
                continue
            gaps.append({"gap": gap, "lead": str(g.get("lead") or "").strip(), "source": label})

    return {
        "people": sorted(people.values(), key=lambda r: (-len(r["sources"]), r["name"].lower())),
        "organizations": sorted(orgs.values(), key=lambda r: (-len(r["sources"]), r["name"].lower())),
        "places": sorted(places.values(), key=lambda r: (-len(r["sources"]), r["name"].lower())),
        "timeline": sort_timeline(timeline),
        "events": events,
        "claims": sorted(claims, key=lambda c: -len(c["sources"])),
        "documents": sorted(documents.values(), key=lambda d: (-len(d["sources"]), d["title"].lower())),
        "relationships": sorted(relationships.values(), key=lambda r: -len(r["sources"])),
        "gaps": gaps,
        "date_conflicts": detect_date_conflicts(timeline),
    }


# ---------------------------------------------------------------------------
# Chronology and conflicts, derived rather than asked for
# ---------------------------------------------------------------------------

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}


def date_sort_key(raw: str) -> Tuple[int, int, int, int]:
    """A lenient ordering key for dates as sources actually write them.

    Sources write "March 12, 1998", "1998-03-12", "spring 1998" and "the following
    Tuesday". Anything unparseable sorts to the END rather than being dropped or
    guessed at -- an undated item is still part of the record, and inventing a
    position for it would be a judgment about where it belongs.
    """
    s = str(raw or "").strip().lower()
    if not s:
        return (1, 0, 0, 0)
    iso = re.search(r"(\d{4})-(\d{1,2})(?:-(\d{1,2}))?", s)
    if iso:
        y, m, d = int(iso.group(1)), int(iso.group(2)), int(iso.group(3) or 0)
        return (0, y, m, d)
    year = re.search(r"\b(1[6-9]\d{2}|20\d{2})\b", s)
    month = next((n for name, n in _MONTHS.items() if name in s), 0)
    day = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", s)
    if year:
        return (0, int(year.group(1)), month, int(day.group(1)) if day and month else 0)
    return (1, 0, 0, 0)


def sort_timeline(entries: Sequence[Dict]) -> List[Dict]:
    """Chronological where a date can be read, original order where it cannot."""
    return sorted(entries, key=lambda e: date_sort_key(e.get("date", "")))


def detect_date_conflicts(entries: Sequence[Dict]) -> List[Dict]:
    """Where two sources date the same occurrence differently.

    Derived here rather than asked of the model: a single extraction call sees one
    source and cannot know another disagrees with it, and asking a later call to
    "find the conflicts" invites it to resolve them. Comparing recorded values is
    the one method that cannot pick a winner.
    """
    conflicts: List[Dict] = []
    used: set = set()
    for i, a in enumerate(entries):
        if i in used:
            continue
        group = [a]
        for j, b in enumerate(entries[i + 1:], start=i + 1):
            if j in used or b["source"] == a["source"]:
                continue
            if _similar(a["event"], b["event"], 0.6):
                group.append(b)
                used.add(j)
        if len(group) < 2:
            continue
        dates = {(g.get("date") or "").strip().lower() for g in group if (g.get("date") or "").strip()}
        if len(dates) > 1:
            conflicts.append({
                "event": a["event"],
                "versions": [{"source": g["source"], "date": g.get("date", "")} for g in group],
            })
    return conflicts


def build_connective_sections(client, record: Dict, source_labels: Sequence[str]) -> Tuple[Dict, List]:
    """The cross-source pass: overview, observations, conflicts, gaps, synthesis.

    Fed the merged SKELETON rather than the source text. By this point the per-source
    facts are already extracted and attributed, so what is left is connective work --
    and keeping the raw text out of this call means it cannot quietly introduce a
    detail that no extraction step ever recorded against a source.
    """
    skeleton = {
        "sources": list(source_labels),
        "people": [{"name": p["name"], "sources": p["sources"]} for p in record["people"][:60]],
        "organizations": [{"name": o["name"], "sources": o["sources"]} for o in record["organizations"][:60]],
        "places": [{"name": p["name"], "sources": p["sources"]} for p in record["places"][:60]],
        "timeline": [{"date": t["date"], "event": t["event"], "source": t["source"]} for t in record["timeline"][:120]],
        "claims": [{"statement": c["statement"], "sources": c["sources"]} for c in record["claims"][:80]],
        "relationships": [{"from": r["from"], "relation": r["relation"], "to": r["to"], "sources": r["sources"]}
                          for r in record["relationships"][:60]],
        "date_conflicts": record["date_conflicts"][:30],
        "gaps_noted": [g["gap"] for g in record["gaps"][:30]],
    }
    user = (
        f"The collection has {len(source_labels)} source(s): {', '.join(source_labels)}.\n\n"
        "Structured record already extracted:\n"
        + json.dumps(skeleton, indent=1)[:60000]
        + "\n\nWrite the connective sections as specified."
    )
    parsed, violations = call_neutral(client, _SYNTHESIS_SYSTEM, user, source_labels, temperature=0.2)
    return (parsed or {}), violations
