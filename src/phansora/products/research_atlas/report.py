"""The Research Atlas report: merged record in, sixteen fixed sections out.

Deterministic. Every heading below is always present and always in the same order, and
the body of each is assembled from the extracted record by code -- the model's prose
appears only in the three connective sections (Overview, Observations, Synthesis), and
even there it has been through the neutrality check twice before it arrives.

Two consequences worth stating, because they are the point rather than side effects:

  No AI table of contents. The predecessor generated its own chapter structure, which
  meant the shape of the document was itself a model output and varied run to run. The
  sections here are the product's definition of a research report, so they are fixed,
  and a reader who has seen one Atlas report can navigate the next one.

  Empty sections are RENDERED, not dropped. "No documents were recorded in the supplied
  material" is a finding about the collection. Silently omitting the heading would let
  absence look like an oversight, and a researcher cannot tell the difference between
  "nothing was found" and "nobody looked".

Source numbering happens here and only here. Extraction carries raw labels (filenames,
URLs) because those are what the pipeline can prove; the report maps them to Source 1..N
for readability, and the Source Index and References print the mapping both ways so an
item can always be walked back to the file it came from.

ASCII only. The Node PDF renderer downstream strips non-ASCII glyphs, so an arrow here
is "->" and a dash is "--".
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from . import neutrality

SECTIONS: Sequence[str] = (
    "Research Overview",
    "Key Research Observations",
    "Source Index",
    "People",
    "Organizations and Groups",
    "Places",
    "Master Timeline",
    "Events",
    "Claims and Accounts",
    "Documents and Supporting Material",
    "Connections and Relationships",
    "Cross-Source Analysis",
    "Conflicts and Contradictions",
    "Information Gaps and Research Leads",
    "Research Synthesis",
    "References",
)

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

    def cite(self, labels: Sequence[str]) -> str:
        """"Sources 1 and 3" / "Source 2" -- the citation as it reads in a sentence."""
        nums = [self.num(l) for l in labels or [] if str(l or "").strip()]
        seen: List[str] = []
        for n in nums:
            if n not in seen:
                seen.append(n)
        if not seen:
            return "unattributed"
        if len(seen) == 1:
            return seen[0]
        joined = ", ".join(seen[:-1])
        return f"{joined.replace('Source ', 'Sources ', 1)} and {seen[-1].replace('Source ', '')}"

    @property
    def labels(self) -> List[str]:
        return list(self._order)


def _ascii(text: Any) -> str:
    """Fold the characters a model reaches for onto the ASCII the PDF renderer keeps."""
    s = str(text if text is not None else "")
    for bad, good in (
        ("—", "--"), ("–", "-"), ("‘", "'"), ("’", "'"),
        ("“", '"'), ("”", '"'), ("…", "..."), ("→", "->"),
        (" ", " "), ("•", "-"),
    ):
        s = s.replace(bad, good)
    return s.encode("ascii", "ignore").decode("ascii").strip()


def _h(title: str, index: int) -> str:
    return f"\n## {index}. {_ascii(title)}\n"


def _bullets(lines: Sequence[str]) -> str:
    lines = [l for l in (_ascii(x) for x in lines) if l]
    return "\n".join(f"- {l}" for l in lines) if lines else _EMPTY


def _corroboration(sources: Sequence[str]) -> str:
    """How many sources carry an item, as a count. Deliberately not a rating.

    "Carried by 3 of 5 sources" is a countable property of the collection. "High
    confidence" is a verdict about whether it is correct, which is exactly the call
    this product does not make -- see neutrality.py.
    """
    n = len({s for s in sources or [] if str(s or "").strip()})
    return f"carried by {n} source{'s' if n != 1 else ''}"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _people_section(record: Dict, sm: SourceMap) -> str:
    rows = []
    for p in record.get("people") or []:
        head = f"**{_ascii(p['name'])}** ({sm.cite(p['sources'])}, {_corroboration(p['sources'])})"
        if p.get("aliases"):
            head += f"\n  - Also referred to as: {_ascii(', '.join(p['aliases']))}"
        for d in p.get("descriptions") or []:
            head += f"\n  - {sm.num(d['source'])} describes: {_ascii(d['text'])}"
        rows.append(head)
    return "\n".join(f"- {r}" for r in rows) if rows else _EMPTY


def _orgs_section(record: Dict, sm: SourceMap) -> str:
    rows = []
    for o in record.get("organizations") or []:
        head = f"**{_ascii(o['name'])}** ({sm.cite(o['sources'])})"
        if o.get("aliases"):
            head += f"\n  - Also referred to as: {_ascii(', '.join(o['aliases']))}"
        for d in o.get("descriptions") or []:
            head += f"\n  - {sm.num(d['source'])} describes: {_ascii(d['text'])}"
        rows.append(head)
    return "\n".join(f"- {r}" for r in rows) if rows else _EMPTY


def _places_section(record: Dict, sm: SourceMap) -> str:
    rows = []
    for p in record.get("places") or []:
        head = f"**{_ascii(p['name'])}** ({sm.cite(p['sources'])})"
        for d in p.get("descriptions") or []:
            head += f"\n  - {sm.num(d['source'])} records: {_ascii(d['text'])}"
        rows.append(head)
    return "\n".join(f"- {r}" for r in rows) if rows else _EMPTY


def _timeline_section(record: Dict, sm: SourceMap) -> str:
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


def _events_section(record: Dict, sm: SourceMap) -> str:
    rows = []
    for e in record.get("events") or []:
        head = f"**{_ascii(e['title'])}**"
        if e.get("date"):
            head += f" -- {_ascii(e['date'])}"
        head += f" ({sm.cite(e['sources'])})"
        for a in e.get("accounts") or []:
            head += f"\n  - {sm.num(a['source'])} records: {_ascii(a['text'])}"
        for field, tag in (("people", "People"), ("organizations", "Organizations"), ("places", "Places")):
            if e.get(field):
                head += f"\n  - {tag} named: {_ascii(', '.join(e[field]))}"
        rows.append(head)
    return "\n".join(f"- {r}" for r in rows) if rows else _EMPTY


def _claims_section(record: Dict, sm: SourceMap) -> str:
    rows = []
    for c in record.get("claims") or []:
        who = _ascii(c.get("attributed_to"))
        lead = f"{sm.cite(c['sources'])} " + ("records that " if not who else f"records {who} as stating that ")
        rows.append(f"{lead}{_ascii(c['statement'])} ({_corroboration(c['sources'])})")
    return _bullets(rows)


def _documents_section(record: Dict, sm: SourceMap) -> str:
    rows = []
    for d in record.get("documents") or []:
        head = f"**{_ascii(d['title'])}** ({sm.cite(d['sources'])})"
        for desc in d.get("descriptions") or []:
            head += f"\n  - {sm.num(desc['source'])} describes: {_ascii(desc['text'])}"
        rows.append(head)
    return "\n".join(f"- {r}" for r in rows) if rows else _EMPTY


def _relationships_section(record: Dict, sm: SourceMap) -> str:
    """Connections the sources support, plus the chains they form.

    The chain view is the one the product brief asks for -- Person A -> worked for
    Organization B -> which operated from Location C. It is assembled ONLY from links
    that were separately extracted and attributed; nothing is bridged because it would
    read well. A missing hop stays missing and turns up under Information Gaps.
    """
    links = record.get("relationships") or []
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
    """
    from .extraction import norm_key

    by_from: Dict[str, List[Dict]] = {}
    for l in links:
        by_from.setdefault(norm_key(l["from"]), []).append(l)

    chains: List[List[Dict]] = []
    for start in links:
        stack = [[start]]
        while stack and len(chains) < limit:
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
                stack.append(extended)
    # Longest first, and only the distinct ones -- a 3-hop chain contains a 2-hop one.
    chains.sort(key=len, reverse=True)
    kept: List[List[Dict]] = []
    for c in chains:
        sig = tuple((norm_key(l["from"]), norm_key(l["to"])) for l in c)
        if any(sig == tuple((norm_key(l["from"]), norm_key(l["to"])) for l in k)[:len(sig)] for k in kept):
            continue
        kept.append(c)
        if len(kept) >= limit:
            break
    return kept


def _cross_source_section(connective: Dict, record: Dict, sm: SourceMap) -> str:
    cs = (connective or {}).get("cross_source") or {}
    parts: List[str] = []

    corroborated = cs.get("corroborated") or []
    parts.append("**Carried by more than one source**\n")
    parts.append(_bullets([f"{_ascii(c.get('statement'))} ({sm.cite(c.get('sources') or [])})"
                           for c in corroborated]))

    single = cs.get("single_source") or []
    parts.append("\n\n**Carried by a single source**\n")
    parts.append(_bullets([f"{_ascii(c.get('statement'))} ({sm.num(c.get('source'))})"
                           for c in single]))

    rels = cs.get("source_relationships") or []
    parts.append("\n\n**How the sources relate to each other**\n")
    parts.append(_bullets([f"{_ascii(r.get('observation'))} ({sm.cite(r.get('sources') or [])})"
                           for r in rels]))
    return "".join(parts)


def _conflicts_section(connective: Dict, record: Dict, sm: SourceMap) -> str:
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
        "No conflicts between the supplied sources were recorded."


def _gaps_section(connective: Dict, record: Dict, sm: SourceMap) -> str:
    rows: List[str] = []
    for g in record.get("gaps") or []:
        line = f"{_ascii(g['gap'])} ({sm.num(g.get('source'))})"
        if g.get("lead"):
            line += f"\n  - Possible next step: {_ascii(g['lead'])}"
        rows.append(line)
    for g in (connective or {}).get("gaps") or []:
        line = _ascii(g.get("gap"))
        if g.get("lead"):
            line += f"\n  - Possible next step: {_ascii(g['lead'])}"
        if line:
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
    """The sixteen sections, in order, as ASCII Markdown."""
    sm = SourceMap(source_labels)
    profiles_by_label = {getattr(p, "source_label", ""): p for p in (profiles or [])}
    c = connective or {}

    body: List[str] = [f"# {_ascii(title)}\n"]

    body.append(_h(SECTIONS[0], 1))
    body.append(_ascii(c.get("overview")) or
                "The supplied material has been organized below. No overview text was produced.")

    body.append(_h(SECTIONS[1], 2))
    body.append(_bullets([f"{_ascii(o.get('observation'))} ({sm.cite(o.get('sources') or [])})"
                          for o in c.get("observations") or []]))

    body.append(_h(SECTIONS[2], 3))
    body.append(_source_index(record, sm, profiles_by_label))

    body.append(_h(SECTIONS[3], 4));  body.append(_people_section(record, sm))
    body.append(_h(SECTIONS[4], 5));  body.append(_orgs_section(record, sm))
    body.append(_h(SECTIONS[5], 6));  body.append(_places_section(record, sm))
    body.append(_h(SECTIONS[6], 7));  body.append(_timeline_section(record, sm))
    body.append(_h(SECTIONS[7], 8));  body.append(_events_section(record, sm))
    body.append(_h(SECTIONS[8], 9));  body.append(_claims_section(record, sm))
    body.append(_h(SECTIONS[9], 10)); body.append(_documents_section(record, sm))
    body.append(_h(SECTIONS[10], 11)); body.append(_relationships_section(record, sm))
    body.append(_h(SECTIONS[11], 12)); body.append(_cross_source_section(c, record, sm))
    body.append(_h(SECTIONS[12], 13)); body.append(_conflicts_section(c, record, sm))
    body.append(_h(SECTIONS[13], 14)); body.append(_gaps_section(c, record, sm))

    body.append(_h(SECTIONS[14], 15))
    body.append(_ascii(c.get("synthesis")) or
                "No synthesis text was produced for the supplied material.")

    body.append(_h(SECTIONS[15], 16))
    body.append("\n".join(f"- **Source {i}** -- {_ascii(l)}" for i, l in enumerate(sm.labels, start=1))
                or _EMPTY)

    return "\n".join(body).strip() + "\n"


def audit_report(markdown: str, source_labels: Sequence[str]) -> List[neutrality.Violation]:
    """Final gate: scan the assembled document for anything judging in its own voice.

    Runs on the rendered text rather than the record, so it also catches phrasing the
    renderer itself introduces. A clean pass here is the claim the product makes.
    """
    return neutrality.scan(markdown, source_labels, path="report")
