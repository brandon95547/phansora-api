"""Cross-source correlation & intelligence synthesis for Dossier Nova.

The rest of the pipeline processes every source INDEPENDENTLY (cleanup, profiling,
chunk-bucketing all see one source at a time). This module is the one stage that
ingests every source TOGETHER and produces a structured "intelligence model":

  - findings   : factual claims MERGED across sources (one entry per fact, with the
                 list of sources that support it and a confidence level), and each
                 tagged fact vs allegation.
  - timeline   : a chronological investigative timeline with per-event sources and
                 cross-source date discrepancies.
  - cross_source: what all sources agree on, what is uniquely reported, what
                 conflicts, and what is unresolved.
  - executive_summary: a one-page case overview, current status, key findings,
                 overall evidence confidence, and major unknowns.

It then renders that model into the Markdown "front matter" that leads the dossier
(Executive Summary first, then Timeline, Key Findings, Cross-Source Findings, and an
Evidence Matrix table). This is Phase 1 of the intelligence-grade upgrade — it runs
a single DeepSeek call with prompt-instructed JSON, mirroring source_profiler.py.

Everything rendered here is ASCII-only: the downstream Node PDF renderer strips
non-ASCII glyphs (so no checkmarks, em-dashes, or smart quotes).
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

# The four-level confidence rubric shown to the model and used to normalize output.
CONFIDENCE_LEVELS = ("Very High", "High", "Medium", "Low")
_CONFIDENCE_LOOKUP = {c.lower(): c for c in CONFIDENCE_LEVELS}


_SYNTHESIS_SYSTEM_PROMPT = """You are a senior intelligence analyst. You are given \
several independent sources about a single case or subject. Correlate them into ONE \
coherent intelligence assessment — do NOT summarize each source separately.

Your job:
1. MERGE duplicate facts. If several sources report the same fact, output it ONCE as a \
single finding and list every source that supports it. Never repeat a fact just \
because multiple sources mention it.
2. Assign a CONFIDENCE level to every finding using this rubric:
   - "Very High": confirmed by 3+ independent sources, OR an official police statement, \
OR court records.
   - "High": confirmed by two independent professional news organizations.
   - "Medium": reported by a single reliable publication.
   - "Low": speculation, anonymous sources, or single-source reporting that is not \
independently corroborated.
3. Distinguish FACTS from ALLEGATIONS. Anything not yet proven (charges, accusations, \
prosecutorial claims) must be type "allegation" with attribution wording (e.g. "Police \
allege", "Prosecutors claim", "According to court documents"). Confirmed facts are type \
"fact".
4. Build a chronological TIMELINE of events with dates (and times if available). If \
sources disagree on a date, note the discrepancy.
5. Produce a CROSS-SOURCE analysis: what ALL sources agree on, what is reported by only \
ONE source, what CONFLICTS between sources, and what remains UNRESOLVED.
6. Write a one-page EXECUTIVE SUMMARY: overview, current status, key findings, overall \
evidence confidence, and the major unanswered questions.

Return ONLY valid JSON in exactly this shape (no markdown, no commentary):
{
  "subject": "short case/subject title",
  "executive_summary": {
    "overview": "2-4 sentence overview of the case",
    "current_status": "current legal/investigative status in one sentence",
    "evidence_confidence": "Very High | High | Medium | Low",
    "key_findings": ["short bullet", "..."],
    "major_unknowns": ["short bullet", "..."]
  },
  "timeline": [
    {"date": "e.g. July 4 or 2024-07-04", "time": "optional, else empty",
     "event": "what happened", "sources": ["Source label", "..."],
     "discrepancy": "note if sources disagree on the date/time, else empty"}
  ],
  "findings": [
    {"statement": "one factual finding, stated once",
     "type": "fact | allegation",
     "attribution": "for allegations: e.g. 'Police allege'; else empty",
     "confidence": "Very High | High | Medium | Low",
     "supporting_sources": ["Source label", "..."]}
  ],
  "cross_source": {
    "agreed": ["fact every source agrees on", "..."],
    "unique": [{"statement": "fact reported by only one source", "source": "Source label"}],
    "conflicting": [{"topic": "what they disagree about",
                     "versions": [{"source": "Source label", "claim": "their version"}]}],
    "unresolved": ["open point not settled by any source", "..."]
  }
}

Use the EXACT source labels given to you. Keep every string plain ASCII text."""


def _extract_json(raw: str) -> Optional[Dict]:
    """Parse JSON from LLM output, tolerating code fences and stray prose."""
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith("```"):
        lines = [l for l in raw.splitlines() if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None


def _normalize_confidence(value: str) -> str:
    """Map a free-form confidence string onto one of the four rubric levels."""
    return _CONFIDENCE_LOOKUP.get(str(value or "").strip().lower(), "Medium")


def _build_synthesis_input(sources, source_profiles, sample_chars: int) -> str:
    """Compact, labeled digest of every source: profile + capped excerpt.

    Profiles (already extracted per source) give the model each source's thesis and
    key claims cheaply; the excerpt grounds it in the actual text. Keeping this bounded
    is what makes a single all-sources correlation call feasible.
    """
    profiles_by_label = {p.source_label: p for p in (source_profiles or [])}
    blocks: List[str] = []
    for src in sources:
        label = src.get("label", "unknown")
        text = (src.get("text") or "").strip()
        prof = profiles_by_label.get(label)
        header = [f"=== SOURCE: {label} ==="]
        if prof is not None:
            header.append(f"Type: {prof.source_type}; role: {prof.rhetorical_role}")
            if prof.central_argument:
                header.append(f"Thesis: {prof.central_argument}")
            if prof.key_claims:
                header.append("Key claims: " + " | ".join(prof.key_claims))
        excerpt = text[:sample_chars]
        blocks.append("\n".join(header) + "\n\n" + excerpt)
    return "\n\n".join(blocks)


def synthesize_dossier(
    sources,
    source_profiles,
    client,
    sample_chars: int = 8000,
) -> Optional[Dict]:
    """Run the one cross-source correlation call. Returns the intelligence model
    dict, or None on failure (the pipeline then proceeds without front matter)."""
    if not sources or len(sources) < 2:
        return None

    digest = _build_synthesis_input(sources, source_profiles, sample_chars)
    source_labels = [s.get("label", "unknown") for s in sources]
    user_prompt = (
        f"There are {len(sources)} sources with labels: {', '.join(source_labels)}.\n\n"
        f"{digest}\n\n"
        "Correlate these sources into one intelligence assessment as specified."
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        raw = (response.choices[0].message.content or "").strip()
        model = _extract_json(raw)
    except Exception as e:  # noqa: BLE001 — never let synthesis break the pipeline
        print(f"[SYNTHESIS] Cross-source synthesis failed: {e}")
        return None

    if not model:
        print("[SYNTHESIS] Could not parse synthesis JSON; skipping front matter.")
        return None

    # Normalize confidence values so rendering and downstream logic can rely on them.
    for f in model.get("findings") or []:
        f["confidence"] = _normalize_confidence(f.get("confidence"))
    es = model.get("executive_summary") or {}
    if es:
        es["evidence_confidence"] = _normalize_confidence(es.get("evidence_confidence"))
    return model


# ---------------------------------------------------------------------------
# Rendering: intelligence model -> ASCII Markdown front matter
# ---------------------------------------------------------------------------

def _display_label(label: str) -> str:
    """Short column/attribution form of a source label (drop .txt, trim)."""
    s = str(label or "source").strip()
    if s.lower().endswith(".txt"):
        s = s[:-4]
    return s


def _sources_str(labels) -> str:
    names = [_display_label(l) for l in (labels or []) if str(l).strip()]
    return ", ".join(names) if names else "unattributed"


def _evidence_matrix_table(findings, source_labels) -> List[str]:
    """A pipe-table (finding x source) the Node renderer draws as a real table.
    Present cell = 'Yes', absent = '-'. ASCII only (no checkmark glyphs)."""
    cols = [_display_label(l) for l in source_labels]
    header = "| Finding | " + " | ".join(cols) + " |"
    sep = "| --- | " + " | ".join("---" for _ in cols) + " |"
    rows = [header, sep]
    for f in findings:
        stmt = str(f.get("statement", "")).strip()
        if not stmt:
            continue
        if len(stmt) > 60:
            stmt = stmt[:57].rstrip() + "..."
        supporting = {_display_label(s) for s in (f.get("supporting_sources") or [])}
        cells = ["Yes" if c in supporting else "-" for c in cols]
        rows.append(f"| {stmt} | " + " | ".join(cells) + " |")
    return rows


def render_front_matter(model: Dict, source_labels) -> str:
    """Render the intelligence model as the Markdown that leads the dossier.

    Reflow-safe by construction: only '#'/'##' headings, '- ' bullets, blank-line
    separated blocks, and '|' pipe-tables — the Node reflow + PDF renderer preserve
    exactly those. All text is plain ASCII.
    """
    if not model:
        return ""
    out: List[str] = []
    es = model.get("executive_summary") or {}

    # --- Executive Summary (first page) ---
    out.append("# Executive Summary")
    out.append("")
    if es.get("overview"):
        out.append(str(es["overview"]).strip())
        out.append("")
    if es.get("current_status"):
        out.append(f"- Current status: {str(es['current_status']).strip()}")
    if es.get("evidence_confidence"):
        out.append(f"- Overall evidence confidence: {es['evidence_confidence']}")
    out.append("")
    key_findings = [str(x).strip() for x in (es.get("key_findings") or []) if str(x).strip()]
    if key_findings:
        out.append("Key findings:")
        out.append("")
        out.extend(f"- {x}" for x in key_findings)
        out.append("")
    unknowns = [str(x).strip() for x in (es.get("major_unknowns") or []) if str(x).strip()]
    if unknowns:
        out.append("Major unanswered questions:")
        out.append("")
        out.extend(f"- {x}" for x in unknowns)
        out.append("")

    # --- Investigative Timeline ---
    timeline = model.get("timeline") or []
    if timeline:
        out.append("# Investigative Timeline")
        out.append("")
        for ev in timeline:
            date = str(ev.get("date", "")).strip()
            time_ = str(ev.get("time", "")).strip()
            when = f"{date} {time_}".strip() or "Undated"
            event = str(ev.get("event", "")).strip()
            line = f"- {when}: {event}"
            srcs = _sources_str(ev.get("sources"))
            if srcs:
                line += f" (Sources: {srcs})"
            disc = str(ev.get("discrepancy", "")).strip()
            if disc:
                line += f" [Discrepancy: {disc}]"
            out.append(line)
        out.append("")

    # --- Key Findings (confidence + supporting sources beneath each) ---
    findings = model.get("findings") or []
    if findings:
        out.append("# Key Findings")
        out.append("")
        for f in findings:
            stmt = str(f.get("statement", "")).strip()
            if not stmt:
                continue
            ftype = str(f.get("type", "fact")).strip().lower()
            attribution = str(f.get("attribution", "")).strip()
            prefix = ""
            if ftype == "allegation":
                prefix = "[ALLEGATION] "
                if attribution:
                    prefix = f"[ALLEGATION] {attribution}: "
            out.append(f"- {prefix}{stmt}")
            out.append(f"  - Confidence: {f.get('confidence', 'Medium')}")
            out.append(f"  - Supporting sources: {_sources_str(f.get('supporting_sources'))}")
        out.append("")

    # --- Cross-Source Findings ---
    cs = model.get("cross_source") or {}
    if cs:
        out.append("# Cross-Source Findings")
        out.append("")
        agreed = [str(x).strip() for x in (cs.get("agreed") or []) if str(x).strip()]
        out.append("Confirmed by all sources:")
        out.append("")
        out.extend(f"- {x}" for x in agreed) if agreed else out.append("- None identified.")
        out.append("")
        unique = cs.get("unique") or []
        out.append("Unique reporting:")
        out.append("")
        if unique:
            for u in unique:
                stmt = str(u.get("statement", "")).strip()
                src = _display_label(u.get("source"))
                if stmt:
                    out.append(f"- {stmt} ({src})")
        else:
            out.append("- None identified.")
        out.append("")
        conflicting = cs.get("conflicting") or []
        out.append("Conflicting reporting:")
        out.append("")
        if conflicting:
            for c in conflicting:
                topic = str(c.get("topic", "")).strip()
                versions = c.get("versions") or []
                vtext = "; ".join(
                    f"{_display_label(v.get('source'))}: {str(v.get('claim', '')).strip()}"
                    for v in versions
                )
                out.append(f"- {topic} -- {vtext}" if topic else f"- {vtext}")
        else:
            out.append("- None identified.")
        out.append("")
        unresolved = [str(x).strip() for x in (cs.get("unresolved") or []) if str(x).strip()]
        if unresolved:
            out.append("Unresolved:")
            out.append("")
            out.extend(f"- {x}" for x in unresolved)
            out.append("")

    # --- Evidence Matrix (finding x source table) ---
    if findings and source_labels:
        out.append("# Evidence Matrix")
        out.append("")
        out.extend(_evidence_matrix_table(findings, source_labels))
        out.append("")

    return "\n".join(out).strip() + "\n"
