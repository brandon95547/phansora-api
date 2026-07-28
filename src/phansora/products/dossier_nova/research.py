"""Structured research dataset for downstream AI use (Narrava Studio).

The dossier itself is a narrative report; this module renders the SAME collected
research into a machine-oriented dataset: subject, executive summary, entities
(people / organizations / locations), timeline, findings with confidence and
supporting sources, cross-source analysis, and a per-source inventory. Narrava
Studio attaches it to a narration brief as the factual knowledge base, and the
dashboard's "View Research" dialog shows it for review.

Built entirely from data the pipeline already produced (the cross-source
intelligence model + source profiles) — no LLM calls happen here, so it can never
add latency or cost to a dossier run.
"""
from __future__ import annotations

from typing import Dict, List, Optional

RESEARCH_FORMAT = "dossier-nova-research"
RESEARCH_VERSION = 1


def _clean_str(value) -> str:
    return str(value).strip() if value is not None else ""


def _str_list(items) -> List[str]:
    if not isinstance(items, list):
        return []
    return [s for s in (_clean_str(x) for x in items) if s]


def _dict_list(items, keys: Dict[str, type]) -> List[Dict]:
    """Keep only dict entries, only the expected keys, with normalized values."""
    out: List[Dict] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        entry: Dict = {}
        for key, kind in keys.items():
            if kind is list:
                entry[key] = _str_list(item.get(key))
            else:
                entry[key] = _clean_str(item.get(key))
        if any(v for v in entry.values()):
            out.append(entry)
    return out


def build_research_dataset(
    intel_model: Optional[Dict],
    sources: List[Dict[str, str]],
    source_profiles,
) -> Optional[Dict]:
    """Assemble the research dataset from what the pipeline already collected.

    ``intel_model`` is the synthesis stage's JSON (may be None when synthesis
    failed or was disabled); ``source_profiles`` are SourceProfile dataclasses.
    Returns None only when there is nothing at all worth shipping.
    """
    profiles_by_label = {p.source_label: p for p in (source_profiles or [])}

    source_inventory: List[Dict] = []
    for src in sources or []:
        label = _clean_str(src.get("label")) or "input"
        prof = profiles_by_label.get(label)
        source_inventory.append({
            "label": label,
            "type": _clean_str(getattr(prof, "source_type", "")) or "unknown",
            "role": _clean_str(getattr(prof, "rhetorical_role", "")),
            "central_argument": _clean_str(getattr(prof, "central_argument", "")),
            "key_claims": _str_list(getattr(prof, "key_claims", None)),
            "char_count": len(src.get("text") or ""),
        })

    model = intel_model if isinstance(intel_model, dict) else {}

    es_raw = model.get("executive_summary") or {}
    executive_summary = {
        "overview": _clean_str(es_raw.get("overview")),
        "current_status": _clean_str(es_raw.get("current_status")),
        "evidence_confidence": _clean_str(es_raw.get("evidence_confidence")),
        "key_findings": _str_list(es_raw.get("key_findings")),
        "major_unknowns": _str_list(es_raw.get("major_unknowns")),
    }

    entities_raw = model.get("entities") or {}
    entities = {
        "people": _dict_list(entities_raw.get("people"), {"name": str, "role": str, "description": str}),
        "organizations": _dict_list(entities_raw.get("organizations"), {"name": str, "role": str}),
        "locations": _dict_list(entities_raw.get("locations"), {"name": str, "relevance": str}),
    }

    timeline = _dict_list(model.get("timeline"), {
        "date": str, "time": str, "event": str, "sources": list, "discrepancy": str,
    })
    findings = _dict_list(model.get("findings"), {
        "statement": str, "type": str, "attribution": str, "confidence": str,
        "supporting_sources": list,
    })

    cs_raw = model.get("cross_source") or {}
    cross_source = {
        "agreed": _str_list(cs_raw.get("agreed")),
        "unique": _dict_list(cs_raw.get("unique"), {"statement": str, "source": str}),
        "conflicting": [
            {
                "topic": _clean_str(c.get("topic")),
                "versions": _dict_list(c.get("versions"), {"source": str, "claim": str}),
            }
            for c in (cs_raw.get("conflicting") or [])
            if isinstance(c, dict)
        ],
        "unresolved": _str_list(cs_raw.get("unresolved")),
    }

    dataset = {
        "format": RESEARCH_FORMAT,
        "version": RESEARCH_VERSION,
        "subject": _clean_str(model.get("subject")),
        "executive_summary": executive_summary,
        "entities": entities,
        "timeline": timeline,
        "findings": findings,
        "cross_source": cross_source,
        "sources": source_inventory,
        "stats": {
            "source_count": len(source_inventory),
            "finding_count": len(findings),
            "timeline_event_count": len(timeline),
            "person_count": len(entities["people"]),
        },
    }

    has_intel = bool(
        dataset["subject"]
        or executive_summary["overview"]
        or findings
        or timeline
        or entities["people"]
    )
    if not has_intel and not source_inventory:
        return None
    return dataset
