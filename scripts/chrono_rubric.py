#!/usr/bin/env python3
"""Score a Chrono Origin trace the way a historian would mark it.

The unit tests pin the rules the pipeline applies. They cannot tell you whether
the result reads like research, because that is a property of the whole document:
whether it separated a text from the copies of it that survive, whether it said
what evidence does not exist, whether anything on the page is still resting on a
wiki. This scores that.

The reference is the outline a user wrote by hand for "Jesus Christ" — context
that is explicitly not influence, the history of the term, reconstructed rather
than documented dates, composition separated from surviving manuscripts, outside
attestation with its own disputes, the language chain, later institutional
development kept later, and the calendar the dates were converted into. Nine of
the checks below are mechanical; the rest are printed for a human to judge,
because "did it name the missing contemporary documentation" is not a regex.

    python scripts/chrono_rubric.py trace.json

Get a trace to score with, e.g.:

    curl -s -X POST localhost:8000/chrono/trace \\
      -H 'content-type: application/json' \\
      -d '{"title":"Jesus Christ"}' > trace.json
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Tuple

PASS, FAIL, LOOK = "PASS", "FAIL", "LOOK"

LEAD_HOSTS = (
    "wikipedia.org", "reddit.com", "medium.com", "britannica.com", "history.com",
    "worldhistory.org", "ancient.eu", "youtube.com", "quora.com", "fandom.com",
)
# The same sources as they appear in prose. A dossier naming "the Wikipedia
# article on X" as its earliest supporting source is the failure this is looking
# for, and it never contains a domain.
LEAD_NAMES = (
    "wikipedia", "reddit", "medium.com", "britannica", "history.com", "youtube",
    "quora", "fandom", "blog post", "web article", "search result",
)


def _nodes(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    if trace.get("origin"):
        out.append(trace["origin"])
    out.extend(trace.get("timeline") or [])
    return out


def _citations(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    cites = list(trace.get("citations") or [])
    for n in _nodes(trace):
        cites.extend(n.get("citations") or [])
    for c in trace.get("connections") or []:
        cites.extend(c.get("citations") or [])
    return cites


def _types(trace: Dict[str, Any]) -> List[str]:
    return [str(n.get("node_type") or "event") for n in _nodes(trace)]


def _dossiers(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [n["evidence"] for n in _nodes(trace) if isinstance(n.get("evidence"), dict)]


# --------------------------------------------------------------- the checks
def check_no_lead_is_evidence(trace) -> Tuple[str, str]:
    """The mandate, stated as one assertion."""
    bad = [
        c for c in _citations(trace)
        if c.get("role") == "evidence" and (c.get("tier_rank") or 5) >= 4
    ]
    if bad:
        return FAIL, f"{len(bad)} citation(s) carry role=evidence at tier 4-5, e.g. {bad[0].get('url')}"
    return PASS, "no tier 4-5 citation is doing evidentiary work"


def check_origin_not_resting_on_a_wiki(trace) -> Tuple[str, str]:
    origin = trace.get("origin") or {}
    ev = origin.get("evidence") or {}
    blob = f"{ev.get('earliest_supporting_source', '')} {ev.get('provenance', '')}".lower()
    hit = next((h for h in LEAD_HOSTS + LEAD_NAMES if h in blob), None)
    if hit:
        return FAIL, f"the origin's supporting source names {hit!r}"
    if not blob.strip():
        return FAIL, "the origin names no supporting source at all"
    return PASS, "the origin names something other than a general-web source"


def check_composition_split_from_manuscripts(trace) -> Tuple[str, str]:
    types = _types(trace)
    if "text_composition" in types and "manuscript_witness" in types:
        return PASS, "a text's composition and its surviving copies are separate entries"
    if "text_composition" in types:
        return FAIL, "texts are dated but no surviving copy is given its own entry"
    return LOOK, "no text_composition entry — expected for a text-based subject"


def check_outside_attestation(trace) -> Tuple[str, str]:
    n = _types(trace).count("external_attestation")
    return (PASS, f"{n} outside-attestation entr{'y' if n == 1 else 'ies'}") if n else (
        FAIL, "nothing attests the subject from outside its own tradition"
    )


def check_context_is_not_influence(trace) -> Tuple[str, str]:
    ids = {n.get("id") for n in _nodes(trace) if n.get("node_type") == "context"}
    if not ids:
        return LOOK, "no context entries — fine for some subjects, thin for most"
    causal = {"derives_from", "retells", "translates", "responds_to"}
    bad = [c for c in trace.get("connections") or []
           if c.get("from_id") in ids and c.get("relation") in causal]
    if bad:
        return FAIL, f"{len(bad)} context entr(ies) are joined by a causal relation"
    return PASS, f"{len(ids)} context entr(ies), none asserting influence"


def check_nothing_projected_backward(trace) -> Tuple[str, str]:
    late = [n for n in _nodes(trace) if n.get("node_type") == "institutional_development"]
    if not late:
        return LOOK, "no institutional-development entries"
    early = [n for n in late if isinstance(n.get("year"), int) and n["year"] < 100]
    if early:
        return FAIL, f"{len(early)} institutional development(s) dated before 100 CE"
    return PASS, f"{len(late)} later development(s), none projected onto the origin"


def check_says_what_is_missing(trace) -> Tuple[str, str]:
    named = [d for d in _dossiers(trace)
             if str(d.get("missing_piece") or "").strip()
             and str(d.get("missing_piece")).strip().lower() not in ("none identified", "none", "unknown")]
    if not named:
        return FAIL, "no claim names the evidence whose absence limits it"
    return PASS, f"{len(named)} claim(s) name what is missing"


def check_admits_uncertainty(trace) -> Tuple[str, str]:
    states = [d.get("verification") for d in _dossiers(trace)]
    weak = sum(1 for s in states if s in ("unverified", "unknown"))
    if not states:
        return FAIL, "no dossiers to grade"
    if weak == 0:
        return FAIL, "every claim is verified — a trace where nothing is uncertain is lying"
    return PASS, f"{weak}/{len(states)} claim(s) marked unverified or unknown"


def check_independent_chains(trace) -> Tuple[str, str]:
    n = trace.get("independent_chain_count") or 0
    return (PASS, f"{n} independent evidence chains") if n >= 2 else (
        FAIL, f"only {n} independent chain(s) — the trace rests on one voice"
    )


def check_breadth(trace) -> Tuple[str, str]:
    """The complaint that started this: the trace was too thin to be a history."""
    kinds = {t for t in _types(trace) if t != "event"}
    count = len(_nodes(trace))
    if count >= 10 and len(kinds) >= 4:
        return PASS, f"{count} entries across {len(kinds)} kinds of claim"
    return FAIL, f"only {count} entries across {len(kinds)} kinds of claim beyond plain events"


CHECKS = [
    ("No lead is cited as evidence", check_no_lead_is_evidence),
    ("The origin is not resting on a wiki", check_origin_not_resting_on_a_wiki),
    ("Composition split from surviving copies", check_composition_split_from_manuscripts),
    ("Attested from outside its own tradition", check_outside_attestation),
    ("Context is not presented as influence", check_context_is_not_influence),
    ("Later development stays later", check_nothing_projected_backward),
    ("Says what evidence is missing", check_says_what_is_missing),
    ("Admits what it could not verify", check_admits_uncertainty),
    ("Rests on more than one voice", check_independent_chains),
    ("Broad enough to be a history", check_breadth),
]

# Things no assertion settles. Printed so the reader knows to look rather than
# assume the machine checked them.
BY_HAND = [
    "Are reconstructed dates typed as reconstructed, not stated as record?",
    "Does provenance name real repositories and shelfmarks, or is it hand-waving?",
    "Is the language/term history present where the subject has one?",
    "Is a claim about a text a claim about the TEXT, not about what it narrates?",
    "Would a specialist recognise the disputes it names as the live ones?",
]


def main(path: str) -> int:
    trace = json.loads(open(path).read())
    if "result" in trace and isinstance(trace["result"], dict):
        trace = trace["result"]  # a stored job row rather than a bare trace

    print(f"\n  {trace.get('title') or path}")
    print(f"  {len(_nodes(trace))} entries · {len(_citations(trace))} citations\n")

    failed = 0
    for name, check in CHECKS:
        try:
            verdict, detail = check(trace)
        except Exception as exc:  # noqa: BLE001 - a broken check must not hide the rest
            verdict, detail = FAIL, f"check errored: {exc}"
        if verdict == FAIL:
            failed += 1
        print(f"  [{verdict}] {name}\n         {detail}")

    types = sorted({t for t in _types(trace)})
    print(f"\n  Entry kinds present: {', '.join(types) or '(none)'}")

    ranks: Dict[int, int] = {}
    for c in _citations(trace):
        r = c.get("tier_rank") or 5
        ranks[r] = ranks.get(r, 0) + 1
    print("  Citations by tier:  " + " · ".join(f"tier {r}: {ranks[r]}" for r in sorted(ranks)))

    print("\n  Read these yourself:")
    for q in BY_HAND:
        print(f"    - {q}")

    print(f"\n  {len(CHECKS) - failed}/{len(CHECKS)} mechanical checks passed\n")
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
