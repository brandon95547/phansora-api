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
development kept later, and the calendar the dates were converted into. The
checks below are mechanical; the questions after them are printed for a human to
judge, because "did it name the missing contemporary documentation" is not a regex.

The list grew after a trace scored 10/10 and was still wrong. Every check it had
asked whether a KIND of entry was PRESENT, and a report with exactly one entry
per kind passes all of them while being far too thin to be a history. So the
calendar, the language chain and the ordering of anything older than the origin
are now asked about directly, and a trace listing surviving manuscripts of texts
it never dated fails instead of being waved through.

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
    if "manuscript_witness" in types:
        # The failure this check was too soft to catch. A trace listing surviving
        # copies without a single composed text has answered "what physically
        # exists" and never asked "what was written, and when" — which is where
        # every biographical claim about a text-based subject actually enters.
        return FAIL, "surviving copies are listed but no text is given a composition date"
    return LOOK, "no text_composition entry — expected for a text-based subject"


def check_dating_framework(trace) -> Tuple[str, str]:
    """Whose calendar are these dates in, and who converted them?

    A report that prints "c. 4 BCE" without saying that nobody alive then was
    counting from there, and that somebody later did the counting, is presenting
    an editorial act as an observation.
    """
    n = _types(trace).count("dating_framework")
    if n:
        return PASS, f"{n} entr{'y' if n == 1 else 'ies'} on how these dates were derived and converted"
    years = [n.get("year") for n in _nodes(trace) if isinstance(n.get("year"), int)]
    if years and min(years) < 1500:
        return FAIL, "dates are stated in a calendar the period did not use, with no entry saying so"
    return LOOK, "no dating_framework entry"


def check_language_chain(trace) -> Tuple[str, str]:
    """A name crossing languages is a chain, not a footnote."""
    forms = [n for n in _nodes(trace) if n.get("node_type") == "linguistic_transmission"]
    if not forms:
        return LOOK, "no linguistic_transmission entries — fine only if the name never moved"
    if len(forms) < 2:
        return FAIL, "one entry summarising a language chain instead of one entry per attested form"
    return PASS, f"{len(forms)} attested forms, traced separately"


def check_ancestry_is_typed_as_ancestry(trace) -> Tuple[str, str]:
    """Anything older than the origin has to say what it is doing there.

    The board draws the origin as the subject's beginning. An entry dated before
    it is either the ancestry the subject emerged among — which the vocabulary
    can say — or a contradiction of the origin the report never noticed.
    """
    origin = trace.get("origin") or {}
    oy = origin.get("year")
    if not isinstance(oy, int):
        return LOOK, "the origin has no year to order the rest against"
    background = {"context", "term_history", "linguistic_transmission", "precursor_context"}
    earlier = [n for n in (trace.get("timeline") or []) if isinstance(n.get("year"), int) and n["year"] < oy]
    untyped = [n for n in earlier if str(n.get("node_type") or "event") not in background]
    if untyped:
        return FAIL, (
            f"{len(untyped)} entr(ies) predate the origin without being typed as background, "
            f"e.g. {untyped[0].get('source_title')!r}"
        )
    if not earlier:
        return LOOK, "nothing predates the origin — thin for a subject with any ancestry"
    return PASS, f"{len(earlier)} ancestor entr(ies), all typed as background"


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
    """The complaint that started this: the trace was too thin to be a history.

    The bar was ten entries across four kinds, which a twelve-entry report
    reviewed as "conspicuously missing" its central figures cleared comfortably.
    Ten was never the number: the hand-written outline this rubric scores against
    has four gospels, the letters, the manuscripts, the outside attestations, a
    four-form name chain and two calendar entries before anything optional. A
    subject with real ancestry and a textual tradition produces entries in the
    twenties, and one that produces twelve has summarised rather than traced.
    """
    kinds = {t for t in _types(trace) if t != "event"}
    count = len(_nodes(trace))
    if count >= 16 and len(kinds) >= 6:
        return PASS, f"{count} entries across {len(kinds)} kinds of claim"
    return FAIL, f"only {count} entries across {len(kinds)} kinds of claim beyond plain events"


CHECKS = [
    ("No lead is cited as evidence", check_no_lead_is_evidence),
    ("The origin is not resting on a wiki", check_origin_not_resting_on_a_wiki),
    ("Composition split from surviving copies", check_composition_split_from_manuscripts),
    ("Says whose calendar these dates are in", check_dating_framework),
    ("Traces the name form by form", check_language_chain),
    ("Ancestry is typed as ancestry", check_ancestry_is_typed_as_ancestry),
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
