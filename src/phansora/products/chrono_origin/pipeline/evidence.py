"""Evidence bookkeeping that belongs in code, not in a prompt.

Three jobs used to be delegated to the model and paid for in tokens on every
trace: collapsing duplicate mentions, deciding which sources are really one
source repeated, and turning an evidence type into a claim class. All three are
deterministic. Doing them here makes them cheaper, repeatable, and — the part
that matters for a product about evidence — auditable, because the rule is
written down instead of re-improvised by a language model each run.

What stays with the model is judgement: what a source says, whether a link is
real, what is missing. That is what the tokens are for.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlsplit

from ..models import CLAIM_CLASS_BY_EVIDENCE_TYPE
from .source_policy import EVIDENCE_RANK, READ_RANK, TIER_ORDER, WORST_RANK, rank_for

# Two rankings, two questions, and conflating them makes traces worse.
#
# READ_RANK (from source_policy) orders what is worth OPENING: an unrecognised
# domain outranks a wiki there, because reading it is how we learn it is a
# national archive. EVIDENCE_RANK orders what a source can SUPPORT, and there an
# unclassified page is tier 5 like any other lead. Reading order is imported
# under its old name so the callers below read as they always did.
_TIER_RANK = READ_RANK

# Hosts where each URL is a distinct work rather than one publisher's voice.
# Grouping these by domain would wrongly collapse two unrelated papers into one
# "chain" and undercount genuine independent corroboration.
_PER_ITEM_HOSTS = (
    "jstor.org", "doi.org", "archive.org", "hathitrust.org", "academia.edu",
    "persee.fr", "openedition.org", "cambridge.org", "oup.com", "springer.com",
    "tandfonline.com", "sciencedirect.com", "brill.com", "degruyter.com",
    "perseus.tufts.edu", "arxiv.org", "zenodo.org", "ncbi.nlm.nih.gov",
)

# Two-part public suffixes common enough to matter for domain grouping.
_MULTI_TLDS = (
    ".ac.uk", ".co.uk", ".org.uk", ".gov.uk", ".sch.uk", ".net.uk",
    ".com.au", ".edu.au", ".gov.au", ".org.au", ".co.nz", ".ac.nz",
    ".co.jp", ".ac.jp", ".co.za", ".ac.za", ".com.br", ".co.in", ".ac.in",
)

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


# --------------------------------------------------------------------- domains
def registrable_domain(url: str) -> str:
    """Best-effort eTLD+1, lowercased. Empty string when the URL is unusable."""
    try:
        host = (urlsplit(url or "").hostname or "").lower()
    except ValueError:
        return ""
    if not host:
        return ""
    host = host[4:] if host.startswith("www.") else host
    for suffix in _MULTI_TLDS:
        if host.endswith(suffix):
            head = host[: -len(suffix)].rsplit(".", 1)[-1]
            return f"{head}{suffix}" if head else host
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def _is_per_item_host(url: str) -> bool:
    u = (url or "").lower()
    return any(h in u for h in _PER_ITEM_HOSTS)


# -------------------------------------------------------------- normalisation
def _norm(text: Any) -> str:
    """Casefold, strip punctuation and collapse whitespace, for equality tests."""
    s = _PUNCT.sub(" ", str(text or "").lower())
    return _WS.sub(" ", s).strip()


def claim_class_for(evidence_type: Any, declared: Any = None) -> str:
    """Map an evidence type to its claim class.

    A declared class is honoured only when it is a legal value AND consistent with
    the evidence type: a model that labels a tradition as direct evidence is
    overruled, because the whole point of the classification is that it cannot be
    talked up.
    """
    derived = CLAIM_CLASS_BY_EVIDENCE_TYPE.get(str(evidence_type or ""), "unknown")
    if declared and declared == derived:
        return derived
    return derived


# ------------------------------------------------------------------- mentions
def dedupe_mentions(mentions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse mentions describing the same source at the same date.

    Rounds overlap heavily — the same manuscript surfaces in three of them — and
    every duplicate used to be re-sent to the synthesis call and re-reasoned about.
    The survivor keeps the highest confidence and the union of the citations, so
    merging loses no evidence, only repetition.
    """
    merged: Dict[Tuple[str, Any], Dict[str, Any]] = {}
    order: List[Tuple[str, Any]] = []

    for m in mentions or []:
        if not isinstance(m, dict):
            continue
        key = (_norm(m.get("source_title")), m.get("year"))
        if not key[0] and key[1] is None:
            continue
        existing = merged.get(key)
        if existing is None:
            copy = dict(m)
            copy["citations"] = list(dict.fromkeys(copy.get("citations") or []))
            merged[key] = copy
            order.append(key)
            continue

        # Union the citations; a mention seen from two angles is better evidenced.
        cites = list(existing.get("citations") or []) + list(m.get("citations") or [])
        existing["citations"] = list(dict.fromkeys(c for c in cites if c))
        try:
            if float(m.get("confidence", 0) or 0) > float(existing.get("confidence", 0) or 0):
                existing["confidence"] = m.get("confidence")
                if m.get("claim"):
                    existing["claim"] = m["claim"]
        except (TypeError, ValueError):
            pass
        # Absences fill in from whichever round actually found something. The
        # type and the span belong on this list for the same reason the rest do:
        # round two often recognises that what round one logged as a plain event
        # is a text being composed, or that it ran to a second date, and a merge
        # that dropped that would hand synthesis the poorer of the two readings.
        for field in (
            "surviving_copy", "chain", "era_label", "source_tier",
            "node_type", "year_end", "published", "cites",
        ):
            if not existing.get(field) and m.get(field):
                existing[field] = m[field]
        # A lead stays flagged as a lead until a round finds better; only a
        # round that produced a real citation clears it.
        if m.get("discovery_only") and "discovery_only" not in existing:
            existing["discovery_only"] = True

    return [merged[k] for k in order]


def new_mention_count(existing: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> int:
    """How many of `incoming` are not already in `existing`.

    The recursion's stop condition: a round that surfaces nothing new is a round
    that will keep surfacing nothing new, and every further round costs a full set
    of searches.
    """
    seen = {(_norm(m.get("source_title")), m.get("year")) for m in existing if isinstance(m, dict)}
    fresh = 0
    for m in incoming or []:
        if not isinstance(m, dict):
            continue
        if (_norm(m.get("source_title")), m.get("year")) not in seen:
            fresh += 1
    return fresh


# --------------------------------------------------------------------- chains
def build_chains(
    citations: Iterable[Dict[str, Any]],
    mentions: List[Dict[str, Any]],
    tier_for: Any,
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """Group citations into information chains.

    Returns ``(url -> chain_id, chains)``. Two signals, in priority order:

      1. A chain the extract stage named ("repeats Josephus"). If the model read
         that three write-ups all trace to one report, those three are one chain
         regardless of who published them.
      2. Publisher. Two articles on one news site are that site's account told
         twice — not two witnesses. Aggregators and repositories are exempt
         (``_PER_ITEM_HOSTS``): two papers on JSTOR really are two works, and
         collapsing them would undercount real corroboration.

    This is the doctrine's "ten websites repeating one article count as ONE"
    turned into arithmetic, so ``independent_chain_count`` means something.
    """
    declared: Dict[str, str] = {}
    for m in mentions or []:
        if not isinstance(m, dict):
            continue
        label = _norm(m.get("chain"))
        if not label:
            continue
        for url in m.get("citations") or []:
            if url:
                declared[url] = label

    chain_of: Dict[str, str] = {}
    groups: Dict[str, Dict[str, Any]] = {}

    for c in citations:
        url = (c or {}).get("url") or ""
        if not url:
            continue
        label = declared.get(url, "")
        if label:
            key = f"repeats:{label}"
        elif _is_per_item_host(url):
            key = f"work:{url}"
        else:
            key = f"site:{registrable_domain(url) or url}"

        group = groups.get(key)
        if group is None:
            group = {"id": f"c{len(groups) + 1}", "label": label, "urls": [], "tier": "unknown"}
            groups[key] = group
        group["urls"].append(url)
        chain_of[url] = group["id"]

    # A chain is represented by its strongest source: that is the tier at which
    # the chain can actually be argued, not the best-looking link in it. Ranked
    # by EVIDENCE_RANK rather than reading order, so the tier printed on a chain
    # means the same thing as the tier printed on the citations inside it.
    for group in groups.values():
        best = min(
            (tier_for(u) for u in group["urls"]),
            key=lambda t: EVIDENCE_RANK.get(t, WORST_RANK),
            default="unknown",
        )
        group["tier"] = best

    return chain_of, list(groups.values())


# ----------------------------------------------------------------- reading set
def select_for_reading(
    citations: List[Dict[str, Any]],
    mentions: List[Dict[str, Any]],
    tier_for: Any,
    *,
    limit: int,
) -> List[str]:
    """Choose the few URLs worth opening in full.

    Reading pages is the expensive half of the budget, so it is spent where it
    changes an answer: on the sources that carry provenance. A museum record or a
    critical edition can settle whether a manuscript is the original or a later
    copy. A blog restating that manuscript cannot, however confidently it is
    written — so discovery-tier links are never read, only followed.

    One page per domain, so a single well-SEO'd site cannot consume the budget.
    """
    if limit <= 0:
        return []

    cited_by: Dict[str, float] = {}
    for m in mentions or []:
        if not isinstance(m, dict):
            continue
        try:
            conf = float(m.get("confidence", 0.5) or 0.5)
        except (TypeError, ValueError):
            conf = 0.5
        for url in m.get("citations") or []:
            if url:
                cited_by[url] = max(cited_by.get(url, 0.0), conf)

    scored: List[Tuple[int, float, str]] = []

    for c in citations or []:
        url = (c or {}).get("url") or ""
        if not url:
            continue
        tier = tier_for(url)
        if tier == "low_authority":
            continue  # leads, never foundations — reading them buys nothing
        scored.append((_TIER_RANK.get(tier, len(TIER_ORDER)), -cited_by.get(url, 0.0), url))

    scored.sort()

    chosen: List[str] = []
    seen_domains: set[str] = set()
    for _, _, url in scored:
        domain = registrable_domain(url)
        if domain and domain in seen_domains:
            continue
        seen_domains.add(domain)
        chosen.append(url)
        if len(chosen) >= limit:
            break
    return chosen


# ------------------------------------------------------------ chasing leads
# "Follow their references backward" was doctrine for as long as this product has
# existed, and was never anything a caller could invoke. These two functions are
# the mechanism: find the claims resting on leads, and find the lead pages whose
# footnotes are worth harvesting. Neither spends a token.
def chase_targets(
    mentions: List[Dict[str, Any]],
    tier_for: Any,
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    """Mentions whose best citation is a lead (rank 4-5) rather than evidence.

    Returns nothing when a trace is already well-sourced, so a strong subject
    pays nothing for this pass. Weakest and least confident first: those are the
    claims where finding the underlying source changes the answer.
    """
    if limit <= 0:
        return []

    scored: List[Tuple[int, float, int, Dict[str, Any]]] = []
    for i, m in enumerate(mentions or []):
        if not isinstance(m, dict):
            continue
        urls = [u for u in (m.get("citations") or []) if u]
        if not urls:
            continue
        rank = min((EVIDENCE_RANK.get(tier_for(u), WORST_RANK) for u in urls), default=WORST_RANK)
        if rank < 4:
            continue  # already resting on something arguable
        try:
            conf = float(m.get("confidence", 0.5) or 0.5)
        except (TypeError, ValueError):
            conf = 0.5
        scored.append((-rank, conf, i, m))

    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    return [m for _, _, _, m in scored[:limit]]


def select_for_reference_mining(
    citations: List[Dict[str, Any]],
    tier_for: Any,
    *,
    limit: int,
) -> List[str]:
    """Lead pages worth opening FOR THEIR FOOTNOTES ONLY.

    The mirror image of select_for_reading, and deliberately not a widening of
    it: that function picks pages to read for their prose, and its refusal to
    read a wiki is correct and stays. This one opens exactly the pages that
    function skips, takes only their reference lists, and never lets a word of
    their text reach the model. Wikipedia becomes what the policy always said it
    was — a lead generator — instead of either an evidentiary source or nothing.
    """
    if limit <= 0:
        return []
    chosen: List[str] = []
    seen: set[str] = set()
    for c in citations or []:
        url = (c or {}).get("url") or ""
        if not url or EVIDENCE_RANK.get(tier_for(url), WORST_RANK) < 5:
            continue
        domain = registrable_domain(url)
        if domain and domain in seen:
            continue
        seen.add(domain)
        chosen.append(url)
        if len(chosen) >= limit:
            break
    return chosen


# ---------------------------------------------------------------- connections
# Background is not influence. A context node joined by "derives_from" asserts
# the exact fallacy this report exists to expose, so the claim is downgraded to
# the relation that states the honest version.
_CAUSAL_RELATIONS = {"derives_from", "retells", "translates", "responds_to"}

_VALID_RELATIONS = {
    "derives_from", "retells", "translates", "responds_to",
    "contradicts", "contemporaneous", "attests", "provides_context",
    "no_established_link",
}


def validate_connections(
    raw: Any,
    valid_ids: set[str],
    *,
    max_connections: int,
    node_types: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Keep only connections that point at real items and say something.

    A model asked for edges will happily emit them between things it did not
    relate, pointing at ids it invented. Three rules survive that: both ends must
    exist, an item cannot connect to itself, and a pair is kept once. An edge with
    no mechanism is downgraded to ``no_established_link`` rather than dropped —
    "these are merely sequential" is a true and useful statement about a timeline,
    and hiding it would let the reader assume a causal story we never verified.

    A fourth rule needs ``node_types``: an edge running out of a context item
    cannot be causal. "Older religions existed, therefore they shaped this one"
    is the oldest error in comparative history, and a context item exists
    precisely to supply background whose influence was NOT established — so the
    relation is rewritten to ``provides_context`` rather than believed.
    """
    if not isinstance(raw, list):
        return []

    out: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()

    for item in raw:
        if not isinstance(item, dict) or len(out) >= max_connections:
            continue
        a = str(item.get("from_id") or "").strip()
        b = str(item.get("to_id") or "").strip()
        if a not in valid_ids or b not in valid_ids or a == b:
            continue
        if (a, b) in seen or (b, a) in seen:
            continue
        seen.add((a, b))

        relation = item.get("relation")
        if relation not in _VALID_RELATIONS:
            relation = "no_established_link"
        if relation in _CAUSAL_RELATIONS and (node_types or {}).get(a) == "context":
            relation = "provides_context"

        ev = item.get("evidence")
        ev = dict(ev) if isinstance(ev, dict) else {}
        if not str(ev.get("mechanism") or "").strip():
            relation = "no_established_link"
            ev.setdefault(
                "mechanism",
                "No mechanism established — these items are adjacent in time only.",
            )

        item = dict(item)
        item["from_id"], item["to_id"], item["relation"], item["evidence"] = a, b, relation, ev
        out.append(item)

    return out
