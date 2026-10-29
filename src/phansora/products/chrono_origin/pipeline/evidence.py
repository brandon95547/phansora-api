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
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import urlsplit

from ..models import CLAIM_CLASS_BY_EVIDENCE_TYPE
from .source_policy import EVIDENCE_RANK, WORST_RANK

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
    # by EVIDENCE_RANK, so the tier printed on a chain means the same thing as the
    # tier printed on the citations inside it.
    for group in groups.values():
        best = min(
            (tier_for(u) for u in group["urls"]),
            key=lambda t: EVIDENCE_RANK.get(t, WORST_RANK),
            default="unknown",
        )
        group["tier"] = best

    return chain_of, list(groups.values())
