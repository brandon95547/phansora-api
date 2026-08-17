"""The source policy, as code rather than as advice.

The doctrine in ``prompts.py`` has always said the right thing: Wikipedia, blogs
and general websites are leads, never the evidentiary foundation of a trace, and
their references should be followed backward to something that can be argued
with. A trace for Jesus Christ nonetheless came back resting on Wikipedia, with
the Gospel of Mark treated as evidence for the events it narrates — because
nothing in the pipeline could tell whether the instruction had been followed.
A tier was a decoration printed next to a citation; no code read it back.

So the policy lives here, where it binds. Five tiers, the same five the product
promises its readers:

  1. PRIMARY EVIDENCE — the manuscript, inscription, excavation report, court
     record, census, letter, photograph, or a newspaper published AT THE TIME —
     and the archive, library or museum that holds and publishes it.
  2. SCHOLARLY ANALYSIS — peer review, university presses, critical editions.
     Conclusions, clearly labelled as conclusions.
  3. SCHOLARLY DISCOVERY — Crossref, DOIs, catalogue records. These establish
     that a work exists and who wrote it. A DOI is not an argument.
  4. INSTITUTIONAL SECONDARY — a university, museum or agency explaining
     something it did not itself record. Good context; follow its citations.
  5. GENERAL WEB — news, blogs, video, forums, wikis. Leads.

Two rules give the tiers teeth, and both live in this module:

``role_for`` decides whether a citation may be the BASIS of a claim (tiers 1-3)
or is only a lead (tiers 4-5). ``cap_evidence_type`` then stops a claim
describing itself as a primary document when nothing better than a wiki stands
under it. Because claim_class is derived from evidence_type, that cap reaches
the green "direct evidence" marker on the board without touching the renderer.

Nothing here removes a citation or drops a claim. An over-filtered trace is a
worse answer than an honestly-labelled weak one, and a claim with nothing behind
it is a finding worth showing: it comes back UNKNOWN, not missing.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional

# --------------------------------------------------------------------- tiers
# The tier strings are unchanged from the first version of this product, and
# stay unchanged deliberately: every trace ever run is stored as JSON in
# `chrono_origin_jobs.result`, is never rewritten, and still has to render.
# What is new is the RANK — the number the policy actually reasons about.
EVIDENCE_RANK: Dict[str, int] = {
    "primary": 1,
    "repository": 1,          # the holding institution is tier 1 with the object
    "academic": 2,
    "reference_index": 3,     # Crossref / DOI / catalogue metadata
    "institutional": 4,
    "press": 5,               # promoted to 1 only when contemporary — see rank_for
    "low_authority": 5,
    "unknown": 5,             # unclassified cannot be the basis of anything
}
WORST_RANK = 5

# Ranking for READING, which is a different question with a different answer.
# `unknown` sits above `low_authority` here on purpose: opening an unrecognised
# page is how we find out it is a national archive on an odd domain, whereas
# opening a wiki tells us nothing we did not already assume. Ranking `unknown`
# as tier 5 for reading too would quietly stop the pipeline reading good sources.
TIER_ORDER = (
    "primary", "repository", "academic", "reference_index",
    "institutional", "press", "unknown", "low_authority",
)
READ_RANK = {t: i for i, t in enumerate(TIER_ORDER)}

VALID_TIERS = set(EVIDENCE_RANK)

# Host signals. A URL is a weak signal but a cheap one, and it is the only signal
# available when the model declines to label a source.
LOW_AUTHORITY_HOSTS = (
    "wikipedia.org", "wikimedia.org", "wikiwand.com", "fandom.com", "reddit.com",
    "quora.com", "medium.com", "substack.com", "blogspot.com", "wordpress.com",
    "tumblr.com", "youtube.com", "youtu.be", "twitter.com", "x.com",
    "facebook.com", "instagram.com", "tiktok.com", "pinterest.com",
    "stackexchange.com", "answers.com", "britannica.com", "history.com",
    "biography.com", "ancient.eu", "worldhistory.org",
    # Generated and devotional encyclopaedias. Listed for the same reason
    # Wikipedia is: not because of who runs them, but because the class of
    # thing they are — a tertiary write-up with no editorial chain a reader can
    # follow — is a lead. Tier 5 is where every unrecognised host already lands;
    # naming these buys the READ budget back, since READ_RANK deliberately opens
    # unclassified pages and would otherwise spend a fetch on one of these.
    "grokipedia.com", "thoughtco.com", "learnreligions.com", "christianity.com",
    "gotquestions.org", "biblestudytools.com", "crosswalk.com", "allaboutgod.com",
)
REPOSITORY_HOSTS = (
    ".museum", "britishmuseum.org", "bl.uk", "loc.gov", "archives.gov",
    "nationalarchives", "bnf.fr", "vatlib.it", "metmuseum.org", "getty.edu",
    "smithsonian", "digitallibrary", "archive.org", "perseus.tufts.edu",
    "hathitrust.org", "e-codices.ch", "csntm.org", "papyri.info",
    "britishlibrary", "bodleian.ox.ac.uk", "dss.collections.imj.org.il",
)
ACADEMIC_HOSTS = (
    ".edu", ".ac.uk", "jstor.org", "cambridge.org", "oup.com",
    "springer.com", "tandfonline.com", "sciencedirect.com", "brill.com",
    "degruyter.com", "jhu.edu", "persee.fr", "openedition.org", "arxiv.org",
    "zenodo.org", "sbl-site.org", "muse.jhu.edu",
)
# Tier 3 is its own thing and used to hide inside the academic list: a DOI
# resolver proves a paper exists and who wrote it. It says nothing about whether
# the paper is right, which is exactly why the mandate gives it a tier of its own.
#
# Author-upload platforms belong here rather than with peer review, and that is
# the correction, not a demotion for its own sake: an Academia.edu or
# ResearchGate URL establishes that a paper exists and who claims to have
# written it. Nothing about the upload says it was reviewed, or published, or
# is the version of record — the authority, when there is any, belongs to
# wherever it actually appeared, which is what tier 3 exists to send a reader
# looking for.
REFERENCE_INDEX_HOSTS = (
    "doi.org", "crossref.org", "openalex.org", "worldcat.org", "semanticscholar.org",
    "orcid.org", "scholar.google", "pubmed.ncbi.nlm.nih.gov",
    "academia.edu", "researchgate.net", "ssrn.com",
)
# Tier 4: institutions writing ABOUT things rather than holding them.
INSTITUTIONAL_HOSTS = (
    ".gov", ".gov.uk", ".mil", "who.int", "un.org", "europa.eu",
    "nasa.gov", "si.edu", "nps.gov", "royalsociety.org", "britac.ac.uk",
)
# Journalism. Its tier depends entirely on WHEN it was published relative to the
# claim, so the list exists to recognise the class, not to rank the outlet. The
# doctrine's rule stands: never judge by masthead, only by what can be followed
# backward.
PRESS_HOSTS = (
    "nytimes.com", "washingtonpost.com", "bbc.co.uk", "bbc.com", "theguardian.com",
    "reuters.com", "apnews.com", "cnn.com", "foxnews.com", "nbcnews.com",
    "abcnews.go.com", "cbsnews.com", "telegraph.co.uk", "thetimes.co.uk",
    "wsj.com", "economist.com", "newsweek.com", "time.com", "npr.org",
    "aljazeera.com", "smithsonianmag.com", "nationalgeographic.com",
    "newspapers.com", "chroniclingamerica.loc.gov",
)

_ABSENCE = ("none identified", "none", "n/a", "na", "not identified", "not found", "unknown", "")

# A year sitting in a URL path or query — /2024/03/, -1963-, ?year=1904. Free,
# and right often enough on news URLs to decide the contemporaneity question
# without spending a token on it.
_URL_YEAR = re.compile(r"(?:^|[/\-_?&=])((?:1[5-9]|20)\d{2})(?:[/\-_?&]|$)")
_TEXT_YEAR = re.compile(r"\b((?:1[0-9]|20)\d{2})\b")


def is_absent(value: Any) -> bool:
    """True when a dossier field is an honest 'nothing here' rather than content."""
    return str(value or "").strip().lower() in _ABSENCE


def default_tier(url: str) -> str:
    """Classify a URL by host alone, for when the model does not label it.

    Order matters: the low-authority check runs first so that a wiki hosted on a
    university domain is still a wiki, and reference indexes are tested before
    the academic list they used to hide in.
    """
    u = (url or "").lower()
    if any(h in u for h in LOW_AUTHORITY_HOSTS):
        return "low_authority"
    if any(h in u for h in REFERENCE_INDEX_HOSTS):
        return "reference_index"
    if any(h in u for h in REPOSITORY_HOSTS):
        return "repository"
    if any(h in u for h in ACADEMIC_HOSTS):
        return "academic"
    if any(h in u for h in PRESS_HOSTS):
        return "press"
    if any(h in u for h in INSTITUTIONAL_HOSTS):
        return "institutional"
    return "unknown"


def resolve_tier(claimed: Any, url: str) -> str:
    """The tier of a URL, taking the model's label but never above its host.

    Every tier in a trace starts as an assertion by a language model, and the one
    assertion that must not be taken on trust is the one that promotes a lead.
    A trace came back having READ a Wikipedia file-description page and an
    AI-generated wiki, both of which the reader is supposed to skip: the host
    check was there, but the model had labelled them ``primary``, and a claimed
    tier was allowed to win. So a known lead host is a floor, not a starting
    point — nothing a model says about a wiki makes it stop being a wiki.

    Demotion is still free in the other direction: an unrecognised host that the
    model recognises as a national archive keeps the tier the model gave it,
    because that is a promotion the host signal genuinely cannot make.
    """
    host = default_tier(url)
    tier = claimed if claimed in EVIDENCE_RANK else host
    if host == "low_authority":
        return "low_authority"
    if EVIDENCE_RANK[tier] < EVIDENCE_RANK[host] and host in ("reference_index", "press"):
        # Author-upload platforms and news sites, claimed as scholarship or as
        # the record itself. Both are the same mistake: crediting the container
        # with the authority of whatever it happens to be carrying.
        return host
    return tier


def url_year(url: str) -> Optional[int]:
    """A publication year read out of the URL, when one is sitting there."""
    m = _URL_YEAR.search(url or "")
    return int(m.group(1)) if m else None


def parse_year(value: Any) -> Optional[int]:
    """First plausible year in a free-text date ('March 1963', 'c. 1904')."""
    if isinstance(value, int):
        return value
    m = _TEXT_YEAR.search(str(value or ""))
    return int(m.group(1)) if m else None


# ----------------------------------------------------------- relational rank
# Whether a source counts as primary evidence is not a property of its domain.
# A newspaper printed the week of an event is a primary record of it; the same
# paper writing about that event sixty years later is a secondary account, and
# the same masthead either way. Tier is a relation between a source and a claim.
CONTEMPORARY_WINDOW_YEARS = 5
_DATED_PRECISIONS = ("exact", "year", "decade")


def rank_for(
    tier: Any,
    *,
    published_year: Optional[int] = None,
    claim_year: Optional[int] = None,
    claim_precision: Any = None,
    url: str = "",
) -> int:
    """The evidentiary rank (1-5) of one citation UNDER ONE CLAIM.

    Promotion is rare and guarded; demotion is cheap. That asymmetry is the whole
    design: wrongly promoting a source hands tier 1 to exactly the class of
    source this policy exists to demote, while wrongly leaving one at tier 5
    costs a claim some confidence it might have earned.
    """
    tier = tier if tier in EVIDENCE_RANK else "unknown"
    base = EVIDENCE_RANK[tier]
    host_tier = default_tier(url) if url else "unknown"

    # Demote a self-declared primary source that is plainly a write-up. This is
    # the "History.com is a primary document" case, and the model asserts it
    # more often than one would hope.
    if base == 1 and host_tier in ("low_authority", "press"):
        base = EVIDENCE_RANK[host_tier]

    # Promote a record published contemporaneously with what it records.
    # Requires a real publication year and a claim dated firmly enough to compare
    # against — never on an inference about either.
    if base >= 4 and tier in ("press", "institutional", "unknown"):
        if (
            published_year is not None
            and claim_year is not None
            and str(claim_precision or "") in _DATED_PRECISIONS
            and abs(published_year - claim_year) <= CONTEMPORARY_WINDOW_YEARS
        ):
            return 1

    # Scholarship is never aged out. A 2024 paper about 1963 is analysis, and
    # analysis does not decay: demoting it for recency would invert the hierarchy.
    return base


def role_for(rank: int) -> str:
    """Whether a citation can be the BASIS of a claim, or is only a lead.

    Tier 4 is a lead on purpose. An institution explaining something it did not
    record is exactly the case the mandate says to follow backward rather than
    rest on.
    """
    return "evidence" if rank <= 3 else "discovery"


def best_rank(ranks: Iterable[int]) -> int:
    """The strongest rank in a set, or the worst when the set is empty."""
    ranks = [r for r in ranks if isinstance(r, int)]
    return min(ranks) if ranks else WORST_RANK


# ------------------------------------------------------------ the two rules
# Evidence types that assert the support IS the record, rather than an account
# of it. These are the claims a tier-5 citation cannot buy.
_DIRECT_TYPES = ("primary_document", "archaeological", "contemporary_record")


def cap_evidence_type(evidence_type: Any, rank: int) -> str:
    """Hold an evidence type to what its citations can actually support.

    The Jesus trace is the case this exists for. "Jesus was crucified c. 30 CE",
    supported by a Wikipedia article, came back as `primary_document` and so
    rendered on the board as direct evidence, in green, next to a dot. No page
    that says a thing is thereby evidence for it. At tier 4-5 the strongest
    honest description is a later account; at tier 3 it is an inference, because
    catalogue metadata establishes that a work exists, not that it is right.

    claim_class is derived from evidence_type, so capping here reaches the
    board's marker, the dossier's "How we know" row and the PDF at once.
    """
    et = str(evidence_type or "scholarly_inference")
    if et not in _DIRECT_TYPES:
        return et
    if rank >= 4:
        return "later_historical_account"
    if rank == 3:
        return "scholarly_inference"
    return et


def verification_for(
    *,
    best_evidence_rank: int,
    has_citations: bool,
    read_from_source: bool,
    earliest_supporting_source: Any = None,
) -> str:
    """Did we actually check? Derived, never asked of the model.

    Separate from claim_class, which asks what the evidence entitles us to say.
    A claim can be a sound inference nobody verified, and the reader is owed both
    facts rather than an average of them.
    """
    if not has_citations or (best_evidence_rank >= 5 and is_absent(earliest_supporting_source)):
        return "unknown"
    if best_evidence_rank <= 2 and read_from_source:
        return "verified"
    return "unverified"
