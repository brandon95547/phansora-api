"""The governing constraint of Research Atlas, as prompt text AND as a checker.

Research Atlas reports what the supplied sources say. It does not decide whether they
are right. That rule is stated to the model in NEUTRALITY_RULES, and it is ENFORCED
afterwards by scan() — because a prompt is a request and this is a requirement.

The distinction the whole module turns on is attribution, not vocabulary. "Conspiracy
theory" is not a forbidden string; it is forbidden as the REPORT'S OWN voice. The same
words are fine when the report is quoting a source that used them, because at that point
they are evidence about the source rather than a verdict on the material:

    BAD   This conspiracy theory alleges that ...
    GOOD  Source 4 describes the account as a "conspiracy theory".

    BAD   This claim has been debunked.
    GOOD  Source 7 disputes the account presented in Source 3.

So scan() does not ask "does this text contain a banned word". It asks "does this
sentence use a banned word WITHOUT handing it to a source", which is the thing that
actually violates the product's promise.

Two deliberate design choices:

  Sentence scope. Attribution has to appear in the same sentence as the term. A
  paragraph-level check would let "Source 3 says X. This is misinformation." pass,
  and that second sentence is exactly the failure mode being guarded against.

  Precision over recall on true/false. Those two are ordinary English ("a true copy",
  "a false positive") and stem-matching them produces noise that trains people to
  ignore the checker. They are matched only in evaluative constructions — "is false",
  "proven true", "false account" — where they ARE a verdict. Everything else uses a
  plain stem, because "debunked" and "pseudoscience" have no innocent register here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# What the model is told
# ---------------------------------------------------------------------------

# Prepended to every Research Atlas prompt. Kept as one block so the rule reaches
# each stage identically -- an extraction call that judges and a rendering call that
# does not would still produce a judging report.
NEUTRALITY_RULES = """NEUTRALITY REQUIREMENT (governs everything below).

You are organizing supplied research material. You are NOT evaluating it.

Report WHAT THE SOURCES SAY. Never decide whether a source is right or wrong.

You MUST NOT, in your own voice, characterize the material as any of:
conspiracy theory, conspiracy, misinformation, disinformation, propaganda,
pseudoscience, fringe, extremist, controversial, debunked, disproven, credible,
unreliable, trustworthy, untrustworthy, true, false, legitimate, illegitimate.

You MUST NOT introduce outside consensus, media positions, institutional positions,
political framing, scientific judgments, or cultural assumptions. If it is not in the
supplied sources, it does not go in the report.

If a SUPPLIED SOURCE uses one of those words, you may include it -- but only with the
attribution made explicit in the same sentence:
  BAD:  This conspiracy theory alleges that ...
  GOOD: Source 4 describes the account as a "conspiracy theory".
  BAD:  This claim has been debunked.
  GOOD: Source 7 disputes the account presented in Source 3.
  BAD:  This proves Person A was involved.
  GOOD: Sources 2 and 6 both connect Person A to Organization B.

Do not assign confidence, reliability, or credibility ratings. Where corroboration
matters, state it as a countable fact -- which sources carry the item -- and let the
reader weigh it.

Never invent a connection because it would make a better story. If the supplied
material does not support a link, record it as an information gap or a research lead
instead of asserting it.

Where sources disagree, PRESERVE the disagreement. Do not resolve it, and do not pick
a winner. State each version with its source.

EXTRACT. ORGANIZE. CROSS-REFERENCE. CONNECT. DOCUMENT. DO NOT JUDGE."""

# Sent back with a violating response when a call is retried.
REWRITE_INSTRUCTION = """Your previous response characterized the material in your own
voice. Rewrite it so that every such description is attributed to the source that made
it ("Source 3 describes ... as ...") or removed entirely. Change nothing else."""


# ---------------------------------------------------------------------------
# What the checker looks for
# ---------------------------------------------------------------------------

# (label, pattern). Stems cover the inflections that carry the same judgment:
# "debunk/debunked/debunking", "credible/credibility", "legitimate/legitimacy".
_STEM_TERMS: Sequence[Tuple[str, str]] = (
    ("conspiracy theory", r"conspirac(?:y|ies)\s+theor(?:y|ies)"),
    ("conspiracy",        r"conspirac(?:y|ies)"),
    ("misinformation",    r"mis-?information"),
    ("disinformation",    r"dis-?information"),
    ("propaganda",        r"propaganda"),
    ("pseudoscience",     r"pseudo-?scien(?:ce|tific)"),
    ("fringe",            r"fringe"),
    ("extremist",         r"extremis(?:t|ts|m)"),
    ("controversial",     r"controversial|controvers(?:y|ies)"),
    ("debunked",          r"debunk(?:s|ed|ing)?"),
    ("disproven",         r"dispro(?:ve|ven|ved|ving|of)"),
    ("credible",          r"credib(?:le|ility)|non-?credible|not\s+credible"),
    ("unreliable",        r"un-?reliab(?:le|ility)"),
    ("trustworthy",       r"trust-?worth(?:y|iness)"),
    ("untrustworthy",     r"un-?trust-?worth(?:y|iness)"),
    ("legitimate",        r"legitima(?:te|cy)"),
    ("illegitimate",      r"il-?legitima(?:te|cy)"),
    # Judgment verbs that smuggle the same verdict in without the noun.
    ("verdict verb",      r"(?:proves|proven|disproves|refutes|refuted|discredits|discredited)"),
)

# true/false only where they are a verdict rather than ordinary English. "a true copy"
# and "a false positive" are not claims about whether the material is correct.
_EVALUATIVE_TF: Sequence[Tuple[str, str]] = (
    ("true/false verdict", r"(?:is|are|was|were|be|been|proven|proved|shown\s+to\s+be|found\s+to\s+be|demonstrably)\s+(?:not\s+)?(?:true|false)"),
    ("true/false verdict", r"(?:true|false)\s+(?:claim|claims|story|stories|account|accounts|statement|statements|allegation|allegations|narrative|narratives)"),
)

BANNED_TERMS: Tuple[str, ...] = tuple(sorted({label for label, _ in _STEM_TERMS} | {"true/false verdict"}))

_PATTERNS: List[Tuple[str, re.Pattern]] = (
    [(label, re.compile(rf"\b(?:{pat})\b", re.IGNORECASE)) for label, pat in _STEM_TERMS]
    + [(label, re.compile(rf"\b(?:{pat})\b", re.IGNORECASE)) for label, pat in _EVALUATIVE_TF]
)

# A term is permitted only when the same sentence names WHO said it.
#
# It is tempting to accept a speech verb as attribution. That is wrong, and the two
# examples in the product spec are precisely why:
#
#     "This conspiracy theory alleges that ..."   <- has "alleges"
#     "This claim has been debunked."             <- has "claim"
#
# Both were written down as things Research Atlas must never say, and both sail past a
# verb-based check: the verb is there, but the attributor is not. An unsourced speech
# verb ("it is alleged", "this claim") is anonymous voice, which is the report's own
# voice wearing a hat. So attribution requires a SOURCE REFERENCE -- a numbered source,
# a known source label, or an explicit "according to" -- and a speech verb on its own
# earns nothing.
#
# This also enforces something worth having independently: every characterization in
# the finished report is traceable, because a sentence that cannot name its source is
# a sentence that does not ship.
_ATTRIBUTION = re.compile(
    r"""
      \baccording\s+to\b
    | \bper\s+(?:source|sources)\b
    | \bcited\s+(?:in|by)\b
    | \bas\s+(?:described|characterized|labeled|labelled|reported|stated)\s+(?:in|by)\b
    | \bsources?\s*\#?\s*\d+          # Source 4 / Sources 2 and 6 / Source #4
    | \bsource\s*[:#]                 # "Source: interview.mp3"
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Sentence-ish segmentation. Markdown bullets and headings are their own units: a bullet
# is a complete assertion, and letting it inherit attribution from the line above it
# would defeat the same-sentence rule.
_SEGMENT = re.compile(r"(?:(?<=[.!?])\s+)|\n+")


@dataclass(frozen=True)
class Violation:
    """One unattributed characterization, with enough context to fix it."""
    term: str          # which rule fired, e.g. "debunked"
    matched: str       # the literal text that matched
    segment: str       # the sentence it appeared in
    path: str = ""     # where it came from, e.g. "claims[3].statement"

    def describe(self) -> str:
        where = f" at {self.path}" if self.path else ""
        return f'unattributed "{self.matched}" ({self.term}){where}: {self.segment.strip()[:160]}'


def _is_attributed(segment: str, source_labels: Iterable[str] = ()) -> bool:
    """Does this sentence hand the characterization to a source?"""
    if _ATTRIBUTION.search(segment):
        return True
    # An exact source label counts even without a speech verb -- "Source 4: ... " in a
    # table cell or list item is attribution, just terser than a sentence.
    for label in source_labels:
        label = (label or "").strip()
        if len(label) >= 3 and label.lower() in segment.lower():
            return True
    return False


def scan(text: str, source_labels: Iterable[str] = (), path: str = "") -> List[Violation]:
    """Every banned characterization used in the report's own voice.

    Attribution is judged per sentence, so a paragraph cannot launder a verdict by
    mentioning a source somewhere earlier in it.
    """
    if not text or not str(text).strip():
        return []
    labels = tuple(source_labels or ())
    out: List[Violation] = []
    for segment in _SEGMENT.split(str(text)):
        if not segment.strip():
            continue
        if _is_attributed(segment, labels):
            # Quoting a source's own wording is reporting, not judging -- and the
            # attribution check above already established which source is speaking.
            continue
        for term, pattern in _PATTERNS:
            for m in pattern.finditer(segment):
                out.append(Violation(term=term, matched=m.group(0), segment=segment, path=path))
    return out


def scan_fields(obj: Any, source_labels: Iterable[str] = (), path: str = "") -> List[Violation]:
    """scan() over every string in a nested structure.

    Walks the parsed object rather than its JSON text so that booleans, keys and
    numbers are never scanned -- `"resolved": false` is a data value, not a verdict,
    and scanning raw JSON would flag it forever.
    """
    out: List[Violation] = []
    if isinstance(obj, str):
        out.extend(scan(obj, source_labels, path))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(scan_fields(v, source_labels, f"{path}.{k}" if path else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.extend(scan_fields(v, source_labels, f"{path}[{i}]"))
    return out


def redact(text: str, source_labels: Iterable[str] = ()) -> str:
    """Last resort: drop the sentences that still judge after a retry.

    Removing a sentence loses information, which is why this runs only after the model
    has been asked once to attribute it properly. Shipping the judgment would violate
    the product's one hard promise; shipping slightly less text does not.
    """
    if not text:
        return text
    kept: List[str] = []
    for segment in re.split(r"(?<=[.!?])\s+", str(text)):
        if segment.strip() and scan(segment, source_labels):
            continue
        kept.append(segment)
    return " ".join(s for s in kept if s.strip()).strip()


def summarize(violations: Sequence[Violation], limit: int = 8) -> str:
    """One-line-per-violation report for logs and the job's warning list."""
    if not violations:
        return "no unattributed characterizations"
    lines = [v.describe() for v in violations[:limit]]
    if len(violations) > limit:
        lines.append(f"... and {len(violations) - limit} more")
    return "\n".join(lines)
