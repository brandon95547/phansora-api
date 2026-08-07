"""
source_profiler.py

Catalogues each supplied source by FORM -- what kind of document it is -- so the
Source Index can tell a reader they are looking at a transcript rather than a filing.

Document forms:
  transcript, article, report, filing, correspondence, reference, book_excerpt,
  webpage, notes, unknown

That is the whole of it. No assessment of a source accompanies its form, and nothing
downstream weighs one form above another: what to make of a transcript is the reader's
call, not the product's. See neutrality.py for why that line is drawn here.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from phansora.shared.ai.deepseek import chat_model


@dataclass
class SourceProfile:
    """What a source IS, structurally. Never what it is trying to achieve.

    The previous taxonomy classified intent -- `advocacy` ("activist material,
    campaign document") and a rhetorical_role of `policy` ("primarily advocates for
    action") -- and asked for "what makes this source's perspective UNIQUE". Those are
    characterizations of a source's agenda, which is exactly the call Research Atlas
    does not make (see neutrality.py). A reader is told the document is a transcript;
    what to make of a transcript is theirs to decide.
    """
    source_label: str                   # filename or label from the caller
    source_type: str = "unknown"        # the document's FORM -- see the prompt below
    subject_matter: str = ""            # what topics it covers, stated flatly
    key_claims: List[str] = field(default_factory=list)  # assertions it makes, in its terms
    char_count: int = 0                 # total characters in the source text


_PROFILE_SYSTEM_PROMPT = """\
Identify what KIND OF DOCUMENT this is and what subject matter it covers.

You are cataloguing an item in a research collection. Describe its form and its
contents. Do NOT assess it, do not describe its purpose, agenda, slant, quality or
reliability, and do not characterize it as credible, unreliable, propaganda, fringe,
conspiratorial, misinformation, or any similar judgment. Those are not yours to make.

Return ONLY valid JSON with this exact structure:

{
  "source_type": "<transcript|article|report|filing|correspondence|reference|book_excerpt|webpage|notes|unknown>",
  "subject_matter": "<one or two flat sentences: what topics, people, places or events this document covers>",
  "key_claims": ["<an assertion the document makes, in its own terms>", "...up to 8"]
}

source_type is the document's FORM, not its viewpoint:
  - transcript: interview, deposition, hearing, speech, recording transcript
  - article: news or magazine piece, blog post, published column
  - report: a study, findings document, institutional or official report
  - filing: legal filing, court record, formal submission
  - correspondence: letter, email, memo
  - reference: encyclopedia entry, glossary, FAQ, chronology
  - book_excerpt: a chapter or passage from a longer work
  - webpage: a page whose form is not otherwise clear
  - notes: informal notes, jottings, working material
  - unknown: cannot determine from the excerpt

Rules:
- Judge the form from the content, not the filename.
- subject_matter states what the document is ABOUT. It does not evaluate it.
- key_claims restate assertions the document makes. Restating is not endorsing, and
  you must not mark any claim as correct or incorrect.
- Do NOT wrap JSON in code fences.
"""


def profile_sources(
    sources: List[Dict[str, str]],
    client: Any,
    sample_chars: int = 4000,
    max_workers: int = 8,
) -> List[SourceProfile]:
    """
    Catalogue each source by document form and subject matter.

    Args:
        sources: List of dicts with keys 'label' and 'text'.
        client: OpenAI-compatible API client.
        sample_chars: How many chars from the start of each source to send.

    Returns:
        List of SourceProfile objects, one per source.
    """
    if not sources:
        return []

    def _profile_one(source: Dict[str, str]) -> SourceProfile:
        label = source.get("label", "unknown")
        text = source.get("text", "")
        sample = text[:sample_chars].strip()

        if not sample:
            return SourceProfile(source_label=label, char_count=len(text))

        prompt = (
            f"Source filename: {label}\n\n"
            f"Source text (first {len(sample)} characters):\n\n"
            f"{sample}\n\n"
            "Identify this document's form and state what it covers."
        )

        try:
            response = client.chat.completions.create(
                model=chat_model("RESEARCH_ATLAS_MODEL"),
                messages=[
                    {"role": "system", "content": _PROFILE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            raw = (response.choices[0].message.content or "").strip()
            data = _extract_json(raw)
            if data:
                return SourceProfile(
                    source_label=label,
                    source_type=str(data.get("source_type", "unknown")).lower(),
                    subject_matter=str(data.get("subject_matter", "")),
                    key_claims=[str(c) for c in (data.get("key_claims") or [])[:8]],
                    char_count=len(text),
                )
        except Exception as e:
            print(f"[PROFILE] Failed to profile '{label}': {e}")

        return SourceProfile(source_label=label, char_count=len(text))

    # Parallel profiling — API calls are I/O-bound
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        profiles = list(executor.map(_profile_one, sources))

    for p in profiles:
        print(
            f"[PROFILE] {p.source_label}: form={p.source_type}, "
            f"claims={len(p.key_claims)}, chars={p.char_count}"
        )

    return profiles


def _extract_json(raw: str) -> Optional[Dict]:
    """Try to parse JSON from LLM output, tolerating fences and stray text."""
    if not raw:
        return None
    raw = raw.strip()
    # Remove code fences if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return None
