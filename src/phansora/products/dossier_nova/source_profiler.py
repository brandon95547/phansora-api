"""
source_profiler.py

Classifies each source document by type and rhetorical role so the dossier
pipeline can preserve source-specific framing and prevent flattening.

Source types:
  - academic, report, essay, transcript, advocacy, case_document, news, reference, unknown

Rhetorical roles:
  - factual, interpretive, policy, narrative, mixed

Each source gets a SourceProfile dataclass containing its classification,
a short summary of its central argument/purpose, and key claims.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SourceProfile:
    """Profile of a single source document."""
    source_label: str                   # filename or label from the caller
    source_type: str = "unknown"        # academic|report|essay|transcript|advocacy|case_document|news|reference|unknown
    rhetorical_role: str = "mixed"      # factual|interpretive|policy|narrative|mixed
    central_argument: str = ""          # one-paragraph summary of the source's thesis/purpose
    key_claims: List[str] = field(default_factory=list)  # up to 8 bullet claims
    char_count: int = 0                 # total characters in the source text


_PROFILE_SYSTEM_PROMPT = """\
You are a document analyst. Given the opening of a source document, \
classify it and extract its core thesis.

Return ONLY valid JSON with this exact structure:

{
  "source_type": "<academic|report|essay|transcript|advocacy|case_document|news|reference|unknown>",
  "rhetorical_role": "<factual|interpretive|policy|narrative|mixed>",
  "central_argument": "<one paragraph describing the source's main thesis or purpose>",
  "key_claims": ["<claim 1>", "<claim 2>", "...up to 8 claims"]
}

Definitions:
- source_type: What kind of document this is.
  - academic: scholarly article, peer-reviewed paper, dissertation
  - report: official report, government document, institutional finding
  - essay: opinion piece, long-form analysis, commentary
  - transcript: interview, deposition, hearing, speech transcript
  - advocacy: activist material, petition, campaign document, policy brief
  - case_document: legal filing, court record, investigation summary
  - news: news article, press release, journalistic piece
  - reference: encyclopedia entry, glossary, FAQ, timeline
  - unknown: cannot determine

- rhetorical_role: How the document frames its content.
  - factual: primarily presents facts, data, chronology
  - interpretive: primarily offers analysis, theory, interpretation
  - policy: primarily advocates for action, reform, or policy change
  - narrative: primarily tells a story or recounts events
  - mixed: combines multiple rhetorical approaches

Rules:
- Base your classification on the actual content, not the filename.
- The central_argument should capture what makes this source's perspective UNIQUE.
- Key claims should be specific, not generic.
- Do NOT wrap JSON in code fences.
"""


def profile_sources(
    sources: List[Dict[str, str]],
    client: Any,
    sample_chars: int = 4000,
) -> List[SourceProfile]:
    """
    Profile each source document by type, role, and central argument.

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
            "Classify this source and extract its thesis."
        )

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
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
                    rhetorical_role=str(data.get("rhetorical_role", "mixed")).lower(),
                    central_argument=str(data.get("central_argument", "")),
                    key_claims=[str(c) for c in (data.get("key_claims") or [])[:8]],
                    char_count=len(text),
                )
        except Exception as e:
            print(f"[PROFILE] Failed to profile '{label}': {e}")

        return SourceProfile(source_label=label, char_count=len(text))

    # Parallel profiling — API calls are I/O-bound
    with ThreadPoolExecutor(max_workers=8) as executor:
        profiles = list(executor.map(_profile_one, sources))

    for p in profiles:
        print(
            f"[PROFILE] {p.source_label}: type={p.source_type}, "
            f"role={p.rhetorical_role}, claims={len(p.key_claims)}, "
            f"chars={p.char_count}"
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
