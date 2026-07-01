"""Prompt templates for the trace pipeline."""
from __future__ import annotations

DECOMPOSE_PROMPT = """\
You are a research planner. The user wants to trace the EARLIEST KNOWN ORIGIN of a story, myth,
or historical event, plus its evolution over time across cultures and texts.

Story title: {title}
Optional context: {context}

Produce a JSON object with:
- "entities": 3-8 key proper nouns / concepts to anchor searches.
- "queries": {max_queries} diverse web search queries that surface (a) the earliest written
  attestations, (b) older parallel myths or precursors, (c) academic / historiographic dating, and
  (d) primary source manuscripts. Queries must be self-contained (no pronouns), in English unless
  another language is clearly required.
- "domains_of_interest": short list of fields (e.g. "Assyriology", "biblical archaeology",
  "ufology", "folklore studies").

Return ONLY JSON. No prose.
"""


SEARCH_PROMPT = """\
You are a research assistant performing a SINGLE web search to help trace the origin of:
"{title}" {context_clause}

Search query: {query}

Search the web for this query. Then write a concise (<= 250 words) factual summary of what the sources say
that is RELEVANT to dating the earliest origin or any historical retelling of this story.
Always mention specific dates, eras, manuscript names, authors, and cultures when sources do.
Do not speculate beyond what the cited sources state.
"""


EXTRACT_PROMPT = """\
From the research notes below, extract every distinct dated mention (or era-tagged mention) of
the story "{title}". Each mention must be tied to at least one citation URL from the notes.

Research notes:
---
{notes}
---

Available citations (use these URLs verbatim):
{citations_block}

Return JSON:
{{
  "mentions": [
    {{
      "year": <signed integer or null>,        // negative = BCE
      "era_label": <string or null>,           // use when year unknown
      "precision": "exact|year|decade|century|millennium|era|unknown",
      "source_title": <string>,                // manuscript / book / event name
      "claim": <one sentence>,
      "citations": [<url>, ...],
      "confidence": <0..1>
    }}
  ]
}}
Only include mentions actually supported by the notes. Return ONLY JSON.
"""


RECURSE_PROMPT = """\
We are trying to push the origin of "{title}" further back in time.

Current earliest known mention:
- year: {year}
- era_label: {era_label}
- source: {source_title}
- claim: {claim}

Generate {max_queries} NEW web search queries that specifically hunt for OLDER predecessors,
parallel myths, source materials, or oral-tradition antecedents that PREDATE the above. Avoid
repeating any of these already-tried queries:
{prior_queries}

Return JSON: {{"queries": [<string>, ...]}}. Only JSON.
"""


SYNTHESIZE_PROMPT = """\
You are writing the final trace report for the story "{title}".

You have collected these dated mentions across multiple research rounds:
{mentions_block}

Available citations:
{citations_block}

Produce a JSON object:
{{
  "origin": {{
    "year": <signed int or null>,
    "era_label": <string or null>,
    "precision": "exact|year|decade|century|millennium|era|unknown",
    "source_title": <string>,
    "summary": <2-4 sentences explaining why this is the earliest defensible origin>,
    "citations": [<url>, ...],
    "confidence": <0..1>
  }},
  "timeline": [
    // chronological, oldest first; one entry per significant retelling / mutation
    {{
      "year": <signed int or null>,
      "era_label": <string or null>,
      "precision": "...",
      "source_title": <string>,
      "claim": <one sentence describing how this version changed or carried the story>,
      "citations": [<url>, ...],
      "confidence": <0..1>
    }}
  ],
  "reasoning": <short paragraph explaining the chain of evidence and any uncertainty>,
  "confidence": <0..1>
}}

Rules:
- Every claim must be backed by at least one citation URL from the provided list.
- Prefer the OLDEST well-attested source as origin; if it's truly oral / prehistoric, set year=null
  and use an era_label.
- Deduplicate mentions that describe the same source.
- Return ONLY JSON.
"""


EXPAND_SEARCH_PROMPT = """\
You are a research assistant performing focused web searches to find sub-events that
happened in, around, or directly because of the following moment in the broader history
of the story "{story_title}"{context_clause}.

Anchor item being expanded:
- when: {when}
- source / event: {parent_source_title}
- claim: {parent_claim}

Search the web to surface specific, dated sub-events tightly related to that anchor:
contemporaneous retellings, immediate predecessors or successors, manuscript variants,
translations, recensions, archaeological finds, related contemporary events, named
people involved, or documented influences. Prefer primary and academic sources.

Write a concise (<= 350 words) factual summary mentioning specific dates, manuscript
names, authors, places, and cultures whenever the sources do. Do not speculate beyond
the cited sources.
"""


EXPAND_EXTRACT_PROMPT = """\
From the research notes below, extract distinct dated sub-events that are tightly
related to this anchor in the history of "{story_title}":

Anchor:
- when: {when}
- source / event: {parent_source_title}
- claim: {parent_claim}

Research notes:
---
{notes}
---

Available citations (use these URLs verbatim):
{citations_block}

Return JSON:
{{
  "events": [
    {{
      "year": <signed integer or null>,
      "era_label": <string or null>,
      "precision": "exact|year|decade|century|millennium|era|unknown",
      "source_title": <string>,
      "claim": <one sentence explaining how this sub-event relates to the anchor>,
      "citations": [<url>, ...],
      "confidence": <0..1>
    }}
  ]
}}

Rules:
- Return AT MOST {max_events} events.
- Every event must be supported by the notes and cite at least one URL.
- Each event must be clearly tied to the anchor (same period, same lineage, direct
  cause/effect, manuscript variant, etc.) - do NOT repeat the anchor itself.
- Order events chronologically, oldest first.
- Return ONLY JSON.
"""