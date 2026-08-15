"""Prompt templates for the trace pipeline."""
from __future__ import annotations

# ---------------------------------------------------------------------------
# The evidence doctrine every stage of the pipeline works under. Chrono Origin
# ranks a claim by the evidence you can follow backwards from it, never by the
# reputation of whoever repeated it last.
# ---------------------------------------------------------------------------
SOURCE_HIERARCHY = """\
SOURCE HIERARCHY — work down this list in order, and record which tier each source sits in:

1. PRIMARY EVIDENCE (highest priority). Original or earliest surviving documents, manuscripts,
   inscriptions, archaeological reports, government and court records, speeches, letters,
   photographs, maps, census records, contemporary newspapers. For ancient subjects you MUST
   record the estimated ORIGINAL COMPOSITION date separately from the date of the EARLIEST
   SURVIVING PHYSICAL COPY — they are frequently centuries apart, and conflating them is the
   single most common error in this kind of work.
2. AUTHORITATIVE REPOSITORIES. Museums, archives, libraries, manuscript collections,
   universities, archaeological databases, national archives, critical editions. Use these to
   establish provenance: what exactly is this object, where did it come from, how old is it?
3. ACADEMIC SCHOLARSHIP. Peer-reviewed papers, university-press books, critical editions,
   specialist historians, archaeologists, linguists. Their conclusions are HISTORICAL
   INTERPRETATION OR INFERENCE and must be labeled as such — never silently promoted to fact.
4. INDEPENDENT CORROBORATION. Actively look for evidence that did NOT originate from the same
   information chain. Ten websites repeating one article count as ONE source, not ten
   confirmations. Say so explicitly when you detect it.
5. MAINSTREAM AND ALTERNATIVE SOURCES. Search both. Do not judge credibility by whether the
   outlet is CNN, Fox, BBC, an independent researcher, a Substack, or an alternative
   publication. Ask only: what evidence does THIS claim cite, and can that citation be followed
   backward?
6. LOW-AUTHORITY DISCOVERY SOURCES. Wikipedia, blogs, forums, social media, video, general
   websites. Useful as leads; they must not become the evidentiary foundation of a trace.
   Follow their references backward to stronger material and cite that instead.

Absence of evidence is a finding. "None identified" is always a better answer than a guess,
an assumed source, or a citation you did not actually see.
"""


DECOMPOSE_PROMPT = """\
You are a research planner. The user wants to trace the EARLIEST KNOWN ORIGIN of a story, myth,
or historical event, plus its evolution over time across cultures and texts.

Story title: {title}
Optional context: {context}

{source_hierarchy}

Produce a JSON object with:
- "entities": 3-8 key proper nouns / concepts to anchor searches.
- "queries": {max_queries} diverse web search queries. Allocate them across the hierarchy above:
  (a) the earliest written attestations and primary documents, (b) manuscript / artefact
  provenance held by named repositories (which museum, archive, shelfmark), (c) academic dating
  and historiography, (d) older parallel traditions or precursors, (e) at least one query hunting
  for INDEPENDENT corroboration from a different information chain, and (f) at least one query
  hunting for evidence that CONTRADICTS or disputes the standard account. Queries must be
  self-contained (no pronouns), in English unless another language is clearly required.
- "domains_of_interest": short list of fields (e.g. "Assyriology", "biblical archaeology",
  "ufology", "folklore studies").

Return ONLY JSON. No prose.
"""


SEARCH_PROMPT = """\
You are a research assistant performing a SINGLE web search to help trace the origin of:
"{title}" {context_clause}

Search query: {query}

{source_hierarchy}

Search the web for this query. Then write a concise (<= 300 words) factual summary of what the
sources say that is RELEVANT to dating the earliest origin or any historical retelling of this story.

In the summary you must:
- Name specific dates, eras, manuscript names, shelfmarks, repositories, authors, and cultures
  whenever the sources do.
- Separate COMPOSITION date from EARLIEST SURVIVING COPY whenever both are discussed.
- Say which tier of the hierarchy each source belongs to (primary / repository / academic /
  press / low-authority).
- Flag when several results are clearly repeating one upstream report, and name that upstream
  source if you can identify it.
- State plainly when something is NOT found, rather than filling the gap.

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
      "year": <signed integer or null>,        // negative = BCE; the COMPOSITION date
      "era_label": <string or null>,           // use when year unknown
      "precision": "exact|year|decade|century|millennium|era|unknown",
      "source_title": <string>,                // manuscript / book / event name
      "claim": <one sentence>,
      "citations": [<url>, ...],
      "confidence": <0..1>,
      "source_tier": "primary|repository|academic|press|low_authority|unknown",
      "surviving_copy": <string or null>,      // oldest EXISTING copy + its date, if the notes say
      "chain": <string or null>                // name the upstream source if this just repeats one
    }}
  ]
}}

Rules:
- Only include mentions actually supported by the notes; never infer a source that is not there.
- "year" is when the source was COMPOSED. If the notes only give the date of a surviving copy,
  put that in "surviving_copy" and leave "year" null unless composition is separately stated.
- Keep "surviving_copy" and "chain" under 15 words each.
- Return ONLY JSON.
"""


RECURSE_PROMPT = """\
We are trying to push the origin of "{title}" further back in time.

Current earliest known mention:
- year: {year}
- era_label: {era_label}
- source: {source_title}
- claim: {claim}

Generate {max_queries} NEW web search queries that specifically hunt for OLDER predecessors,
parallel traditions, source materials, or oral-tradition antecedents that PREDATE the above.
Prioritise queries likely to surface primary documents, named manuscripts and their holding
repositories, excavation reports, and critical editions rather than general write-ups. Include at
least one query aimed at INDEPENDENT corroboration from a different information chain, and one
aimed at scholarship that DISPUTES the dating above.

Avoid repeating any of these already-tried queries:
{prior_queries}

Return JSON: {{"queries": [<string>, ...]}}. Only JSON.
"""


SYNTHESIZE_PROMPT = """\
You are writing the final trace report for the story "{title}".

You have collected these dated mentions across multiple research rounds:
{mentions_block}

Available citations:
{citations_block}

{source_hierarchy}

Every entry carries an "evidence" dossier — a plain, arguable statement of what actually backs
the claim. Fill each field honestly; "None identified" is a legitimate and valuable answer, and
is always better than a plausible-sounding invention.

The evidence dossier shape (same for the origin and every timeline entry):
{{
  "claim": <the claim restated as ONE testable proposition, e.g. "X was born in Y">,
  "earliest_supporting_source": <named source + what kind of thing it is, or "None identified">,
  "estimated_source_date": <when that source was COMPOSED; a range is fine, or "Unknown">,
  "earliest_surviving_copy": <oldest physically existing copy + its date + repository if known,
                              or "None identified">,
  "contemporary_evidence": <evidence created at the time of the event itself, or "None identified">,
  "independent_corroboration": <support NOT descending from the same information chain, or
                                "None identified"; if many sources trace to one report, say so>,
  "contradictory_evidence": <sources that dispute it, or "None identified">,
  "evidence_type": "primary_document|archaeological|contemporary_record|near_contemporary_account|
                    later_historical_account|scholarly_inference|tradition|disputed|absent",
  "confidence_label": "high|moderate|low|speculative",
  "why": <1-2 sentences of plain language explaining the assessment>,
  "missing_piece": <the single absent piece of evidence that most limits this claim>
}}

Produce a JSON object:
{{
  "origin": {{
    "year": <signed int or null>,
    "era_label": <string or null>,
    "precision": "exact|year|decade|century|millennium|era|unknown",
    "source_title": <string>,
    "summary": <2-4 sentences explaining why this is the earliest defensible origin>,
    "citations": [<url>, ...],
    "confidence": <0..1>,
    "evidence": {{ ...dossier... }}
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
      "confidence": <0..1>,
      "evidence": {{ ...dossier... }}
    }}
  ],
  "source_tiers": {{ "<url>": "primary|repository|academic|press|low_authority|unknown", ... }},
  "reasoning": <short paragraph explaining the chain of evidence and any uncertainty>,
  "confidence": <0..1>
}}

Rules:
- Every claim must be backed by at least one citation URL from the provided list.
- Prefer the OLDEST well-attested source as origin; if it's truly oral / prehistoric, set year=null
  and use an era_label.
- "year" is the COMPOSITION date. Never present the date of a surviving copy as the date of the
  source itself; that distinction belongs in "earliest_surviving_copy".
- A scholar's conclusion is "scholarly_inference", not a record. A transmitted belief with no
  documentary trail is "tradition". Only use "primary_document", "archaeological" or
  "contemporary_record" when the evidence genuinely is one.
- Sources that repeat one upstream report count as ONE. Do not describe them as corroboration.
- "confidence_label" reflects how well-attested the claim is, and must be consistent with the
  numeric "confidence" (high >= 0.75, moderate 0.5-0.75, low 0.3-0.5, speculative < 0.3).
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

{source_hierarchy}

Search the web to surface specific, dated sub-events tightly related to that anchor:
contemporaneous retellings, immediate predecessors or successors, manuscript variants,
translations, recensions, archaeological finds, related contemporary events, named
people involved, or documented influences.

Write a concise (<= 350 words) factual summary mentioning specific dates, manuscript
names, shelfmarks, repositories, authors, places, and cultures whenever the sources do.
Separate composition dates from the dates of surviving copies, note which tier of the
hierarchy each source sits in, and flag results that merely repeat one upstream report.
Do not speculate beyond the cited sources.
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
      "year": <signed integer or null>,        // the COMPOSITION date
      "era_label": <string or null>,
      "precision": "exact|year|decade|century|millennium|era|unknown",
      "source_title": <string>,
      "claim": <one sentence explaining how this sub-event relates to the anchor>,
      "citations": [<url>, ...],
      "confidence": <0..1>,
      "evidence": {{
        "claim": <the claim restated as ONE testable proposition>,
        "earliest_supporting_source": <named source + what it is, or "None identified">,
        "estimated_source_date": <when it was COMPOSED, or "Unknown">,
        "earliest_surviving_copy": <oldest existing copy + date + repository, or "None identified">,
        "contemporary_evidence": <evidence from the time of the event, or "None identified">,
        "independent_corroboration": <support from a different information chain, or "None identified">,
        "contradictory_evidence": <disputing sources, or "None identified">,
        "evidence_type": "primary_document|archaeological|contemporary_record|near_contemporary_account|later_historical_account|scholarly_inference|tradition|disputed|absent",
        "confidence_label": "high|moderate|low|speculative",
        "why": <1-2 plain sentences>,
        "missing_piece": <the absent evidence that most limits this claim>
      }}
    }}
  ]
}}

Rules:
- Return AT MOST {max_events} events.
- Every event must be supported by the notes and cite at least one URL.
- Each event must be clearly tied to the anchor (same period, same lineage, direct
  cause/effect, manuscript variant, etc.) - do NOT repeat the anchor itself.
- Fill the evidence dossier from the notes only. "None identified" is correct and expected
  when the notes do not establish something; never invent a source or a date to fill a field.
- A historian's conclusion is "scholarly_inference", not a record; a transmitted belief with no
  documentary trail is "tradition".
- Sources that repeat one upstream report are ONE source, not corroboration.
- Order events chronologically, oldest first.
- Return ONLY JSON.
"""