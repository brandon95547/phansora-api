"""Prompt templates for the trace pipeline.

A note on cost, because it shapes every template here. The doctrine below is the
most expensive text in the product: it used to be pasted into all ~20 search
calls of a trace, where it could not be acted on — a search step summarising five
snippets cannot weigh a manuscript's provenance, so it was paying full price for
instructions it had no way to follow.

So the doctrine is split. The full hierarchy goes only to the two stages that
exercise judgement: planning what to look for, and deciding what the evidence
supports. Everything else gets SEARCH_DOCTRINE, the short form covering only what
a summariser can actually do. Same behaviour, a fraction of the tokens, and the
savings pay for reading real source pages instead.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# The evidence doctrine. Chrono Origin ranks a claim by the evidence you can
# follow backwards from it, never by the reputation of whoever repeated it last.
# Injected into DECOMPOSE and SYNTHESIZE only — the two stages that can act on it.
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


# The short form, for stages that summarise rather than judge. Everything a
# search step can actually act on, and nothing it cannot.
SEARCH_DOCTRINE = """\
EVIDENCE RULES for this summary:
- Prefer original documents, artefacts and the repositories holding them over write-ups about them.
- Give the COMPOSITION date and the EARLIEST SURVIVING COPY date separately whenever both appear.
- Judge each source by the evidence it cites, not by its brand — mainstream and alternative alike.
- If several results repeat one upstream report, say so and name it: that is ONE source, not several.
- State plainly what is NOT found. Never fill a gap with a plausible guess.
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

{search_doctrine}

Search the web for this query. Then write a concise (<= 300 words) factual summary of what the
sources say that is RELEVANT to dating the earliest origin or any historical retelling of this story.

Name specific dates, eras, manuscript names, shelfmarks, repositories, authors and cultures
whenever the sources do, and tag each source with its tier (primary / repository / academic /
press / low-authority). Do not speculate beyond what the cited sources state.
"""


# Extract and plan-the-next-round in one call. These were two calls until the
# merge: the model that has just read the notes is better placed to say what is
# still missing than a second call given only the earliest year, and the round's
# planning now costs nothing extra.
EXTRACT_PROMPT = """\
From the research material below, extract every distinct dated mention (or era-tagged mention) of
the story "{title}". Each mention must be tied to at least one citation URL from the material.

Research notes:
---
{notes}
---
{pages_block}
Available citations (use these URLs verbatim):
{citations_block}

Earliest mention established so far: {earliest_known}

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
      "surviving_copy": <string or null>,      // oldest EXISTING copy + its date, if stated
      "chain": <string or null>                // name the upstream source if this just repeats one
    }}
  ],
  "next_queries": [<string>, ...],   // up to {max_queries}
  "gaps": [<string>, ...]            // up to 3: what evidence is still missing
}}

Rules for "mentions":
- Only include mentions actually supported by the material; never infer a source that is not there.
- "year" is when the source was COMPOSED. If only a surviving copy's date is given, put that in
  "surviving_copy" and leave "year" null unless composition is separately stated.
- Where a SOURCE PAGE is provided, prefer what it actually says over the search summaries.
- Keep "surviving_copy" and "chain" under 15 words each.

Rules for "next_queries" — searches to run NEXT to push this origin further back:
- Target OLDER predecessors, parallel traditions, source materials and oral antecedents that
  predate the earliest mention above.
- Favour queries likely to surface primary documents, named manuscripts with their holding
  repositories, excavation reports and critical editions over general write-ups.
- Include at least one aimed at INDEPENDENT corroboration from a different information chain,
  and one aimed at scholarship DISPUTING the dating above.
- Return an empty list if the evidence looks exhausted; do not pad it.
- Never repeat any of these already-tried queries:
{prior_queries}

Return ONLY JSON.
"""


SYNTHESIZE_PROMPT = """\
You are writing the final trace report for the story "{title}".

Dated mentions gathered across all research rounds (already deduplicated):
{mentions_block}

Available citations:
{citations_block}
{pages_block}
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
      "id": "t1",                              // unique, stable, referenced by connections
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
  "connections": [ ...see CONNECTIONS below... ],
  "source_tiers": {{ "<url>": "primary|repository|academic|press|low_authority|unknown", ... }},
  "reasoning": <short paragraph explaining the chain of evidence and any uncertainty>,
  "confidence": <0..1>
}}

CONNECTIONS — the most important and most easily faked part of this report.

A timeline implies that each item leads to the next. That implication is a CLAIM, and it is
usually the weakest one on the page: "A, then B, therefore A caused B" is the error this report
exists to expose. So every connection is judged on the same terms as any other claim.

Emit up to {max_connections} connections:
{{
  "from_id": <id of the earlier item; the origin's id is "origin">,
  "to_id": <id of the later item>,
  "relation": "derives_from|retells|translates|responds_to|contradicts|contemporaneous|attests|
               no_established_link",
  "citations": [<url>, ...],
  "evidence": {{
    "mechanism": <ONE sentence: HOW does the earlier item lead to the later one? Name the route —
                  a translator, a manuscript family, a named borrowing, a documented transmission>,
    "supporting_evidence": <what actually evidences this link, or "None identified">,
    "contradictory_evidence": <scholarship or evidence against the link, or "None identified">,
    "independent_corroboration": <support from a different information chain, or "None identified">,
    "evidence_type": <same vocabulary as the dossier, applied to the LINK not the events>,
    "confidence": <0..1>,
    "confidence_label": "high|moderate|low|speculative",
    "why": <1-2 sentences on how strong this link really is>,
    "missing_piece": <what evidence would settle whether this link is real>
  }}
}}

Connection rules — read these twice:
- Two items being consecutive in time is NOT a connection. If you cannot state a concrete
  mechanism and name evidence for it, use "no_established_link" and say in "why" that the
  sequence is chronological only. That is a correct, valuable answer, not a failure.
- Scholarly consensus that A influenced B is "scholarly_inference", NOT "primary_document",
  no matter how widely the influence is repeated.
- Prefer few well-evidenced connections to many speculative ones.
- Connect the origin to what it actually led to; do not chain every item to its neighbour by default.
- Where sources disagree about whether a link exists, use "contradicts" or set
  "contradictory_evidence" and lower the confidence accordingly.

General rules:
- Every claim must be backed by at least one citation URL from the provided list.
- Prefer the OLDEST well-attested source as origin; if it's truly oral / prehistoric, set year=null
  and use an era_label.
- "year" is the COMPOSITION date. Never present the date of a surviving copy as the date of the
  source itself; that distinction belongs in "earliest_surviving_copy".
- A scholar's conclusion is "scholarly_inference", not a record. A transmitted belief with no
  documentary trail is "tradition". Only use "primary_document", "archaeological" or
  "contemporary_record" when the evidence genuinely is one.
- Where a SOURCE PAGE was read, prefer what it actually says over any summary of it.
- Sources that repeat one upstream report count as ONE. Do not describe them as corroboration.
- "confidence_label" must be consistent with the numeric "confidence" (high >= 0.75,
  moderate 0.5-0.75, low 0.3-0.5, speculative < 0.3).
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

{search_doctrine}

Search the web to surface specific, dated sub-events tightly related to that anchor:
contemporaneous retellings, immediate predecessors or successors, manuscript variants,
translations, recensions, archaeological finds, related contemporary events, named
people involved, or documented influences.

Write a concise (<= 350 words) factual summary naming specific dates, manuscript names,
shelfmarks, repositories, authors, places and cultures whenever the sources do. Do not
speculate beyond the cited sources.
"""


EXPAND_EXTRACT_PROMPT = """\
From the research material below, extract distinct dated sub-events that are tightly
related to this anchor in the history of "{story_title}":

Anchor (its id is "{parent_id}"):
- when: {when}
- source / event: {parent_source_title}
- claim: {parent_claim}

Research notes:
---
{notes}
---
{pages_block}
Available citations (use these URLs verbatim):
{citations_block}

Return JSON:
{{
  "events": [
    {{
      "id": "e1",                              // unique, referenced by connections
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
  ],
  "connections": [
    // ONE per sub-event, stating why it belongs under this anchor at all.
    {{
      "from_id": "{parent_id}",
      "to_id": <the sub-event's id>,
      "relation": "derives_from|retells|translates|responds_to|contradicts|contemporaneous|attests|
                   no_established_link",
      "citations": [<url>, ...],
      "evidence": {{
        "mechanism": <ONE sentence: how does the anchor lead to, or relate to, this sub-event?>,
        "supporting_evidence": <what evidences the link, or "None identified">,
        "contradictory_evidence": <evidence against it, or "None identified">,
        "independent_corroboration": <support from a different chain, or "None identified">,
        "evidence_type": <same vocabulary, applied to the LINK>,
        "confidence": <0..1>,
        "confidence_label": "high|moderate|low|speculative",
        "why": <1-2 sentences on how strong the link is>,
        "missing_piece": <what would settle it>
      }}
    }}
  ]
}}

Rules:
- Return AT MOST {max_events} events.
- Every event must be supported by the material and cite at least one URL.
- Do NOT repeat the anchor itself.
- Fill the evidence dossier from the material only. "None identified" is correct and expected
  when it does not establish something; never invent a source or a date to fill a field.
- A historian's conclusion is "scholarly_inference", not a record; a transmitted belief with no
  documentary trail is "tradition".
- Where a SOURCE PAGE is provided, prefer what it actually says over the search summaries.
- Sources that repeat one upstream report are ONE source, not corroboration.
- Being near the anchor in time is NOT a connection. If you cannot name a mechanism, use
  "no_established_link" and say so — a sub-event shown under a false relationship is worse
  than one shown with none.
- Order events chronologically, oldest first.
- Return ONLY JSON.
"""
