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
SOURCE TIERS — every source sits in one of these five. Record which.

TIER 1 — PRIMARY EVIDENCE. The thing itself: earliest surviving manuscripts, inscriptions,
  archaeological and excavation reports, government and court records, census returns, letters,
  diaries, photographs, recordings, and newspapers published AT THE TIME of what they report.
  Prefer the institution that actually holds or publishes the object — national archives,
  libraries, museums, manuscript repositories, universities — and name it, with a shelfmark if
  one is given. For any text you MUST record the estimated ORIGINAL COMPOSITION date separately
  from the date of the EARLIEST SURVIVING PHYSICAL COPY. They are frequently centuries apart,
  and conflating them is the single most common error in this kind of work.
TIER 2 — SCHOLARLY ANALYSIS. Peer-reviewed articles, academic books, critical editions,
  specialist scholarship. These are CONCLUSIONS and must be labeled as such — never silently
  promoted to fact, however widely repeated.
TIER 3 — SCHOLARLY DISCOVERY AND VERIFICATION. Crossref, DOIs, catalogue and index records.
  These establish that a work exists, who wrote it, and where it sits in the literature. A DOI
  does not make a claim true.
TIER 4 — INSTITUTIONAL SECONDARY. Universities, government agencies, museums, libraries,
  professional associations and research institutes writing ABOUT something they did not
  themselves record. Useful for context; follow their citations backward.
TIER 5 — GENERAL WEB. News organisations mainstream and alternative, independent researchers,
  specialist sites, blogs, video, forums, Wikipedia, Reddit, social media, and search-result
  snippets. These are LEAD GENERATORS, NOT EVIDENCE.

The two rules that decide a trace:

  NEVER let tiers 4-5 be the final evidentiary basis of a claim when the underlying primary or
  scholarly source can reasonably be located. Use them to find it: ask what evidence THIS claim
  cites, then follow the chain backward and cite what you find at the end of it.

  If reliable evidence cannot be located, say UNKNOWN or UNVERIFIED. Do not fill the gap.
  "None identified" is always a better answer than a guess, an assumed source, or a citation you
  did not actually see. Absence of evidence is a finding, and this report exists to report it.

Do not judge credibility by masthead. CNN, Fox, the BBC, a Substack and an independent
researcher are all tier 5 by default and all promoted the same way: by what they can be followed
back to. Ten websites repeating one article count as ONE source, not ten confirmations — say so
explicitly when you detect it.
"""


# The short form, for stages that summarise rather than judge. Everything a
# search step can actually act on, and nothing it cannot.
SEARCH_DOCTRINE = """\
EVIDENCE RULES for this summary:
- Prefer original documents, artefacts and the repositories holding them over write-ups about them.
- Wikis, blogs, forums, video and news write-ups are LEADS. Name the source THEY cite — author,
  work, repository, shelfmark, DOI — and report that instead of summarising them.
- Give each named source's publication date. One published at the time of the event is evidence;
  one published later is commentary on it.
- Give the COMPOSITION date and the EARLIEST SURVIVING COPY date separately whenever both appear.
- Judge each source by the evidence it cites, not by its brand.
- If several results repeat one upstream report, say so and name it: that is ONE source, not several.
- State plainly what is NOT found. Never fill a gap with a plausible guess.
"""


# The extract stage assigns every source its tier and is the only stage that ever
# saw none of the doctrine. It runs a handful of times per trace, not twenty, so
# it can afford the vocabulary it is being asked to apply.
EXTRACT_DOCTRINE = """\
TIERS: 1 primary evidence (the object/record itself, or the archive, library or museum holding
it) · 2 peer-reviewed and academic scholarship, critical editions · 3 Crossref/DOI/catalogue
metadata · 4 institutional write-ups (university, museum, agency pages ABOUT something) ·
5 general web: news, blogs, video, forums, Wikipedia, social media.

Tiers 4-5 are leads, never the basis of a claim. When a mention rests only on them, say so and
name what THEY cite so it can be chased. Never fill an unknown with a plausible guess.
"""


DECOMPOSE_PROMPT = """\
You are a research planner. The user wants a historian's map of where a story, claim, person,
object or event actually comes from — not a summary of what is generally said about it.

Subject: {title}
Optional context: {context}

{source_hierarchy}

A trace is built from STRANDS. Pick the ones this subject genuinely has; do not force the rest.
A medieval poem has manuscripts, a 1960s assassination has government records, a phrase has
attestations. The strands:

- precursor_context — the intellectual and textual world this emerged INSIDE: the literature,
  scriptures, languages, institutions and ideas already in circulation, and how they were
  transmitted. Reach back centuries, not decades — a subject that appears with no ancestry has
  been researched from its own point of view. BACKGROUND ONLY: including it never asserts that
  it influenced the subject.
- term_history — the history of the NAME or TITLE itself, which is usually older than the thing
  it now names and often meant something else first. Trace it back to its EARLIEST attested
  sense and to the people or things it applied to BEFORE the subject, sense by sense.
- reconstructed_date — dates scholars reconstruct where no record states them, and the reasoning
  that puts them there. This covers the existence and dating of MOVEMENTS, communities and
  phenomena as well as of individuals: where historians place something decades before the
  earliest text that describes it, that reconstruction is its own item, and the argument for it
  is the substance of the item.
- text_composition — when a text was WRITTEN. Distinct from when the events it describes
  supposedly happened, and distinct from any surviving copy. Cover the EARLIEST texts first,
  including ones earlier than the famous ones, and the relationships of dependence between them.
- manuscript_witness — the physically surviving copies: which one, how old, which repository.
- external_attestation — sources OUTSIDE the tradition that refer to the subject, and what
  their own texts and transmission actually establish.
- linguistic_transmission — how a name or term moved across languages and scripts, FORM BY FORM:
  each attested spelling, in its own language and script, with the date and the text it is
  attested in. The chain is the point; a single note that "the name is Greek" is not this strand.
- institutional_development — canons, offices, doctrines, standards and organisations, dated to
  when they are ATTESTED, never projected backward.
- dating_framework — the calendars and eras ACTUALLY IN USE at the time (regnal years, consular
  years, olympiads, a founding era, a local era), and separately when the era system now used to
  state the subject's dates was devised and adopted. People do not live in a calendar invented
  after them, and a report that prints their dates without saying who converted them, and when,
  is presenting an editorial act as a fact.

Produce a JSON object with:
- "entities": 3-8 key proper nouns / concepts to anchor searches.
- "strands": the applicable strand names from the list above, each as
  {{"strand": <name>, "why": <one line on what this subject specifically needs here>}}.
  Prefer 5-9 strands. A subject with only two strands is usually a subject researched lazily.
  Three are near-universal and are usually omitted by mistake rather than by judgement:
  dating_framework (any subject older than the calendar its dates are stated in),
  linguistic_transmission (any subject whose name has moved between languages), and
  precursor_context (any subject that did not appear out of nothing). Include them unless this
  subject genuinely has no such history.
- "queries": {max_queries} diverse web search queries covering the strands you chose, weighted
  toward the ones most likely to yield tier 1-2 sources. Include at least one hunting for
  INDEPENDENT corroboration from a different information chain, and at least one hunting for
  evidence that CONTRADICTS or disputes the standard account. Queries must be self-contained
  (no pronouns), in English unless another language is clearly required.
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
From the research material below, extract every distinct historical item relating to "{title}".

An item does NOT have to be a dated event. A language shifting over four centuries, a title
gradually acquiring a new meaning, a canon forming — these are items too, and dropping them
because they lack a year is how a trace ends up a thin list of dates instead of a history. Give
those a "year_end" and an era label instead of pretending they happened in one year.

{extract_doctrine}

Research notes:
---
{notes}
---
{pages_block}
Available citations (use these URLs verbatim):
{citations_block}

Earliest mention established so far: {earliest_known}
Strands still uncovered: {open_strands}

Return JSON:
{{
  "mentions": [
    {{
      "year": <signed integer or null>,        // negative = BCE; the COMPOSITION date
      "year_end": <signed integer or null>,    // set only when this SPANS a period
      "era_label": <string or null>,           // use when year unknown
      "precision": "exact|year|decade|century|millennium|era|unknown",
      "node_type": "event|reconstructed_date|text_composition|manuscript_witness|
                    external_attestation|term_history|linguistic_transmission|
                    institutional_development|dating_framework|context",
      "source_title": <string>,                // manuscript / book / event / process name
      "claim": <one sentence>,
      "citations": [<url>, ...],
      "confidence": <0..1>,
      "source_tier": "primary|repository|academic|reference_index|institutional|press|
                      low_authority|unknown",
      "published": <string or null>,           // when THIS source was published, if stated
      "discovery_only": <true when this appears only on a tier 4-5 page>,
      "cites": <string or null>,               // what that page cites for it, if it says
      "surviving_copy": <string or null>,      // oldest EXISTING copy + its date, if stated
      "chain": <string or null>                // name the upstream source if this just repeats one
    }}
  ],
  "next_queries": [<string>, ...],   // up to {max_queries}
  "gaps": [<string>, ...]            // up to 3: what evidence is still missing
}}

Rules for "mentions":
- Only include mentions actually supported by the material; never infer a source that is not there.
- "year" is when the source was COMPOSED, or when the item is attested — never when the events a
  text describes are said to have happened. If only a surviving copy's date is given, put that in
  "surviving_copy" and leave "year" null unless composition is separately stated.
- A text and its surviving copies are TWO mentions: one "text_composition", one
  "manuscript_witness". Do not merge them.
- A NAME CROSSING LANGUAGES IS ONE MENTION PER FORM, not one mention about the name. Each
  attested spelling gets its own "linguistic_transmission" item, whose "source_title" is the
  form itself with its language and script ("Iesous (Greek, Ἰησοῦς)"), whose date is when that
  form is attested, and whose claim names the text it is attested in. A single item saying a
  name "was rendered into Greek and Latin" is the summary this rule exists to prevent.
- The same applies to a TITLE ACQUIRING A NEW SENSE: one "term_history" item per attested sense,
  each with the date and the text it is attested in, not one item about the word.
- A CALENDAR IS NOT A FACT ABOUT THE PAST. Where the material says what dating system people
  actually used, or when the era used to state these dates was devised, that is a
  "dating_framework" mention of its own.
- Where a SOURCE PAGE is provided, prefer what it actually says over the search summaries.
- Keep "surviving_copy", "cites" and "chain" under 15 words each.

Rules for "next_queries" — what to search NEXT:
- Cover the UNCOVERED STRANDS listed above first. A trace is finished when its strands are
  covered, not when nothing older turns up.
- Where a mention is marked "discovery_only", write a query hunting the source it cites, naming
  that source as specifically as the material allows.
- Also push backward: older predecessors, parallel traditions, source materials and oral
  antecedents that predate the earliest mention above.
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
You are writing the final trace report for "{title}".

Items gathered across all research rounds (already deduplicated):
{mentions_block}

Available citations:
{citations_block}
{pages_block}
RESEARCH PLAN AND WHAT IT COVERED:
{strands_block}
{source_hierarchy}

WHAT THIS REPORT IS. Not a summary of what is believed about the subject, and not a list of
events. It is a map of what can be EVIDENCED about it, with each kind of claim kept separate
from the others, because they rest on different evidence and fail in different ways:

  A text was WRITTEN at some date              -> node_type "text_composition"
  A COPY of it physically survives             -> node_type "manuscript_witness"
  Someone OUTSIDE the tradition mentions it    -> node_type "external_attestation"
  Scholars RECONSTRUCT a date no record gives  -> node_type "reconstructed_date"
  A word or title has its own history          -> node_type "term_history"
  A name moved between languages               -> node_type "linguistic_transmission"
  Canons, offices and doctrines formed later   -> node_type "institutional_development"
  The dates themselves were derived/converted  -> node_type "dating_framework"
  Background it emerged among                  -> node_type "context"
  Something attested to have happened          -> node_type "event"

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
  "provenance": <who holds the object, under what shelfmark, and how it reached them, or
                 "None identified">,
  "contemporary_evidence": <evidence created at the time of the event itself, or "None identified">,
  "independent_corroboration": <support NOT descending from the same information chain, or
                                "None identified"; if many sources trace to one report, say so>,
  "contradictory_evidence": <evidence that contradicts it, or "None identified">,
  "scholarly_dispute": <live disagreement AMONG SCHOLARS about this claim — a different thing
                        from evidence against it — or "None identified">,
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
    "year_end": <signed int or null>,
    "era_label": <string or null>,
    "precision": "exact|year|decade|century|millennium|era|unknown",
    "node_type": <from the list above>,
    "attribution": "established|attributed|disputed|anonymous|not_applicable",
    "source_title": <string>,
    "summary": <2-4 sentences explaining why this is the earliest defensible origin>,
    "citations": [<url>, ...],
    "confidence": <0..1>,
    "evidence": {{ ...dossier... }}
  }},
  "timeline": [
    // chronological, oldest first
    {{
      "id": "t1",                              // unique, stable, referenced by connections
      "year": <signed int or null>,
      "year_end": <signed int or null>,        // set when this SPANS a period, not a moment
      "era_label": <string or null>,
      "precision": "...",
      "node_type": <from the list above>,
      "attribution": "established|attributed|disputed|anonymous|not_applicable",
      "source_title": <string>,
      "claim": <one sentence stating what this entry establishes>,
      "citations": [<url>, ...],
      "confidence": <0..1>,
      "evidence": {{ ...dossier... }}
    }}
  ],
  "connections": [ ...see CONNECTIONS below... ],
  "source_tiers": {{ "<url>": "primary|repository|academic|reference_index|institutional|press|
                     low_authority|unknown", ... }},
  "source_dates": {{ "<url>": "<publication year or range, or 'unknown'>", ... }},
  "reasoning": <short paragraph explaining the chain of evidence and any uncertainty>,
  "confidence": <0..1>
}}

STRUCTURAL RULES — these decide whether this report is worth anything:

- EVERY RESEARCHED STRAND APPEARS. The block above says which strands the research rounds
  actually found material for. Each of those must produce at least one entry, of the matching
  node_type, built from the items above. Items in the list carry a "type=" — that is the
  extractor's reading of what kind of thing each one is, and it is the strongest signal you
  have; do not silently retype an item to something weaker or fold several strands into one
  entry. Dropping a researched strand throws away the research and tells the reader the evidence
  was not there.
- A STRAND WITH SEVERAL ITEMS GETS SEVERAL ENTRIES. Four texts composed at four dates are four
  "text_composition" entries; the earliest surviving texts get entries even when later ones are
  more famous. A name crossing four languages is four "linguistic_transmission" entries — one
  per attested FORM, each titled with the form in its own language and script, dated to when
  that form is attested, and joined to the previous form by a "translates" connection whose
  mechanism names the text or translator that carried it across. One entry summarising a chain
  is not a chain.
- THE CALENDAR IS PART OF THE ANSWER. Where the subject predates the era system its dates are
  stated in, emit "dating_framework" entries for BOTH: what people at the time actually dated
  by, and when and by whom the era now used was devised and adopted. Nobody alive in the period
  was using the labels at the top of this report, and a reader is owed that.
- THE ORIGIN IS THE EARLIEST DEFENSIBLE ORIGIN OF THE SUBJECT, NOT THE OLDEST ROW. Entries that
  legitimately predate it are the ancestry it emerged among: type them "context", "term_history"
  or "linguistic_transmission", and join them to the origin with "provides_context". They are
  not competing origins and must not be described as the subject's beginning.
- A TEXT AND ITS SURVIVING COPIES ARE TWO ENTRIES. The "text_composition" entry carries the
  composition date. The "manuscript_witness" entry carries the physical copy's date, its
  repository and its shelfmark in "provenance". Never merge them, and never present a surviving
  copy's date as the date of the text.
- A TEXT IS NOT EVIDENCE FOR THE EVENTS IT NARRATES. It is evidence that the text existed by a
  certain date and that its community held what it says. For a "text_composition" entry the
  claim must be a proposition about the TEXT. If you want to claim a narrated event happened,
  that is a separate entry needing its own evidence — and if the only support is the narrative
  itself, its evidence_type is "later_historical_account" or "tradition", never
  "primary_document" or "contemporary_record".
- WHAT DEVELOPED LATER IS DATED LATER. Canons, creeds, titles, offices, standards and
  institutions are "institutional_development", dated to when they are ATTESTED. Never project
  them backward onto the earliest evidence.
- SAY WHAT IS MISSING. Where a subject has no contemporary documentation, that absence is one of
  the most important findings in the report: name what does not exist, in "missing_piece" and in
  "contemporary_evidence".
- DATES ARE A CLAIM TOO. Where dates were reconstructed, use "reconstructed_date", and put the
  ARGUMENT in the dossier's "why": what a reconstruction is worth is entirely the reasoning
  behind it. Where a subject, or the movement around it, is placed decades or centuries before
  the earliest surviving text describing it, say in that entry why historians place it there
  anyway and what would move it. Where the dates were converted into a calendar nobody alive at
  the time used, give that its own "dating_framework" entry rather than presenting the converted
  date as a plain fact.
- BACKGROUND IS NOT INFLUENCE. Older parallel traditions go in as "context" entries joined with
  "provides_context". Including one never asserts that it shaped the subject.

CONNECTIONS — the most important and most easily faked part of this report.

A timeline implies that each item leads to the next. That implication is a CLAIM, and it is
usually the weakest one on the page: "A, then B, therefore A caused B" is the error this report
exists to expose. So every connection is judged on the same terms as any other claim.

Emit up to {max_connections} connections:
{{
  "from_id": <id of the earlier item; the origin's id is "origin">,
  "to_id": <id of the later item>,
  "relation": "derives_from|retells|translates|responds_to|contradicts|contemporaneous|attests|
               provides_context|no_established_link",
  "citations": [<url>, ...],
  "evidence": {{
    "mechanism": <ONE sentence: HOW does the earlier item lead to the later one? Name the route —
                  a translator, a manuscript family, a named borrowing, a documented transmission>,
    "supporting_evidence": <what actually evidences this link, or "None identified">,
    "contradictory_evidence": <evidence against the link, or "None identified">,
    "independent_corroboration": <support from a different information chain, or "None identified">,
    "scholarly_dispute": <disagreement among scholars about the link, or "None identified">,
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
- Background that merely predates the subject is "provides_context", never "derives_from".
  Shared themes are not descent, and similarity is not transmission.
- Scholarly consensus that A influenced B is "scholarly_inference", NOT "primary_document",
  no matter how widely the influence is repeated.
- Prefer few well-evidenced connections to many speculative ones.
- Connect the origin to what it actually led to; do not chain every item to its neighbour by default.
- Where sources disagree about whether a link exists, use "contradicts" or set
  "contradictory_evidence" and lower the confidence accordingly.

General rules:
- LENGTH IS NOT A VIRTUE, BUT NEITHER IS BREVITY. There is no entry limit. Every distinct item
  above that carries its own date, its own evidence and its own way of being wrong is its own
  entry. Merging two items because they are related loses the thing this report is for: they
  rest on different evidence and one can fail without the other.
- Every claim must be backed by at least one citation URL from the provided list.
- Where the only citations available for a claim are tier 4-5, say so in "why" and set the
  evidence_type to what those sources can actually support. Do not describe a claim as primary
  or contemporary evidence because a website stated it confidently.
- Prefer the OLDEST well-attested source as origin; if it's truly oral / prehistoric, set year=null
  and use an era_label.
- "year" is the COMPOSITION or attestation date, never the date of events a text describes.
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

{extract_doctrine}

Return JSON:
{{
  "events": [
    {{
      "id": "e1",                              // unique, referenced by connections
      "year": <signed integer or null>,        // the COMPOSITION date
      "era_label": <string or null>,
      "precision": "exact|year|decade|century|millennium|era|unknown",
      "node_type": "event|reconstructed_date|text_composition|manuscript_witness|
                    external_attestation|term_history|linguistic_transmission|
                    institutional_development|dating_framework|context",
      "attribution": "established|attributed|disputed|anonymous|not_applicable",
      "source_title": <string>,
      "claim": <one sentence explaining how this sub-event relates to the anchor>,
      "citations": [<url>, ...],
      "confidence": <0..1>,
      "evidence": {{
        "claim": <the claim restated as ONE testable proposition>,
        "earliest_supporting_source": <named source + what it is, or "None identified">,
        "estimated_source_date": <when it was COMPOSED, or "Unknown">,
        "earliest_surviving_copy": <oldest existing copy + date + repository, or "None identified">,
        "provenance": <holding institution + shelfmark, or "None identified">,
        "contemporary_evidence": <evidence from the time of the event, or "None identified">,
        "independent_corroboration": <support from a different information chain, or "None identified">,
        "contradictory_evidence": <disputing sources, or "None identified">,
        "scholarly_dispute": <disagreement among scholars, or "None identified">,
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
                   provides_context|no_established_link",
      "citations": [<url>, ...],
      "evidence": {{
        "mechanism": <ONE sentence: how does the anchor lead to, or relate to, this sub-event?>,
        "supporting_evidence": <what evidences the link, or "None identified">,
        "contradictory_evidence": <evidence against it, or "None identified">,
        "independent_corroboration": <support from a different chain, or "None identified">,
        "scholarly_dispute": <disagreement among scholars about the link, or "None identified">,
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
  than one shown with none. Background that merely predates the anchor is "provides_context".
- A text and its surviving copies are TWO events: "text_composition" carries the composition
  date, "manuscript_witness" carries the surviving copy, its date and its repository. A text is
  evidence that the text existed, not evidence for the events it narrates.
- Order events chronologically, oldest first.
- Return ONLY JSON.
"""


# The chase. A claim resting on a wiki or a news write-up is a claim whose real
# evidence, if it has any, is one hop away — named in that page's own references.
# This asks for that hop specifically, because a general re-search returns the
# same summaries the first search already found.
CHASE_SEARCH_PROMPT = """\
A claim in a historical trace of "{title}" currently rests only on a general-web or
institutional source, which is a lead rather than evidence. Find what it rests ON.

Claim: {claim}
Currently cited: {weak_source}
That source appears to cite: {cites}
Harvested references from that page: {references}

{search_doctrine}

Search for the UNDERLYING source: the manuscript, inscription, excavation report, archival
record, critical edition or peer-reviewed study that this claim ultimately depends on. Name it
as specifically as the evidence allows — author, work, book and section, repository, shelfmark,
DOI, catalogue number.

Do NOT summarise the general-web source. If the claim turns out to rest on nothing more than
that source repeating itself, say exactly that: it is the most useful finding you can return.

Write <= 250 words.
"""
