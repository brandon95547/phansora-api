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
EXPAND_DOCTRINE = """\
TIERS: 1 primary evidence (the object/record itself, or the archive, library or museum holding
it) · 2 peer-reviewed and academic scholarship, critical editions · 3 Crossref/DOI/catalogue
metadata · 4 institutional write-ups (university, museum, agency pages ABOUT something) ·
5 general web: news, blogs, video, forums, Wikipedia, social media.

Tiers 4-5 are leads, never the basis of a claim. When a mention rests only on them, say so and
name what THEY cite so it can be chased. Never fill an unknown with a plausible guess.
"""
# ---------------------------------------------------------------------------
# The research pass. One grounded call; the model runs its own searches.
#
# Two things in here were measured rather than reasoned about, and both matter
# more than they look.
#
# The quoted QUESTION. Asked as an instruction ("trace backward from the earliest
# evidence"), the model reached the material a subject descends from in 2 runs of
# 15. Asked as a question it can search verbatim, 13 of 13. In a grounded call the
# model's whole job is turning intent into queries, and a question is already
# query-shaped where a described goal is not.
#
# The sentence about copies. Without it the model traces a text's OWN transmission
# — this manuscript from that archetype from a lost autograph — which looks like
# going backward and is not. "Copy" and "translate" both read as scribal descent,
# so the different-older-work reading has to be said explicitly.
#
# The enumerated list drives search COUNT: queries track the number of separately
# findable things asked for, so six named categories buy roughly six searches where
# one sweeping request buys two. "At least eight" moves it further; "multiple" is
# unfalsifiable and gets satisfied with three.
#
# `{context_clause}` carries the dashboard's context box, so "Mercury" the planet
# stays distinct from "Mercury" the god.
#
# To iterate on this, do not edit here and deploy. Use the lab:
#   python -m phansora.products.chrono_origin.prompt_lab "Jesus Christ" -n 5 --shipped
# ---------------------------------------------------------------------------

RESEARCH_PROMPT = """\
Using live web search, trace the complete historical, material, and conceptual lineage of "{title}"{context_clause}, starting from the absolute earliest known foundational bedrock (such as pre-existing texts, raw materials, or primitive prototypes, even if dating deep into the BCE or prehistoric era) Do not answer from memory. Every item must come from a search you actually ran. Before finalizing the trace, walk backward through each item found and ask: “What earlier identifiable evidence did this derive from or build upon?” If an earlier supported source exists, research and include it, then repeat until no earlier supported connection can be found. If an item comes from a larger source work, corpus, collection, or predecessor, include that source as its own item rather than replacing it with one example, translation, or surviving copy.

Then present everything forward in date order, oldest to new. For each item:

Date
Evidence — what it is
Source — where our knowledge of it comes from
Original vs surviving — composition date and oldest surviving witness, when different
Connection — what evidence links this item to the next
Establishes — what it actually shows
Uncertain — what is estimated, disputed, or unknown
Consulted — the pages you actually opened

Keep interpretation out of the chain. Ideas, traditions, influences and developments are not
items — put them in a NOTES section after the chain and say which items they rest on.

Where the evidence stops, say so. Never fill a gap.
"""
SYNTHESIZE_PROMPT = """\
Below is research already gathered about "{title}". Turn it into the JSON structure at the
end of this message.

Research material:
{mentions_block}

Available citations:
{citations_block}
{pages_block}
YOUR JOB IS TO FORMAT, NOT TO JUDGE.

- ONE ENTRY PER ITEM the research reports. If it lists fifteen items, return fifteen.
- Do NOT merge two items into one entry, even where they are usually named together.
- Do NOT drop an item because it seems minor, uncertain, duplicated or hard to place.
- Do NOT add an item the research did not report.
- Keep its dates, its wording, its sources. Where it gives a composition date and a
  separate surviving-copy date, keep both: the composition date goes in "year", the
  copy goes in the dossier.
- The oldest item becomes "origin". Everything after it goes in "timeline", oldest first.
- Anything the research put under NOTES — ideas, traditions, influences, developments —
  becomes a "conclusions" entry rather than a timeline entry.
- Where the research says something is unknown, disputed or absent, carry that through
  rather than resolving it.

"node_type" is a LABEL for display, not a test an item has to pass. Pick the closest:

  "text"                — a work known through copies: a history, a treatise, a report
  "manuscript"          — a specific physical copy: codex, papyrus, parchment leaf
  "scroll"              — a rolled manuscript
  "letter"              — correspondence
  "inscription"         — cut or written on a durable surface
  "document"            — an issued instrument: decree, charter, patent, filing
  "record"              — a register kept as a series: census, court, tax, parish
  "artifact"            — an object carrying evidence: coin, seal, ostracon, tablet
  "archaeological_find" — an excavated site, structure or assemblage
  "event"               — anything the labels above do not fit

Use "event" rather than dropping an item. Nothing is excluded for being the wrong shape.

Every entry carries an "evidence" dossier — what actually backs it, as the research
reported it. "None identified" is a legitimate answer and always better than inventing one.

The evidence dossier shape (same for the origin and every entry):
{{
  "claim": <what this establishes, as ONE testable proposition>,
  "earliest_supporting_source": <named source + what kind of thing it is, or "None identified">,
  "estimated_source_date": <when it was COMPOSED or made; a range is fine, or "Unknown">,
  "earliest_surviving_copy": <oldest physically existing copy + its date + repository if known,
                              or "None identified">,
  "provenance": <who holds it, under what shelfmark, and how it reached them, or
                 "None identified">,
  "contemporary_evidence": <evidence created at the time, or "None identified">,
  "independent_corroboration": <support NOT descending from the same chain, or "None identified">,
  "contradictory_evidence": <evidence that contradicts it, or "None identified">,
  "scholarly_dispute": <live disagreement among scholars, or "None identified">,
  "evidence_type": "primary_document|archaeological|contemporary_record|near_contemporary_account|
                    later_historical_account|scholarly_inference|tradition|disputed|absent",
  "confidence_label": "high|moderate|low|speculative",
  "why": <1-2 sentences of plain language>,
  "missing_piece": <the single absent piece of evidence that most limits this>
}}

Produce a JSON object:
{{
  "origin": {{
    "year": <signed int or null>,
    "year_end": <signed int or null>,
    "era_label": <string or null>,
    "precision": "exact|year|decade|century|millennium|era|unknown",
    "node_type": <one of the labels above>,
    "attribution": "established|attributed|disputed|anonymous|not_applicable",
    "source_title": <named as a reader would look for it>,
    "summary": <2-4 sentences: what this is, as the research described it>,
    "citations": [<url>, ...],
    "confidence": <0..1>,
    "evidence": {{ ...dossier... }}
  }},
  "timeline": [
    // oldest first; one entry per remaining item in the research
    {{
      "id": "t1",
      "year": <signed int or null>,
      "year_end": <signed int or null>,        // set when production SPANS a period
      "era_label": <string or null>,
      "precision": "...",
      "node_type": <one of the labels above>,
      "attribution": "established|attributed|disputed|anonymous|not_applicable",
      "source_title": <named as a reader would look for it>,
      "claim": <one sentence stating what this establishes>,
      "citations": [<url>, ...],
      "confidence": <0..1>,
      "evidence": {{ ...dossier... }}
    }}
  ],
  "conclusions": [
    // whatever the research put under NOTES
    {{
      "statement": <one plain proposition>,
      "rests_on": [<ids this rests on: "origin", "t1", ...>],
      "confidence_label": "high|moderate|low|speculative",
      "reasoning": <how the named evidence gets you to the statement>,
      "dissent": <serious disagreement with this reading, or "None identified">
    }}
  ],
  "connections": [ ...see below... ],
  "reasoning": <short paragraph on the shape of the chain and where it is thin>,
  "confidence": <0..1>
}}

CONNECTIONS. The research reports, per item, what links it to the next. Carry those across —
do not invent links it did not state. Up to {max_connections}:
{{
  "from_id": <id of the earlier item; the origin's id is "origin">,
  "to_id": <id of the later item>,
  "relation": "derives_from|retells|translates|responds_to|contradicts|contemporaneous|attests|
               provides_context|no_established_link",
  "citations": [<url>, ...],
  "evidence": {{
    "mechanism": <ONE sentence: HOW does the earlier item lead to the later one, as the
                  research stated it — a translator, a manuscript family, a quotation>,
    "supporting_evidence": <what evidences this link, or "None identified">,
    "contradictory_evidence": <evidence against the link, or "None identified">,
    "independent_corroboration": <support from a different chain, or "None identified">,
    "scholarly_dispute": <disagreement about the link, or "None identified">,
    "evidence_type": <same vocabulary as the dossier, applied to the LINK>,
    "confidence": <0..1>,
    "confidence_label": "high|moderate|low|speculative",
    "why": <1-2 sentences on how strong this link really is>,
    "missing_piece": <what evidence would settle whether this link is real>
  }}
}}

Where the research did not state a link between two consecutive items, use
"no_established_link" and say in "why" that the order is chronological only. That is a
correct answer, not a failure.

OUTPUT:
- Every entry needs at least one citation URL from the list above where the research gave one.
- Dates: "year" is a signed integer, negative for BC/BCE. Use "year_end" for a span.
- Set "confidence" from the numeric scale (high >= 0.75, moderate 0.5-0.75, low 0.3-0.5,
  speculative < 0.3).
- Return ONLY JSON.
"""




# What each expansion mode goes looking for.
#
# Two directives per mode: one aims the WEB SEARCH, one aims the EXTRACTION. They are
# separate because the stages can act on different things — a search can only be
# pointed at a topic, while extraction can be told what to reject.
#
# Every one of these has to read sensibly for a manuscript, a telescope observation, a
# treaty, an excavation and a patent alike. Chrono Origin does not know which kind of
# subject it is looking at, so a directive that assumes documents would quietly fail on
# half the product's range.
EXPAND_MODES = {
    "discovery": {
        "query": "oldest surviving manuscript discovery excavation found",
        "label": "Path to Discovery",
        "search": (
            "the surviving RECORDS of how this came to be known: the excavation report, the "
            "first publication announcing it, field notes, site photographs, the accession or "
            "purchase record, the paper that deciphered or identified it — with dates, authors "
            "and where each is held"
        ),
        "extract": (
            "Return the surviving records that document how this became known — the excavation "
            "report, the first published announcement, the field notes, the photographs, the "
            "accession record. The DISCOVERY ITSELF is an event and cannot be a step; the report "
            "that records it is a document and can. Date each to when the record was made."
        ),
    },
    "preservation": {
        "query": "manuscript transmission copies custody conservation",
        "label": "Path to Preservation",
        "search": (
            "the surviving objects and records that carried this forward: later copies, "
            "translations, editions, conservation and restoration reports, inventories, catalogues "
            "and archive records naming it, with their dates and repositories"
        ),
        "extract": (
            "Return the surviving things that did the carrying — the copies, the translations, "
            "the editions, the conservation report, the inventory or catalogue entry that proves "
            "it was held somewhere at a date. Each is an object with its own date, not a statement "
            "that the subject survived."
        ),
    },
    "verification": {
        "query": "radiocarbon dating authentication palaeography analysis",
        "label": "Path to Verification",
        "search": (
            "the surviving reports of tests and analysis: radiocarbon and scientific dating "
            "papers, palaeographic studies, provenance and authentication reports, forensic "
            "examinations, and published challenges to its authenticity"
        ),
        "extract": (
            "Return the published reports and studies that establish or dispute what is known — "
            "the dating paper, the analysis, the authentication report, the article contesting it. "
            "The test is an event; the paper reporting it is a text, and that is the step. A report "
            "concluding something is FALSE belongs here as much as one confirming it."
        ),
    },
    "related": {
        "query": "associated finds same site archive catalogue",
        "label": "Related Evidence",
        "search": (
            "other surviving evidence DIRECTLY connected to this: companion finds from the same "
            "site, hoard or archive, works by the same hand, contemporaneous records naming it, "
            "objects catalogued alongside it"
        ),
        "extract": (
            "Return other surviving evidence with a direct, statable relationship to the anchor — "
            "found with it, produced by the same hand, naming it, held with it. Name the "
            "relationship in the claim. Proximity in time alone is not a relationship."
        ),
    },
    "earlier": {
        "query": "earlier sources older manuscript predecessor origins",
        "label": "Earlier Evidence",
        "search": (
            "surviving evidence from BEFORE this that bears on its origin: what it was copied "
            "from, drew on, translated or replaced, and older objects and texts in the same "
            "tradition"
        ),
        "extract": (
            "Return surviving evidence PREDATING the anchor that fills a gap in how it came about. "
            "It must be genuinely earlier and genuinely new — the point is the missing stretch of "
            "timeline, not the step that already sits before this one."
        ),
    },
    "later": {
        "query": "later citations quoted influence reception",
        "label": "Later Influence",
        "search": (
            "surviving later evidence showing this had an effect: texts that quote, copy, answer "
            "or build on it, and works that cite it by name"
        ),
        "extract": (
            "Return surviving later evidence showing this had an effect — texts that quote, reuse, "
            "answer or build on it, with the route named. It must be genuinely new: the step that "
            "already follows this one on the timeline is not an influence."
        ),
    },
}


def expand_mode(mode: str) -> dict:
    """The directives for a mode, defaulting to the least presumptuous one."""
    return EXPAND_MODES.get(mode) or EXPAND_MODES["related"]


def format_existing_block(existing) -> str:
    """What the board already shows, so an expansion can avoid handing it back.

    The most common way an expansion wastes a call is by returning the anchor's own
    neighbour — ask for material around Paul's letters and the gospels come back, which
    are already the next step along. Naming them is cheaper than any instruction about
    novelty, because the model cannot avoid a duplicate it has not been shown.
    """
    items = [str(t).strip() for t in (existing or []) if str(t or "").strip()]
    if not items:
        return "(nothing else on the timeline yet)"
    return "\n".join(f"- {t}" for t in items[:40])


EXPAND_SEARCH_PROMPT = """\
You are a research assistant performing focused web searches about one specific item in
the broader history of {story_title}{context_clause}.

Search query: {parent_source_title} {mode_query}

Anchor item being expanded:
- when: {when}
- source / event: "{parent_source_title}"
- claim: {parent_claim}

{search_doctrine}

WHAT TO LOOK FOR — this expansion is aimed at one axis, not at "anything nearby":

{mode_search}

ALREADY ON THE TIMELINE. These are shown to the user already, so finding them again
adds nothing. Search past them:
{existing_block}

Write a concise (<= 350 words) factual summary naming specific dates, manuscript names,
shelfmarks, repositories, authors, places and cultures whenever the sources do. Do not
speculate beyond the cited sources.
"""


EXPAND_EXTRACT_PROMPT = """\
From the research material below, extract distinct dated sub-events for ONE named axis
of this anchor in the history of "{story_title}".

WHAT THIS EXPANSION IS FOR — {mode_label}:
{mode_extract}

A BRANCH MAY BE AN EVENT. The main chain admits only surviving objects, because a
chain asserts what exists. A branch does a different job — it explains the node it hangs
from — so a dated, cited happening is a legitimate answer here: a discovery, an
excavation, a publication, a decipherment, a test. Prefer the object when there is one
(name the excavation report rather than the dig), but do not return nothing merely
because the honest answer is something that happened.

EVERY SUB-EVENT CARRIES A DATE. "year", or "year"+"year_end" for something produced over
a period. A branch takes a position on the timeline exactly as a step does, so an undated
one has nowhere to sit and is DISCARDED before it reaches the board — an estimate with a
range beats a correct answer with no date on it. Date the RECORD, not the thing it
records: an excavation report is dated to its publication.

An expansion exists to GROW the timeline. These are already on it, and returning any of
them costs the user a call and shows them a card they have already read:
{existing_block}

Do not return anything already listed above. If this axis genuinely has nothing beyond
what is listed, returning nothing is correct — but look properly first: most axes have
surviving records that simply have not been asked for yet, and an empty answer given
too readily is the same failure as a padded one.

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

{expand_doctrine}

Return JSON:
{{
  "events": [
    {{
      "id": "e1",                              // unique, referenced by connections
      "year": <signed integer or null>,        // the COMPOSITION date
      "era_label": <string or null>,
      "precision": "exact|year|decade|century|millennium|era|unknown",
      "node_type": "text|manuscript|scroll|letter|inscription|document|record|artifact|
                    archaeological_find|event",
      // The nine object kinds, OR "event" — and event is allowed HERE and not in the
      // main chain. A branch explains an anchor; the chain asserts what survives. "A
      // farmer turned it up while ploughing" can be the true answer to how something was
      // discovered, and refusing it because a farmer is not an artefact makes the
      // question unanswerable. Use an object kind whenever the thing
      // IS one — the excavation report, the first edition — and "event" when the honest
      // answer is something that happened.
      "is_evidence": <true if this is a surviving object; false for an event or an account>,
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
- A text and its surviving copies are TWO events: the work as "text" carries the composition
  date, the physical copy as "manuscript" or "scroll" carries its own date and its repository. A
  text is evidence that the text existed, not evidence for the events it narrates.
- SUB-EVENTS ARE SURVIVING EVIDENCE TOO. Expanding a step means naming the next surviving objects
  underneath it, not the story around it. If a sub-event is not something someone could go and
  examine, it does not belong here.
- Order events chronologically, oldest first.
- Return ONLY JSON.
"""
