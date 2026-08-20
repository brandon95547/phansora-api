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

A trace is a CHAIN OF SURVIVING EVIDENCE. Every step in the finished trace is a thing that
still exists and can be examined — a text, a manuscript, a scroll, a letter, an inscription, a
document, a record, an object, an excavated find — placed in the order it was produced. Your job
here is to plan the hunt for those objects. Not for the story, not for what happened, not for
what the period was like: for what SURVIVES.

The test that decides everything downstream: CAN SOMEONE GO AND LOOK AT IT? Plan searches that
end at a repository, a shelfmark, an excavation report or a critical edition.

The strands are kinds of surviving evidence. Pick the ones this subject genuinely has; do not
force the rest. A medieval poem has manuscripts, a 1960s assassination has government records, a
disputed site has excavation reports.

- precursor_evidence — what survives from BEFORE the subject that the subject's own evidence
  DESCENDS FROM: the corpus its texts quote, the tradition they are written inside, the language
  they translate. This is where the chain starts, and it is the strand most often researched and
  then thrown away. Reach back centuries, not decades — for a first-century subject that means
  asking what scriptures, translations and copies already existed and are still readable now.
  It is evidence of what existed, never of a mood, a climate or an expectation.
- earliest_texts — the earliest surviving texts about or by the subject, with estimated
  COMPOSITION dates and the basis for each. Earliest first, including ones earlier than the
  famous ones, and the dependence between them.
- manuscripts — the physically surviving copies: which one, how old, which repository, what
  shelfmark, and how far the copy postdates the composition.
- external_sources — surviving texts from OUTSIDE the tradition that mention the subject, their
  own composition dates, and how those texts themselves survive.
- documents_records — surviving documents and administrative records: decrees, censuses, court,
  tax, parish or official registers, and the archive that holds them.
- inscriptions_artifacts — surviving inscriptions, coins, seals, ostraca and inscribed objects,
  with find context, date and holding institution.
- archaeology — excavated sites, structures and assemblages, as published in excavation reports.

NOT STRANDS, because they are not surviving objects: expectations, movements, moods, climates of
belief, developments, influences, reconstructed events, periods of oral transmission. Those are
conclusions to be drawn AFTER the evidence is laid out, and the final report has a separate place
for them. Do not spend searches establishing them.

Produce a JSON object with:
- "entities": 3-8 key proper nouns / concepts to anchor searches.
- "strands": the applicable strand names from the list above, each as
  {{"strand": <name>, "why": <one line on what this subject specifically needs here>}}.
  Prefer 3-5 strands, and pick the ones where this subject's evidence ACTUALLY IS. Every
  strand you name reserves a search in every later round, so a strand chosen out of
  completeness rather than judgement costs the whole trace and returns nothing.
  precursor_evidence and manuscripts are near-universal — almost nothing appears with no
  surviving ancestry, and a text known only through copies has a copy history rarely the
  same age as the text. The other five are subject-specific: name documents_records,
  inscriptions_artifacts or archaeology only when this subject plainly has them.
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

Return JSON. AT MOST 12 MENTIONS — the best ones, not everything you can find. This is one
round of several, and an unbounded list here overruns the output budget, gets cut off mid-JSON
and is regenerated from scratch, which costs more time than the searches that produced it.

{{
  "mentions": [
    {{
      "year": <signed integer or null>,        // negative = BCE; the COMPOSITION date
      "year_end": <signed integer or null>,    // set only when this SPANS a period
      "era_label": <string or null>,           // use when year unknown
      "precision": "exact|year|decade|century|millennium|era|unknown",
      "node_type": "text|manuscript|scroll|letter|inscription|document|record|artifact|
                    archaeological_find",   // what kind of SURVIVING thing this is
      "is_evidence": <true only if this is a surviving object someone could go and examine;
                      false for readings, developments, movements, expectations, inferred events>,
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
- EXTRACT SURVIVING THINGS. A mention is worth recording when it names something that still
  exists and can be examined: a text, a copy, an object, a find, a record. Set "is_evidence" true
  for those. Where the material asserts something that is NOT a surviving object — a movement, an
  expectation, a development, an event inferred rather than recorded — you may still record it,
  but set "is_evidence" false. The report keeps those separate from the chain, so mislabelling
  one as evidence is the most damaging error you can make here.
- A text and its surviving copies are TWO mentions: the work as "text", the physical copy as
  "manuscript" or "scroll" with its repository. Do not merge them, and never give the work the
  copy's date.
- NAME THE OBJECT SO IT CAN BE FOUND. Where the material gives a shelfmark, repository,
  excavation report or critical edition, keep it — that is what makes the mention usable.
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
You are assembling the chain of evidence for "{title}".

Items gathered across all research rounds (already deduplicated):
{mentions_block}

Available citations:
{citations_block}
{pages_block}
RESEARCH PLAN AND WHAT IT COVERED:
{strands_block}
{source_hierarchy}

THE ONE RULE. A trace is a chain of SURVIVING EVIDENCE, in the order it was produced. Every
step is a thing that still exists and can be examined: a text, a manuscript, a scroll, a letter,
an inscription, a document, a record, an object, an excavated find. At each step you are
answering exactly one question:

    WHAT IS THE NEXT SURVIVING PIECE OF EVIDENCE?

Nothing else is a step. Not a movement, not a mood, not an expectation, not a development, not a
period, not a reconstruction, not an event you infer from a text rather than read in one. Those
are readings OF evidence, and a reading placed in the chain inherits the authority of an
artefact it has not earned. A reader who meets "messianic expectation" between two manuscripts
has been shown a conclusion dressed as a find. Put it in "conclusions" instead, where it belongs
and where it is still fully visible.

The test for every step, applied without mercy: CAN SOMEONE GO AND LOOK AT IT? Name the holding
institution, the shelfmark, the excavation report or the critical edition that publishes it. If
you cannot name where the thing is or what publishes it, it is not a step in the chain.

WHAT COUNTS AS A STEP — pick the "node_type" that says what kind of surviving thing it is:

  "text"                — a work surviving through copies: a gospel, a history, a treatise
  "manuscript"          — a specific physical copy: codex, papyrus, parchment leaf
  "scroll"              — a rolled manuscript, where the find is dated and reported as such
  "letter"              — correspondence, surviving as a text or as an object
  "inscription"         — cut, carved, painted or stamped onto a durable surface
  "document"            — a charter, deed, contract, decree or administrative instrument
  "record"              — a register kept as a series: census, court, tax, parish
  "artifact"            — an object carrying evidence: coin, seal, ostracon, tablet
  "archaeological_find" — an excavated site, structure or assemblage, as reported

A worked example of the shape wanted, for a trace of Jesus. The chain is exactly this, and
nothing between the links:

  Hebrew scriptures, composed over centuries, reaching their form by roughly 400 BC   [text]
      the corpus the later steps quote and are written inside
  the Septuagint, the Greek translation, 250-100 BC                                   [text]
      the Greek text the New Testament writers actually quote from
  the Dead Sea Scrolls, 250 BC - 70 AD                                                [scroll]
      surviving physical witnesses to that corpus, centuries older than any other copy
  the authentic letters of Paul, 50-65 AD                                             [letter]
      the earliest surviving writing about the subject — a step despite being written
      from inside the movement, which is a fact for its dossier, not a disqualification
  the four Gospels, 65-100 AD                                                         [text]
  Josephus, Antiquities, 93-94 AD                                                     [text]
      a writer outside the movement
  Tacitus, Annals, around 116 AD                                                      [text]

Note where that chain BEGINS. Not at the subject, and not at the earliest text about him — it
begins roughly four centuries earlier, with the corpus his sources quote. The first three steps
earn their place by documentary descent, not by being about him: each is something the later
steps translate, quote or physically witness. A chain for this subject that opens in the first
century has dropped its first four hundred years, and that is the single most common way this
goes wrong.

Note what is ABSENT from that chain and must stay absent from yours: no "messianic expectation",
no "rise of the Jesus movement", no "oral tradition period", no birth, no ministry, no
crucifixion as steps. Some of those may well be true; none of them is a surviving object. What
the chain shows is which documents exist and when. What they mean comes afterwards.

DATING A STEP. "year" and "year_end" date the EVIDENCE ITSELF — when the text was composed, when
the object was made, when the record was kept. Never the date of the events it describes. Where
composition and the earliest surviving physical copy differ, and they are frequently centuries
apart, the composition dates the step and the surviving copy is named in the dossier's
"earliest_surviving_copy" and "provenance". A step whose composition date is genuinely a
scholarly reconstruction stays a step — it is still a surviving object — but the dossier's "why"
must carry the argument for the date, and the confidence must reflect that it is reconstructed.

ORDER. Strictly chronological by the date of the evidence, earliest first.

WHERE THE CHAIN STARTS — and this is the rule most often got wrong.

The chain does NOT start at the earliest evidence ABOUT the subject. It starts at the earliest
surviving evidence the subject's own evidence DESCENDS FROM: the corpus its texts quote, the
tradition they are composed inside, the text they translate. A trace of a first-century figure
that opens in the first century has skipped the entire written tradition its sources are writing
within, and has silently answered a much smaller question than the one asked.

The test for reaching back one more step is DOCUMENTARY, not thematic: does a later item in the
chain quote, translate, continue, or sit inside this earlier one? If yes, it is a step. If the
only link is shared theme, shared culture, or a mood of the period, STOP — that is background,
and background is not a step and not a conclusion either; it is simply out of scope.

That test is also what stops the chain regressing forever. It reaches back while documents point
at documents, and halts where they stop pointing.

A SURVIVING COPY IMPLIES THE WORK IT COPIES, AND BOTH ARE STEPS. If the chain contains a
manuscript, a scroll or a fragment, then the work it carries is itself a step, dated by its
COMPOSITION and placed earlier. A chain that lists the Dead Sea Scrolls but not the scriptures
they preserve has recorded the witness and dropped what it witnesses — which is the older half,
and the half the later steps quote. Emit both: the text at its composition date, the copy at the
date of the object.

A STEP IS EITHER ABOUT THE SUBJECT OR DESCENDED FROM. Those are the only two ways in. Evidence
that merely dates the period, fixes a background fact, or establishes the setting is neither,
however genuine the object — a census inscription that helps date a nativity is doing
chronology, not descent, and belongs in "conclusions" with the dating argument it supports.

EVERY STEP CARRIES A DATE. "year", or "year"+"year_end" for something produced over a period.
This is the one field a step cannot do without: a chain is an order, and an undated object has
no position in it. Where the date is a scholarly estimate or a wide range, give the range and say
so in the dossier — a range is a date, "unknown" is not. An object you genuinely cannot date is
not a step; put what it shows in "conclusions" instead.

Corpora get ranges: scriptures composed across centuries take the span of their composition, not
the date of one book inside them, and not nothing.

ONE WORK PER STEP WHEN THE DATES DIFFER. "New Testament writings (Gospels, Epistles), 50-100" is
two steps wearing one label — the letters of Paul at roughly 50-65 and the gospels at roughly
65-100 have different authors, different dates and different evidence, and either can be wrong
without the other. Group only what was genuinely produced and dated as a unit.

NOTHING ENTERS THE CHAIN THAT NOTHING DESCENDS FROM. A later church, a site, an object from the
right region and the wrong century may be perfectly real evidence of something else. The test is
still the one above: does a later step quote, translate, continue or physically witness it? If
you find yourself writing that a step does not bear on the subject, you have written a
conclusion, and it belongs in "conclusions".

WHO WROTE IT DOES NOT DECIDE WHETHER IT IS A STEP.

A text written from inside the tradition, by a believer, a partisan, a follower or an
interested party, is still a surviving object and still a step. Its provenance changes what it
can be used to CONCLUDE, not whether it exists. Independence belongs in the dossier's
"independent_corroboration" — it is recorded there, never used to keep a document out of the
chain.

Getting this wrong produces a specific and confident-looking failure: the chain silently becomes
"earliest attestation from OUTSIDE the movement", skipping the earlier writings by people within
it, and opening decades late on a source that happens to be disinterested. That is a legitimate
question, and it is not this one. The earliest surviving text is the earliest surviving text
whoever wrote it.

The same applies to a work containing a disputed or interpolated passage. Suspected editing is a
matter for "contradictory_evidence" and "scholarly_dispute" and a lower confidence — never a
reason to promote a compromised source above an earlier sound one, and never a reason to drop it.

A TEXT IS NOT EVIDENCE FOR WHAT IT NARRATES. It is evidence that the text existed by a date and
that someone wrote what it says. Any claim about a narrated event is a conclusion, not a step,
and if the only support for it is the narrative itself, say so plainly in the conclusion.

Every step carries an "evidence" dossier — a plain, arguable statement of what actually backs it.
"None identified" is a legitimate and valuable answer, always better than a plausible invention.

The evidence dossier shape (same for the origin and every step):
{{
  "claim": <what this evidence establishes, as ONE testable proposition about the OBJECT,
            e.g. "A Greek translation of the Hebrew scriptures existed by the 2nd century BC">,
  "earliest_supporting_source": <named source + what kind of thing it is, or "None identified">,
  "estimated_source_date": <when it was COMPOSED or made; a range is fine, or "Unknown">,
  "earliest_surviving_copy": <oldest physically existing copy + its date + repository if known,
                              or "None identified">,
  "provenance": <who holds the object, under what shelfmark, and how it reached them, or
                 "None identified">,
  "contemporary_evidence": <evidence created at the time, or "None identified">,
  "independent_corroboration": <support NOT descending from the same chain, or "None identified";
                                if many sources trace to one report, say so>,
  "contradictory_evidence": <evidence that contradicts it, or "None identified">,
  "scholarly_dispute": <live disagreement AMONG SCHOLARS — a different thing from evidence
                        against it — or "None identified">,
  "evidence_type": "primary_document|archaeological|contemporary_record|near_contemporary_account|
                    later_historical_account|scholarly_inference|tradition|disputed|absent",
  "confidence_label": "high|moderate|low|speculative",
  "why": <1-2 sentences of plain language explaining the assessment>,
  "missing_piece": <the single absent piece of evidence that most limits this>
}}

Produce a JSON object:
{{
  "origin": {{
    "year": <signed int or null>,
    "year_end": <signed int or null>,
    "era_label": <string or null>,
    "precision": "exact|year|decade|century|millennium|era|unknown",
    "node_type": <one of the nine kinds above>,
    "attribution": "established|attributed|disputed|anonymous|not_applicable",
    "source_title": <the evidence, named as a reader would look for it>,
    "summary": <2-4 sentences: what this object is, and why it is the earliest surviving
                evidence relevant to the subject>,
    "citations": [<url>, ...],
    "confidence": <0..1>,
    "evidence": {{ ...dossier... }}
  }},
  "timeline": [
    // chronological, oldest first, evidence only
    {{
      "id": "t1",
      "year": <signed int or null>,
      "year_end": <signed int or null>,        // set when composition SPANS a period
      "era_label": <string or null>,
      "precision": "...",
      "node_type": <one of the nine kinds above>,
      "attribution": "established|attributed|disputed|anonymous|not_applicable",
      "source_title": <the evidence, named as a reader would look for it>,
      "claim": <one sentence stating what this surviving object establishes>,
      "citations": [<url>, ...],
      "confidence": <0..1>,
      "evidence": {{ ...dossier... }}
    }}
  ],
  "conclusions": [
    // What the evidence above is taken to show. AFTER the chain, never inside it.
    // This is where everything that is not a surviving object goes: developments,
    // expectations, movements, influences, reconstructed events, what a text implies.
    {{
      "statement": <one plain proposition>,
      "rests_on": [<ids of the steps this rests on: "origin", "t1", ...>],
      "confidence_label": "high|moderate|low|speculative",
      "reasoning": <how the named evidence gets you to the statement>,
      "dissent": <serious scholarly disagreement with this reading, or "None identified">
    }}
  ],
  "connections": [ ...see CONNECTIONS below... ],
  "reasoning": <short paragraph on the shape of the chain and where it is thin>,
  "confidence": <0..1>
}}

RULES FOR THE CHAIN:

- EVIDENCE ONLY, NO EXCEPTIONS. If a candidate step is not a surviving object, it is a
  conclusion. This holds however central it feels to the story, and however confident the
  scholarship is. A trace of seven documents and eight conclusions is a good trace. A trace of
  fifteen steps where six are inferred developments is a broken one.
- A STEP MUST BE LOCATABLE. Name the repository, shelfmark, excavation report or critical
  edition in "provenance". A step you cannot locate anywhere is a step you should not emit.
- ONE OBJECT, ONE STEP; A CORPUS MAY BE ONE STEP. Where a group was produced together and is
  dated and published as a group — the Dead Sea Scrolls, the four Gospels — one step for the
  group is correct, with the range in "year"/"year_end". Where members carry genuinely different
  dates and evidence and one can fail without the other, they are separate steps.
- COMPOSITION AND SURVIVING COPY ARE NOT THE SAME FACT. The step is dated by composition; the
  physical copy goes in "earliest_surviving_copy" and "provenance". Where a specific manuscript
  is itself the point — because it is the earliest witness, or was found in an excavation — it
  earns its own "manuscript" or "scroll" step, dated to the object.
- SAY WHAT IS MISSING. Where there is no contemporary documentation, that absence is one of the
  most valuable findings here: name what does not exist, in "missing_piece" and in
  "contemporary_evidence", and put the significance of the silence in "conclusions".
- EVERY RESEARCHED STRAND IS ACCOUNTED FOR. The block above says what the rounds found. Material
  that is a surviving object becomes a step; material that is not becomes a conclusion. Nothing
  researched is silently dropped — dropping it tells the reader the evidence was not there.

RULES FOR CONCLUSIONS:

- EVERY CONCLUSION NAMES ITS EVIDENCE in "rests_on". A conclusion resting on no step is allowed
  and is sometimes the most important line in the report — but say so in "reasoning": that this
  is widely held and the chain above does not contain anything that establishes it.
- CONCLUSIONS ARE NOT DATED and are not ordered in time. They are readings, not moments.
- DO NOT LAUNDER A CONCLUSION INTO A STEP by attaching a date to it. That is the exact failure
  this structure exists to prevent.

CONNECTIONS — the most important and most easily faked part of this report.

A chain implies that each item leads to the next. That implication is a CLAIM, and usually the
weakest one on the page: "A, then B, therefore A caused B" is the error this report exists to
expose. Two documents being consecutive in time says nothing about descent.

Emit up to {max_connections} connections:
{{
  "from_id": <id of the earlier item; the origin's id is "origin">,
  "to_id": <id of the later item>,
  "relation": "derives_from|retells|translates|responds_to|contradicts|contemporaneous|attests|
               provides_context|no_established_link",
  "citations": [<url>, ...],
  "evidence": {{
    "mechanism": <ONE sentence: HOW does the earlier item lead to the later one? Name the route —
                  a translator, a manuscript family, a named borrowing, a documented quotation>,
    "supporting_evidence": <what actually evidences this link, or "None identified">,
    "contradictory_evidence": <evidence against the link, or "None identified">,
    "independent_corroboration": <support from a different information chain, or "None identified">,
    "scholarly_dispute": <disagreement among scholars about the link, or "None identified">,
    "evidence_type": <same vocabulary as the dossier, applied to the LINK not the objects>,
    "confidence": <0..1>,
    "confidence_label": "high|moderate|low|speculative",
    "why": <1-2 sentences on how strong this link really is>,
    "missing_piece": <what evidence would settle whether this link is real>
  }}
}}

Connection rules — read these twice:
- Two items being consecutive is NOT a connection. If you cannot state a concrete mechanism and
  name evidence for it, use "no_established_link" and say in "why" that the order is
  chronological only. That is a correct and valuable answer, not a failure.
- A later text quoting or translating an earlier one IS a mechanism, and one you can usually
  evidence: name the passage. Shared themes are not descent, and similarity is not transmission.
- Scholarly consensus that A influenced B is "scholarly_inference", NOT "primary_document",
  no matter how widely the influence is repeated.
- Prefer few well-evidenced connections to many speculative ones, and do not chain every item to
  its neighbour by default.

General rules:
- AT MOST 14 STEPS AND 8 CONCLUSIONS. Every distinct object with its own date, its own evidence
  and its own way of being wrong deserves a step, but a chain longer than this is not a chain a
  reader follows, and the whole answer is generated in one pass — overrun it and the JSON is cut
  off mid-object and the trace fails outright. Choose the load-bearing objects.
- Every claim must be backed by at least one citation URL from the provided list.
- Where the only citations available are tier 4-5, say so in "why" and set the evidence_type to
  what those sources can actually support.
- Where a SOURCE PAGE was read, prefer what it actually says over any summary of it.
- Sources that repeat one upstream report count as ONE. Do not describe them as corroboration.
- "confidence_label" must be consistent with the numeric "confidence" (high >= 0.75,
  moderate 0.5-0.75, low 0.3-0.5, speculative < 0.3).
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
        "query": "discovery excavation report first publication announcement found",
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
        "query": "surviving copies transmission conservation custody archive history",
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
        "query": "radiocarbon dating authentication palaeography analysis authenticity",
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
        "query": "associated finds companion objects same archive catalogue",
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
        "query": "earlier sources predecessors what it was copied from origins",
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
        "query": "later citations influence quoted by reception",
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

{extract_doctrine}

Return JSON:
{{
  "events": [
    {{
      "id": "e1",                              // unique, referenced by connections
      "year": <signed integer or null>,        // the COMPOSITION date
      "era_label": <string or null>,
      "precision": "exact|year|decade|century|millennium|era|unknown",
      "node_type": "text|manuscript|scroll|letter|inscription|document|record|artifact|
                    archaeological_find",   // what kind of SURVIVING thing this is
      "is_evidence": <true only if this is a surviving object someone could go and examine;
                      false for readings, developments, movements, expectations, inferred events>,
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
