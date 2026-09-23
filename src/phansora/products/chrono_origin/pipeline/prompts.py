"""Prompt templates for the trace pipeline.

A note on cost, because it shaped every template here. The evidence doctrine was
once the most expensive text in the product: pasted into all ~20 search calls of
a trace, where it could not be acted on — a search step summarising five snippets
cannot weigh a manuscript's provenance, so it paid full price for instructions it
had no way to follow.

It was split, then dropped: the tier hierarchy left the expand extract (see the
note below), and the short form left with the two-prompt expand path that was its
last caller. What grades a source now is code — source_policy.default_tier from
the URL, and the caps it puts on evidence_type — which was always the half that
could not talk itself up.
"""
from __future__ import annotations

from typing import Optional

# Expanding used to carry a tier hierarchy of its own — five ranks of source, with
# "tiers 4-5 are leads, never the basis of a claim" under them. It was removed
# deliberately: on an axis that asks for earlier parallels, much of the honest material
# surfaces first on general-web pages, and a ranking applied before the claim is even
# read narrows what comes back rather than grading it.
#
# What replaced it is not nothing, and it is not in a prompt. Tiers are assigned in code
# from the URL (source_policy.default_tier), and cap what a claim standing on them may
# call itself — a claim resting on a wiki cannot describe itself as a primary document,
# and because claim_class derives from evidence_type that cap reaches the marker on the
# board. A grader that cannot talk itself up was always the more reliable half.
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
Using live web search, trace the complete historical lineage of "{title}" from the earliest known foundations to the present day.

Search broadly and repeatedly for the historical sources, predecessor works, larger source collections, writings, concepts, names, words, traditions, artifacts, inscriptions, materials, technologies, manuscripts, translations, and later developments connected to "{title}".

Search backward beyond the first appearance of "{title}". When you discover an earlier source, work, tradition, concept, technology, material, or predecessor, search for it too.

The goal is to produce a large, complete chronological list with rich metadata, not a brief summary.

Include every relevant historical item found in the research.

If you find an individual item that belongs to a larger source, collection, corpus, tradition, technological system, or historical development, include that larger source or parent as a separate item too.

If you find an original work and a translation, adaptation, version, copy, reproduction, manuscript, fragment, or later form, include them separately.

If you find a larger corpus, collection, tradition, technology, or historical movement and an important individual item within it, include both separately.

Include important:

* archaeological and material foundations
* raw materials
* inscriptions and early physical evidence
* names and linguistic predecessors
* cultural and historical traditions
* predecessor works and source materials
* individual important works or examples
* larger collections, corpora, or parent traditions
* early concepts and practices
* important technologies and techniques
* primitive or early forms
* major intermediate developments
* surviving artifacts, copies, manuscripts, or specimens
* translations and adaptations
* regional or cultural transmission
* major later versions and forms
* important people or groups directly involved in its development
* major historical developments
* important modern developments
* present-day form

Do not summarize several different historical items into one entry.
Do not omit a larger source, collection, tradition, technology, or parent development because you already listed something contained within it.
Do not omit an important individual item because you already listed the larger source, collection, tradition, or development containing it.
Do not replace an original with a translation, adaptation, manuscript, fragment, surviving example, reproduction, or later version.
Do not include speculative connections merely because two things are similar. Include items supported by the web research as historically relevant to the development or transmission of "{title}".

After researching, combine all valid items you found, remove only true duplicates, and sort the entire list by date from oldest to newest.

Return the result as a JSON array and nothing else. One object per item, in date order, oldest first:

[
  {
    "title": "Item title",
    "date": "Date or date range as the research supports it, e.g. c. 3400 BCE, 3rd century BCE, 1400-1200 BCE, 1611 CE, present day",
    "origin": "Geographic origin and provenance",
    "material": "Physical material, medium, and language",
    "authorship": "Authorship or source community",
    "significance": "Historical significance and function",
    "is_collection": true or false
  }
]

Every object must have "title" and "date". Leave any other TEXT field as an empty string when the research does not support it. Do not guess a value to fill a field.

"is_collection" is a true/false judgement and must be present on every object. Set it true when the item is a COLLECTION — a set of separately made works gathered under one name: an anthology, a canon, a body of collected letters, a manuscript cache, a series, a catalogue, a product line, a standards family, a repertoire. Set it false for a single work, object, person, place or event.

This decides which question the item is asked later: a collection is expanded into the works it is made of, where a single thing is expanded into the record of how it emerged. A collection marked false hands back manuscripts of itself instead of its contents. Most items are not collections, but any lineage of any length contains several, so do not default the whole list to false.

It is a judgement AT A LEVEL, not a fact about a subject. A canon is a collection of books; one book of it may itself be a collection of poems or letters; a single letter is not a collection. Judge each item as you have named it.

Return as many valid items as the research supports. Do not shorten the list.

No prose before or after the array. No code fences.

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
  "provenance": <who holds it now, under what identifier — shelfmark, accession,
                  catalog or lot number, registration — and how it reached them, or
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

COLLECTIONS. For every item, decide whether it is a single thing or a COLLECTION — a set of
separately made works gathered under one name. An anthology, a canon, a body of collected
letters, a manuscript cache, a series, a catalogue, a product line, a standards family, a
repertoire. Mark each of those `is_collection: true`. Most items are not collections, but
collections are common enough in any lineage that assuming false is wrong.

This is not cosmetic. A collection is asked a different question when it is expanded: it
returns the works it is made of, where a single thing returns the record of how it emerged.
A collection marked false hands back manuscripts OF itself instead of its contents.

It is a judgement AT A LEVEL, not a fact about a subject. A canon is a collection of books; a
single book of it may itself be a collection of poems or letters; one letter is not a
collection. Judge each item as it is named here.

Produce a JSON object:
{{
  "origin": {{
    "year": <signed int or null>,
    "year_end": <signed int or null>,
    "era_label": <string or null>,
    "precision": "exact|year|decade|century|millennium|era|unknown",
    "node_type": <one of the labels above>,
    "is_collection": <true when this item is a set of separately made works gathered under
                      one name, per COLLECTIONS above; false for a single work, object,
                      person, place or event>,
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




# What each expansion mode goes looking for, and the one call that asks it.
#
# There used to be two prompts and three model steps: a grounded search that wrote a
# summary, a pass that fetched source pages, and an extraction that turned the lot into
# JSON. The axis was split across the first and the last, so steering a result meant
# editing both and trusting the summary in the middle to carry what the extractor had
# been told to want. It did not. Asked for the earlier parallels of a subject, the chain
# returned old documents from the right part of the world and none of the counterparts
# anyone would name — the search found records because the extractor's rules about
# records were nowhere near it.
#
# One call now. What is written here is what the model is judged on, and the answer comes
# back in the same breath as the search.
#
# Each mode is one body, addressed to `{subject}` — the NODE being expanded, not the
# trace. Bodies are inserted, never `.format()`ed, so the JSON braces inside them need no
# escaping; the subject is substituted by `.replace()` in `expand_body`.
#
# All three answer in the SAME six flat fields, so one adapter in the orchestrator turns
# any of them into the board's events and edges. What varies is the classification
# vocabulary, which differs by axis because "direct_source" is the wrong word for a
# discovery record and "records" is the wrong word for a parallel. Every value any mode
# can emit is mapped in orchestrator._RELATION_WORDS.

# ASKING FOR JSON IS WHAT STOPPED THE SEARCH HAPPENING, so this is no longer part of any
# mode body — it is the second call's whole job (EXPAND_EXTRACT_PROMPT below).
#
# `google_search` is model-ELECTED and cannot be forced, and a prompt that ends "Return JSON
# only" is complied with: the model goes straight to emitting the shape and never reaches for
# the tool. MEASURED against the live API, one call per line, same subject:
#
#   paragraphs 1-8 of the discovery body   1835 chars   searched
#   + "Return JSON only ..."               1903 chars   no search
#   + the template below                   2016 chars   no search
#
# So every expand since the feature shipped answered from recall. It looked like a tier
# problem — flash-lite is cheap and declining a tool is what cheap tiers do — but a plain
# question grounds on flash-lite and the full expand prompt grounds on NEITHER flash-lite nor
# flash, which is the opposite result. The client already guarded the API-level version of
# this (see `json_out`: a forced mime type alongside a search tool is refused); the
# prompt-level version does the same damage and nothing was watching for it.
#
# The trace pipeline never had the bug because it was always two calls: grounded_search to
# research, then reason_json to shape. Expand now matches it.
# The naming rule below is here, in the one template every mode's names come out of, because
# a name in this product is not a label — it is the next prompt's input. Every name becomes a
# card, and expanding that card feeds the string back verbatim as `{subject}`. So a qualifier
# written into a name is not decoration, it is a restriction imposed on every expansion made
# under it afterwards, by a call that has no way to know the qualifier was ever optional.
#
# Measured: a card minted as a collection's name plus a parenthesis naming two of its
# divisions expanded into the works of those two divisions and stopped — correctly, for the
# subject it was handed. The corpus branch was not at fault and neither was the model; the
# scope had been baked into the title one call earlier, where nothing was checking. Renaming
# the card fixes that card. This fixes the next one.
_JSON_TAIL = """\
Return JSON only, ordered chronologically from earliest to latest:

{"events":[{"name":"","year":<signed integer, negative = BCE>,"group":"","relation":"","shared":"","url":"","is_collection":false}]}

One object per result in the research. If the research contains results, the list is never
empty.

`name` — what the result is called. Something that HAS a name takes its own name and nothing
else: never the name plus a parenthesis, subtitle or appositive narrowing it to the part
relevant here, because expanding a card sends this exact string back as the next subject to
be researched, and a scope written into a name limits everything found under it later. A
result that is an EVENT rather than a named thing takes a short phrase saying what happened.

`shared` — why it is connected. `group` — which part or category it belongs to.

`is_collection` — true when the result is itself a set of separately made works gathered
under one name: an anthology, a canon, collected letters, a series, a catalogue, a product
line, a standards family. False for a single work, object, person, place or event."""


EXPAND_EXTRACT_PROMPT = _JSON_TAIL + """

THE SHAPE COMES FIRST AND THE RESEARCH LAST, which is not a matter of taste: with the
research first and the template trailing after it, this call answered `{}` every time —
measured, same research blob, both model tiers. Put the template at the top and the same
blob yields one event per result.

Every distinct result in the research below becomes one event. Add nothing the research does
not support, and drop nothing it does. Keep each result's own source URL where the research
gives one.

Research already gathered about "{subject}":
{research}

{citations_block}
"""


# What an axis other than discovery does when the subject is a collection.
#
# The failure it prevents: ask a collection for People and the answer spans everyone who
# touched any member of it — for a canon, a cast list fifteen centuries wide with no frame.
# That is the same shapelessness that got "Historical Context" removed. The fix is not to
# block the axis, which would also block the axes that are BETTER at collection level (a
# manuscript witnesses the whole collection, not one book of it); it is to say which level
# to answer at.
COLLECTION_RULE = """\
{subject} is a COLLECTION — a set of independently transmitted works gathered under one
name. ANSWER AT THE LEVEL OF THE COLLECTION.

- Return what belongs to the collection as a whole: how it was assembled, who assembled,
  transmitted or governed it, where and under what rules, and what it is held to be.
- Do NOT enumerate what belongs to a single member of it. Those are reached by expanding
  that member, and returned here they are a list with no frame — the members of a large
  collection can span centuries and disagree with each other.
- Where something is a witness to, or a property of, the WHOLE collection, say so. Where it
  is the clearest instance and comes from one member, name the member it comes from.

"""


EXPAND_MODES = {
    # The corpus branch at the top exists because a collection has no discovery. Asked how
    # the Hebrew Bible "first emerged", the mode answered correctly and uselessly: the Ketef
    # Hinnom amulets and the paper announcing them, two cards for a library of thirty-nine
    # books, because the prompt described one object with one emergence and the model had no
    # licence to read the subject as plural.
    #
    # The branch then asked for too much. It named the works AND wanted each one's
    # manuscripts, discoveries and publications researched in the same call, with an escape
    # hatch if there were too many — three jobs and a get-out, so the model split the
    # difference and returned six whole-corpus objects (the Dead Sea Scrolls, the Aleppo and
    # Leningrad codices) each tagged with a book name to satisfy `group`. The same failure
    # as before, now wearing per-book labels.
    #
    # So a corpus expands into its works and stops. One job, and the cheap one: naming what
    # a collection is made of needs no search, which is the step that was failing. Each work
    # comes back as a card, and expanding THAT is the discovery research — which is the
    # non-corpus branch below, unchanged, because a single book is a single subject.
    "discovery": {
        "query": "discovered first published announced found introduced recorded",
        "label": "Path to Discovery",
        # Two bodies, chosen by `is_collection` rather than by the model.
        #
        # This branch used to open "Before searching, decide whether {subject} is a single
        # historical subject or a COMPOSITE CORPUS", and the model re-decided on every call.
        # It did not decide the same way twice: the same Hebrew Bible card returned nine
        # works in one run and eleven in the next, eleven minutes apart. The judgement is
        # now made once, when the node is minted, and carried on the node — so this prompt
        # states it rather than asking for it, and every subject stops paying for the
        # paragraph that posed the question.
        "collection_body": """\
{subject} is a COLLECTION — a set of independently transmitted works gathered under one
name. ITS CONSTITUENT WORKS ARE THE WHOLE ANSWER:

- Return one result per principal constituent work, named as the work itself.
- Return nothing else. No manuscripts, discoveries, excavations, codices or publications: a
  collection has no discovery of its own, and a manuscript of the whole collection is not one of
  its works. Those belong to the individual works, and the reader reaches them by expanding
  the work they want.
- Do not stop at a representative few. A collection of forty works returns forty results.
- The subject may reach you carrying a qualifier — a parenthesis, subtitle or appositive
  naming some of the collection's divisions, periods or parts. Expand the collection itself
  anyway. A qualifier records which part the reader arrived through; it does not shrink what
  the collection contains, and treating it as a limit redefines the collection as whichever
  part of it someone once named. Use the divisions it names for `group`, not as a filter.
- `relation` is `direct_source` — the collection is made of these.
- `year` is when that work was composed or assembled, as closely as it is known, and null
  where it is not. A work with no defensible date is still returned.
- `group` is the division of the collection the work belongs to when the collection has divisions,
  and the collection's own name when it does not.
- Listing what a collection is made of does not need a live search. Search only to confirm the
  contents if you are unsure of them.

""",
        # The FALLBACK body, and it asks the model to decide for itself — restored
        # verbatim from 4b54f9b because it worked.
        #
        # 4e6a9ca replaced it with a hard dependency on `is_collection`, and that was a
        # regression the moment the flag was not set: a node minted before the field
        # existed, a node served from a cached trace, or a synthesis that answered false
        # all took this branch and returned manuscripts OF a collection instead of its
        # contents. A board that had been expanding the Hebrew Bible into Genesis, Exodus
        # and the rest stopped doing so.
        #
        # So the flag is now a SHORT-CUT, not a precondition. Known true -> collection_body,
        # which states it and skips the question. Anything else -> this, which asks the
        # question the way it always did and cannot be wrong-footed by an absent flag.
        # The cost is that an unflagged expansion pays for the longer body again, which is
        # the correct trade: the saving was never worth the failure it bought.
        "body": """\
Before searching, decide whether {subject} is a single historical subject or a COMPOSITE
CORPUS — a collection of independently transmitted works. An anthology, a canon, a
manuscript library, a multi-part textual tradition, a body of writings assembled over time —
and equally a catalogue, a product line, a series, a repertoire, a standards family, or any
other set of separately made things gathered under one name.

IF IT IS A COMPOSITE CORPUS, THE CONSTITUENT WORKS ARE THE WHOLE ANSWER:

- Return one result per principal constituent work, named as the work itself.
- Return nothing else. No manuscripts, discoveries, excavations, codices or publications: a
  corpus has no discovery of its own, and a manuscript of the whole collection is not one of
  its works. Those belong to the individual works, and the reader reaches them by expanding
  the work they want.
- Do not stop at a representative few. A corpus of forty works returns forty results.
- The subject may reach you carrying a qualifier — a parenthesis, subtitle or appositive
  naming some of the collection's divisions, periods or parts. Expand the collection itself
  anyway. A qualifier records which part the reader arrived through; it does not shrink what
  the collection contains, and treating it as a limit redefines the collection as whichever
  part of it someone once named. Use the divisions it names for `group`, not as a filter.
- `relation` is `direct_source` — the corpus is made of these.
- `year` is when that work was composed or assembled, as closely as it is known, and null
  where it is not. A work with no defensible date is still returned.
- `group` is the division of the corpus the work belongs to when the corpus has divisions,
  and the corpus's own name when it does not.
- Listing what a corpus is made of does not need a live search. Search only to confirm the
  contents if you are unsure of them.

IF {subject} IS NOT A COMPOSITE CORPUS:

Using live web search, find the surviving RECORDS of how {subject} first emerged, was
documented, or became known.

Include whichever of these actually exist:

- earliest surviving text, manuscript, inscription, account, or depiction
- discovery, excavation, recovery or first-sighting records
- first publication announcing the find
- the field, laboratory, survey or investigation report
- the observation or experiment that established it, and the record that reported it
- patent, filing, or registration
- notes, drawings, photographs, facsimiles, or transcriptions made at the time
- catalog, accession, inventory, registry or listing record
- the study that identified, dated, authenticated, attributed or deciphered it
- later discoveries that materially improved knowledge of the subject

The emergence itself is an event and cannot be returned; the record that captures it can. Date
each to when the RECORD was made, not to what it records: a report is dated to its publication,
not to the work or the event it describes.

The goal is to produce a large, complete chronological list with rich metadata, not a brief
summary.

For each result:

- Name the specific thing it records, and the work it records it about.
- Put the kind of record in `group`.
- Classify it as `direct_source`, `records`, or `disputed_parallel`.
- Use the earliest defensible attestation date.
- Include every qualifying result found, rather than selecting only the strongest examples.

Before returning the results, search separately for each category above. Finding one result in
a category does not complete that category.

""",
    },
    # Written by the owner and shipped as supplied. The two things it does that nothing
    # before it did: it refuses to let a parallel be disqualified for lacking a
    # transmission nobody has ever shown, and it asks for the whole motif rather than one
    # participant — a nursing-mother scene is a parallel to a nursing-mother scene, and
    # reduced to either figure alone it stops being one.
    "earlier": {
        "query": "earlier parallels analogues counterparts precedents predecessors origins",
        "label": "Earlier Parallels & Origins",
        "body": """\
Using live web search, find earlier parallels and origins for {subject}.

Include two balanced categories:

1. PARALLELS — earlier cross-cultural figures, stories, motifs, rituals, symbols, relationships,
practices, mechanisms, designs, and visual/iconographic traditions sharing specific features with
{subject}. Include well-known and disputed comparisons even when no influence or transmission is
demonstrated.

2. ANCESTRY — identifiable sources, predecessors, traditions, inventions, translations, or
practices that {subject} demonstrably or probably drew upon or descended from.

Search textual, narrative, theological, ritual, symbolic, relational, and visual/iconographic
categories equally. Consider {subject} together with associated figures and relationships. Do not
reduce a relational or iconographic parallel to one participant; name the complete motif, scene,
relationship, or tradition and all figures essential to the comparison.

The goal is to produce a large, complete chronological list with rich metadata, not a brief
summary.

For each result:

- Name the specific shared or transmitted feature.
- Classify it as `direct_source`, `probable_influence`, `possible_influence`,
  `independent_parallel`, or `disputed_parallel`.
- Use the earliest defensible attestation date.
- Include every qualifying result found, rather than selecting only the strongest examples.

Before returning the results, perform separate completeness searches for each category above. For
relational and iconographic parallels, independently search parent-child, mother-child,
father-child, birth, family, teacher-follower, adversary, death, enthronement, and protective
relationships or scenes. Include every famous or frequently proposed qualifying comparison found;
finding one result in a category does not complete that category.

""",
    },
    "people": {
        "query": "people person who involved role figure founder maker author",
        "label": "People",
        "body": """\
Using live web search, find the INDIVIDUALS directly associated with {subject} and what each one did.

Include whichever of these actually exist:

- whoever made, wrote, built, designed, founded or commissioned it
- whoever transmitted, copied, translated, edited, manufactured or distributed it
- whoever recorded, reported, catalogued, studied, dated or authenticated it
- whoever opposed, suppressed, disputed, prosecuted or competed with it
- whoever it is chiefly named for, attributed to, or associated with in later memory

A person qualifies by a stated role, not by being alive nearby. Give the role in each case.
Where attribution is traditional rather than established, say so rather than dropping the person.

The goal is to produce a large, complete chronological list with rich metadata, not a brief
summary.

For each result:

- Name the specific way it bears on {subject}, not merely that it does.
- Classify it as `direct_source`, `records`, `contemporaneous` or `context`.
- Use the earliest defensible attestation date.
- Include every qualifying result found, rather than selecting only the strongest examples.

Before returning the results, search separately for each category above. Finding one result in
a category does not complete that category.

""",
    },
    "events": {
        "query": "events happened occurred incident episode milestone",
        "label": "Events",
        "body": """\
Using live web search, find the significant EVENTS connected to {subject}.

Include whichever of these actually exist:

- what brought it about, and what it in turn brought about
- its making, release, publication, enactment, construction or first use
- disruptions to it: loss, destruction, suppression, banning, decline, revival
- disputes, trials, controversies, rivalries and conflicts it was the subject of
- moments it was adopted, imitated, superseded, rediscovered or reinterpreted

An event qualifies by bearing on {subject}, not by sharing its period. Say what the bearing is.

The goal is to produce a large, complete chronological list with rich metadata, not a brief
summary.

For each result:

- Name the specific way it bears on {subject}, not merely that it does.
- Classify it as `direct_source`, `records`, `contemporaneous` or `context`.
- Use the earliest defensible attestation date.
- Include every qualifying result found, rather than selecting only the strongest examples.

Before returning the results, search separately for each category above. Finding one result in
a category does not complete that category.

""",
    },
    "chronology": {
        # Two lines, after the six-category version returned an essay.
        #
        # It asked for competing dates, relative chronology, the METHODS that date the
        # subject, and revisions where a date was overturned — every one of which is an
        # ARGUMENT rather than a thing. The research call answered them well: 6,131 and
        # 7,301 characters on Genesis, the Mesopotamian dependency, palaeographic dating
        # of the Qumran fragments, the Albright-to-Van-Seters revision. Then the shaping
        # call found no discrete items with names and years in any of it and returned
        # nothing twice, which the user was told as "no evidence found for this
        # direction" — the opposite of what had happened.
        #
        # Every other axis asks for THINGS (people, places, texts, objects) and things
        # become cards. This one asks for events with dates, and nothing else.
        "query": "chronology timeline date order events sequence",
        "label": "Dates & Chronology",
        "body": """\
Using live web search, list the chronology of {subject} in date order, oldest first.

One entry per event: the date, and one line on what happened.

Return every event the research supports, rather than a selection of the notable ones.
""",
    },
    "places": {
        "query": "place location where site region route geography",
        "label": "Places",
        "body": """\
Using live web search, find the PLACES that matter to {subject}, and how it moved between them.

Include whichever of these actually exist:

- where it originated, was made, was found, or was first attested
- where it was kept, held, housed, installed, performed or used
- routes and movements: trade, migration, pilgrimage, distribution, export, exile, looting
- places it spread to, was copied in, or took a distinct local form
- sites excavated, surveyed or documented in connection with it

Name the specific place rather than the region when the research supports it, and say what happened there.

The goal is to produce a large, complete chronological list with rich metadata, not a brief
summary.

For each result:

- Name the specific way it bears on {subject}, not merely that it does.
- Classify it as `direct_source`, `contemporaneous` or `context`.
- Use the earliest defensible attestation date.
- Include every qualifying result found, rather than selecting only the strongest examples.

Before returning the results, search separately for each category above. Finding one result in
a category does not complete that category.

""",
    },
    "lineages": {
        "query": "lineage descent ancestry succession dynasty genealogy transmission",
        "label": "Genealogies & Lineages",
        "body": """\
Using live web search, find the LINES OF DESCENT running into and out of {subject}.

Include whichever of these actually exist:

- ancestry: what it descends from, and what that descended from
- descendants: what descends from it, including branches that diverged
- succession: who or what held it, followed it, or inherited it, in order
- dynasties, houses, families and lines of office where those are the subject
- intellectual and craft lineages: teacher to student, school to school, workshop to workshop
- transmission chains: how it passed from hand to hand, copy to copy, or version to version

Give the order of descent, and say what each step rests on. Where a line is claimed rather than
demonstrated, return it as claimed rather than omitting it.

The goal is to produce a large, complete chronological list with rich metadata, not a brief
summary.

For each result:

- Name the specific way it bears on {subject}, not merely that it does.
- Classify it as `direct_source`, `probable_influence` or `context`.
- Use the earliest defensible attestation date.
- Include every qualifying result found, rather than selecting only the strongest examples.

Before returning the results, search separately for each category above. Finding one result in
a category does not complete that category.

""",
    },
    "nations": {
        "query": "people nation culture civilization kingdom tribe population group",
        "label": "Nations & Peoples",
        "body": """\
Using live web search, find the PEOPLES and COLLECTIVE GROUPS bound up with {subject}.

Include whichever of these actually exist:

- the cultures, civilizations or societies it arose within
- kingdoms, states, empires, polities and administrations that produced, used or governed it
- tribes, clans, ethnic groups, communities and diasporas connected to it
- the populations who made it, used it, carried it, or were affected by it
- groups that opposed, displaced, absorbed, or were displaced by it

Name the group as the research names it, and say what its relationship to {subject} was.
A group qualifies by a stated connection, not by geographic proximity.

The goal is to produce a large, complete chronological list with rich metadata, not a brief
summary.

For each result:

- Name the specific way it bears on {subject}, not merely that it does.
- Classify it as `direct_source`, `contemporaneous` or `context`.
- Use the earliest defensible attestation date.
- Include every qualifying result found, rather than selecting only the strongest examples.

Before returning the results, search separately for each category above. Finding one result in
a category does not complete that category.

""",
    },
    "beliefs": {
        "query": "belief concept idea doctrine principle theory symbol meaning",
        "label": "Beliefs & Concepts",
        "body": """\
Using live web search, find the IDEAS bound up with {subject}.

Include whichever of these actually exist:

- doctrines, teachings, tenets and positions it asserts or embodies
- theories, principles, models and design ideas behind it
- the worldview or cosmology it assumes, and the framework it argues within
- symbols, motifs and imagery it carries, and what those were taken to mean
- ideas it opposed, replaced, or was itself opposed by
- later reinterpretations that changed what it was understood to mean

State the idea itself, not merely that an idea was involved. Where an interpretation is contested,
name whose it is.

The goal is to produce a large, complete chronological list with rich metadata, not a brief
summary.

For each result:

- Name the specific way it bears on {subject}, not merely that it does.
- Classify it as `direct_source`, `probable_influence` or `context`.
- Use the earliest defensible attestation date.
- Include every qualifying result found, rather than selecting only the strongest examples.

Before returning the results, search separately for each category above. Finding one result in
a category does not complete that category.

""",
    },
    "texts": {
        "query": "text source document manuscript record edition account writing",
        "label": "Texts & Sources",
        "body": """\
Using live web search, find the WRITTEN AND DOCUMENTARY SOURCES for {subject}.

Include whichever of these actually exist:

- primary texts of it, and the earliest surviving witnesses to those
- manuscripts, copies, recensions, fragments and inscriptions bearing it
- documentary evidence: records, registers, accounts, correspondence, filings, specifications
- editions, translations, critical editions and facsimiles, with their editors
- contemporary accounts and later testimonies describing it
- the scholarship that established, dated, deciphered or disputed the text

Distinguish the WORK from a witness to it: a copy is not the composition, and an edition is not
the copy. Return both, named separately.

The goal is to produce a large, complete chronological list with rich metadata, not a brief
summary.

For each result:

- Name the specific way it bears on {subject}, not merely that it does.
- Classify it as `direct_source`, `records` or `contemporaneous`.
- Use the earliest defensible attestation date.
- Include every qualifying result found, rather than selecting only the strongest examples.

Before returning the results, search separately for each category above. Finding one result in
a category does not complete that category.

""",
    },
    "objects": {
        "query": "object artifact physical remains monument specimen evidence material",
        "label": "Objects & Artifacts",
        "body": """\
Using live web search, find the PHYSICAL THINGS that evidence {subject}.

Include whichever of these actually exist:

- surviving examples, specimens, prototypes and production pieces
- monuments, structures, sites and installations
- inscriptions, marks, stamps, seals, signatures and maker's marks
- tools, instruments, equipment and machinery used to make or use it
- artworks, images, depictions and representations of it
- materials and samples analysed as evidence for it

Say where the object is now and what it evidences. An object qualifies by being physical evidence
bearing on {subject}, not by depicting a related theme.

The goal is to produce a large, complete chronological list with rich metadata, not a brief
summary.

For each result:

- Name the specific way it bears on {subject}, not merely that it does.
- Classify it as `direct_source`, `records` or `contemporaneous`.
- Use the earliest defensible attestation date.
- Include every qualifying result found, rather than selecting only the strongest examples.

Before returning the results, search separately for each category above. Finding one result in
a category does not complete that category.

""",
    },
    "practices": {
        "query": "law rule custom practice ritual institution regulation procedure",
        "label": "Laws, Rules & Practices",
        "body": """\
Using live web search, find the RULES AND PRACTICES governing or surrounding {subject}.

Include whichever of these actually exist:

- laws, statutes, decrees, commands, charters and rulings bearing on it
- regulations, standards, specifications, codes and licensing that constrain it
- customs, conventions, etiquette and unwritten rules around it
- rituals, ceremonies, observances and procedures it is used in or requires
- institutions, offices, guilds, bodies and organisations that administer it
- prohibitions, restrictions, bans and enforcement against it

State the rule and who it bound. Say whether it was actually enforced where the research says so.

The goal is to produce a large, complete chronological list with rich metadata, not a brief
summary.

For each result:

- Name the specific way it bears on {subject}, not merely that it does.
- Classify it as `direct_source`, `contemporaneous` or `context`.
- Use the earliest defensible attestation date.
- Include every qualifying result found, rather than selecting only the strongest examples.

Before returning the results, search separately for each category above. Finding one result in
a category does not complete that category.

""",
    },
    "traditions": {
        "query": "tradition legend prophecy vision oral account folklore story",
        "label": "Visions, Prophecies & Traditions",
        "body": """\
Using live web search, find what is TRANSMITTED ABOUT {subject} as tradition rather than as record.

Include whichever of these actually exist:

- recorded visions, dreams, revelations and experiences associated with it
- prophecies, predictions and forecasts about it, and whether they were held to be fulfilled
- oral traditions, folklore and stories transmitted about it
- legends and founding accounts: how it is said to have begun
- attributions and claims made about it that the evidence does not establish
- popular beliefs, rumours and received wisdom about it, and where those arose

Report what is transmitted and who transmits it, without asserting it as fact and without
dismissing it. Where the record contradicts the tradition, say what the record says.
Date each to when the tradition is first ATTESTED, not to what it describes.

The goal is to produce a large, complete chronological list with rich metadata, not a brief
summary.

For each result:

- Name the specific way it bears on {subject}, not merely that it does.
- Classify it as `records`, `probable_influence`, `disputed_parallel` or `context`.
- Use the earliest defensible attestation date.
- Include every qualifying result found, rather than selecting only the strongest examples.

Before returning the results, search separately for each category above. Finding one result in
a category does not complete that category.

""",
    },
}


# A node reached by expanding another one is named as the CHILD knows itself, and that name
# is often ambiguous on its own: "Genesis" is a book, a band, a console and a Sega magazine.
# The board knows what it came out of, so the prompt says so.
#
# The second sentence is not padding. Naming a parent without it sends the model off to
# research the PARENT — the failure behind e66ff44, where a corpus expansion came back as
# whole-corpus objects wearing book names. "Research X itself, not Y" pins the subject.
#
# Only the IMMEDIATE parent. The trace root would be noise and worse than noise: expanding
# an unrelated object inside a trace of Jesus Christ should not tell the model to look for
# one connected to Jesus Christ.
_PART_OF_LINE = ('\n"{subject}" here means the one that is part of {part_of} — not any other '
                 'thing of that name. Research {subject} itself, not {part_of}.\n')


def part_of_line(subject: str, part_of: Optional[str]) -> str:
    """The disambiguating line, or nothing when this node has no parent.

    Empty for a node that came from the trace itself, which has only the trace title above
    it — and a trace title is a subject, not a container.
    """
    subject = (subject or "").strip()
    parent = (part_of or "").strip()
    if not parent or not subject or parent.casefold() == subject.casefold():
        return ""
    return _PART_OF_LINE.replace("{subject}", subject).replace("{part_of}", parent)


def expand_mode(mode: str) -> dict:
    """The directives for a mode, defaulting to the one the dialog preselects."""
    return EXPAND_MODES.get(mode) or EXPAND_MODES["discovery"]


def expand_extract(subject: str, research: str, citations_block: str) -> str:
    """The shaping call's prompt, with the research already gathered dropped into it.

    `.replace`, not `.format`, for the same reason `expand_body` uses it: this prompt ends in
    the JSON template, and `.format` reads every brace in that template as a field and raises
    KeyError on the first one. Which it did — caught by a test, not by production, where the
    orchestrator's except-and-fall-back would have swallowed it and then read the research
    prose as JSON, producing an expansion with zero events and no error anywhere.
    """
    return (
        EXPAND_EXTRACT_PROMPT
        .replace("{subject}", subject or "this subject")
        .replace("{research}", research or "")
        .replace("{citations_block}", citations_block or "")
    )


def expand_body(mode: dict, subject: str, is_collection: bool = False) -> str:
    """A mode's body with the node's own title in it, and its level set by the node.

    `.replace`, not `.format`: the body ends in a JSON template, and every brace in it
    would otherwise have to be doubled by hand — which is exactly the kind of edit that
    silently breaks a prompt the next person pastes in.

    A collection takes a different body where one exists (discovery returns the works it
    is made of) and COLLECTION_RULE in front of the usual one where it does not (every
    other axis answers about the collection rather than about its members).
    """
    body = mode.get("collection_body") if is_collection else None
    if body is None:
        body = (COLLECTION_RULE + mode["body"]) if is_collection else mode["body"]
    return body.replace("{subject}", subject or "this subject")


def format_existing_block(existing) -> str:
    """What the board already shows, so an expansion can avoid handing it back.

    NO LONGER WIRED INTO EXPAND_PROMPT. The block rode on every expand call and could run
    to 150 titles, and the client filters duplicates by title on arrival anyway ("told not
    to repeat; enforced anyway"), so it was paying tokens for a guarantee it already had.
    Kept, with its tests, because putting it back is one placeholder.

    The most common way an expansion wastes a call is by returning the anchor's own
    neighbour — ask for material around Paul's letters and the gospels come back, which
    are already the next step along. Naming them is cheaper than any instruction about
    novelty, because the model cannot avoid a duplicate it has not been shown.
    """
    items = [str(t).strip() for t in (existing or []) if str(t or "").strip()]
    if not items:
        return "(nothing else on the timeline yet)"
    # 40 was sized for a board where one expansion added six cards. One now adds up to a
    # hundred, so a list cut at 40 stops naming most of what is already there — and every
    # title it fails to name is a repeat the user pays for and then watches the client
    # drop as a duplicate.
    return "\n".join(f"- {t}" for t in items[:150])


# The wrapper, and everything in it earns its line.
#
# `Search query:` — the DeepSeek client reads this line to build its web query
# (_derive_queries). Without it the client falls back to the first QUOTED string in the
# prompt, which in a body ending in a JSON template is the word "events". Gemini elects
# its own queries and ignores this line.
#
# The anchor is QUOTED there for the same reason: _derive_queries takes the first quoted
# string as a second search angle, and an unquoted anchor leaves "events" as the first
# quoted string in the prompt. So the quotes buy two things — a phrase search on the
# subject, and a second angle that is the subject rather than a JSON key.
#
# The exclusions come BEFORE the body, not after it: the body ends with the JSON
# template, and the last instruction in a prompt is the one that gets followed.
#
# `{context_line}` carries the dashboard's context box, so "Mercury" the planet stays
# distinct from "Mercury" the god. Empty when the box is.
EXPAND_PROMPT = """\
Search query: "{parent_source_title}" {mode_query}
{part_of_line}{context_line}
{mode_body}
"""
