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
    "significance": "Historical significance and function"
  }
]

Every object must have "title" and "date". Leave any other field as an empty string when the research does not support it. Do not guess a value to fill a field.

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

_JSON_TAIL = """\
Return JSON only, ordered chronologically from earliest to latest:

{"events":[{"name":"","year":<signed integer, negative = BCE>,"group":"","relation":"",
"shared":"","url":""}]}"""


EXPAND_MODES = {
    "discovery": {
        "query": "discovered excavation first published announced found record",
        "label": "Path to Discovery",
        "body": """\
Using live web search, find the surviving RECORDS of how {subject} first emerged, was
documented, or became known.

Include whichever of these a subject of this kind actually has: the earliest surviving text,
account or depiction of it; the first publication announcing it; the report of the excavation,
observation or experiment that established it; the patent or filing; the notes, drawings or
photographs made at the time; the accession record; the study that identified or deciphered it.

The emergence itself is an event and cannot be returned; the record that captures it can. Date
each to when the RECORD was made, not to what it records: an excavation report is dated to its
publication.

The goal is to produce a large, complete chronological list with rich metadata, not a brief
summary.

For each result:

- Name the specific thing it records about {subject}.
- Classify it as `direct_source`, `records`, or `disputed_parallel`.
- Use the earliest defensible attestation date.
- Include every qualifying result found, rather than selecting only the strongest examples.

Before returning the results, search separately for each category above; finding one result in a
category does not complete that category.

"""
        + _JSON_TAIL,
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

"""
        + _JSON_TAIL,
    },
    "context": {
        "query": "contemporary events culture society technology at the time",
        "label": "Historical Context",
        "body": """\
Using live web search, find what was happening AROUND {subject} at the time.

Include contemporaneous events, the culture and society it sat in, the people and institutions
involved, the technologies and materials available, the beliefs of the period, and the
circumstances that explain its place in history.

Each result must be a dated source in its own right, and must say how it bears on {subject}.
Sharing a century is not a relationship.

The goal is to produce a large, complete chronological list with rich metadata, not a brief
summary.

For each result:

- Name the specific way it bears on {subject}.
- Classify it as `contemporaneous`, `context`, or `direct_source`.
- Use the earliest defensible attestation date.
- Include every qualifying result found, rather than selecting only the strongest examples.

Before returning the results, search separately for events, people, institutions, technologies and
beliefs; finding one result in a category does not complete that category.

"""
        + _JSON_TAIL,
    },
}


def expand_mode(mode: str) -> dict:
    """The directives for a mode, defaulting to the one the dialog preselects."""
    return EXPAND_MODES.get(mode) or EXPAND_MODES["discovery"]


def expand_body(mode: dict, subject: str) -> str:
    """A mode's body with the node's own title in it.

    `.replace`, not `.format`: the body ends in a JSON template, and every brace in it
    would otherwise have to be doubled by hand — which is exactly the kind of edit that
    silently breaks a prompt the next person pastes in.
    """
    return mode["body"].replace("{subject}", subject or "this subject")


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
{context_line}
ALREADY ON THE TIMELINE. These are shown to the user already, so returning one costs a
call and shows a card that has already been read. Do not return any of these:
{existing_block}

{mode_body}
"""
