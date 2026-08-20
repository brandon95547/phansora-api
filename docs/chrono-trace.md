# What happens when you start a Chrono Origin trace

Every number here is read from the code as it runs today, not from an older design
note. Where a figure is configurable the setting is named, so it can be checked
rather than trusted.

The short version: **8 model calls, ~14 web searches, 12 page fetches, one JSON
generation.** Typical wall time on the production box is 2½–5 minutes.

---

## The rule the whole thing serves

A trace is a **chain of surviving evidence**, in the order it was produced. Every
step is a thing that still exists and can be examined — a text, a manuscript, a
scroll, a letter, an inscription, a document, a record, an object, an excavated
find. At each step it answers one question and only that one:

> What is the next surviving piece of evidence?

Anything that is not a surviving object is a *reading of* the evidence. Those are
kept, but they go in `conclusions`, stated after the chain rather than inside it.
A reading placed among artifacts inherits an authority it has not earned.

Two ways in, and only two: a step is **about the subject**, or something later
**descends from** it — quotes it, translates it, continues it, physically
witnesses it.

---

## Stage 0 — Cache

Key is `v5` + normalised title + context. A hit returns instantly and costs
nothing; the version prefix means a change to the trace's *shape* retires old
entries rather than serving them under rules they were never generated under.

Note: the key does **not** hash the prompt text. Changing a rule and re-tracing the
same subject will serve the cached answer until it is invalidated:

```bash
curl -X POST .../chrono/cache/invalidate -d '{"title":"Jesus Christ"}'
```

## Stage 1 — Gather (one round, no planner)

**Seven searches, fired at once.** One per evidence strand, each a fixed template
filled with the subject:

| Strand | Looks for |
|---|---|
| `precursor_evidence` | what survives from *before* the subject that its sources descend from |
| `earliest_texts` | earliest surviving texts about or by the subject, with composition dates |
| `manuscripts` | physical copies: shelfmark, repository, palaeographic date |
| `external_sources` | surviving texts from outside the tradition that mention it |
| `documents_records` | decrees, censuses, court, tax and official registers |
| `inscriptions_artifacts` | inscriptions, coins, seals, ostraca |
| `archaeology` | excavated sites and assemblages, as published in excavation reports |

There is **no planning call**. The queries are fixed text, so they cost nothing to
produce. There is also **no round loop** — see [Why it works this way](#why-it-works-this-way).

Each of the seven then runs one `grounded_search`, which is:

1. **Derive up to 2 web queries** — the strand query, plus the subject name as a
   second angle. No model call; they are read out of the prompt.
2. **Run them against DuckDuckGo**, concurrently. Up to **10 results each**
   (`CHRONO_SEARCH_RESULTS`), capped at **2 per domain** so five pages of one
   publisher can't crowd out corroboration. Retries three times with backoff.
3. **One model call** summarises those results into grounded prose that cites the
   exact URLs.

So the gather stage is **7 model calls and ~14 web searches**, returning up to ~140
raw results before de-duplication. Up to **8 citations** are kept per answer
(`chrono_max_sources_per_stage`).

Searches are gated to **4 in flight** against DuckDuckGo (`CHRONO_SEARCH_CONCURRENCY`).
That gate is deliberate: the keyless backend signals throttling by returning an
*empty result set*, which is indistinguishable from a genuinely empty search, so too
much concurrency shows up as a thinner trace rather than an error.

## Stage 2 — The corpus

The gathered text is assembled as prose, per query:

```
[the query that found this]
<the model's grounded summary>
- <result title>: <up to 400 chars of snippet>
- <result title>: <up to 400 chars of snippet>
```

Both halves matter. A summariser writing a few hundred words about ten results
necessarily discards most of what they said — and the discarded part is the dates,
shelfmarks and repositories the extraction is hunting for. The snippets are already
fetched and already paid for.

## Stage 3 — Read the strongest sources properly

Search snippets are leads. Before judging anything, the pipeline opens the best
pages and reads them.

- **12 candidates** ranked by source tier — primary evidence and repositories
  first, `low_authority` never read at all, one page per registrable domain.
- Fetched **concurrently**, 15s timeout each, 3 MB cap.
- Keeps the first **6 successes** (`chrono_read_sources`) plus up to **2 failures**.

Failures are kept on purpose. "The British Museum record could not be opened" is a
real gap in the evidence, and the extraction should see it rather than infer silence.
Each page contributes up to **6000 characters** to the prompt.

## Stage 4 — One extraction

**A single model call.** This is the only place the pipeline generates JSON.

It receives: the corpus, up to **60 citations**, the read pages, which strands were
researched, the five-tier source doctrine, and the chain rules.

It returns: the origin, the timeline (**≤14 steps**), `conclusions` (**≤8**),
`connections` (**≤8**), and a reasoning paragraph. Every step carries an evidence
dossier — earliest supporting source, earliest surviving copy, provenance,
contemporary evidence, independent corroboration, what contradicts it, scholarly
dispute, confidence, and the single missing piece that most limits it.

Output budget starts at **32000 tokens**, ceiling **64000**. It starts high on
purpose: at 8000 this call generated, was cut off, regenerated at 16000, was cut off
again, and only succeeded on the third attempt — three generations for one answer.
The budget is a cap, not a charge.

## Stage 5 — What the code decides, not the model

The model proposes; these are enforced regardless of what it returns.

- **Only the nine evidence kinds may be steps.** Anything else is demoted into
  `conclusions` — never dropped, because a silent drop hides a claim the model made.
- **An undated step is not a step.** A chain is an order; an object with no date has
  no position in one. Demoted the same way.
- **Chronological sort**, oldest first, ids assigned before sorting so connections
  survive the reordering.
- **If the origin arrives undated** it swaps places with the first dated step, ids
  included, so `origin` keeps naming the head of the chain.
- **Citations are ranked per claim, not per URL** — a 1963 newspaper is primary
  evidence for 1963 and commentary on everything else.
- **Connections are validated** against the ids that actually exist.

---

## The counts

| | Count | Setting |
|---|---|---|
| Model calls | **8** (7 summaries + 1 extraction) | — |
| Web searches | **~14** | 7 strands × up to 2 derived |
| Results per search | 10 | `CHRONO_SEARCH_RESULTS` |
| Concurrent searches | 4 | `CHRONO_SEARCH_CONCURRENCY` |
| Citations kept per answer | 8 | `chrono_max_sources_per_stage` |
| Citations into the prompt | ≤60 | `MAX_CITATIONS_IN_PROMPT` |
| Pages fetched / read | 12 / 6 | `chrono_read_sources` |
| Chain steps | ≤14 | prompt |
| Conclusions / connections | ≤8 / ≤8 | prompt / `chrono_max_connections` |

Provider is chosen by **`CHRONO_LLM_PROVIDER`** (`deepseek` or `openai`); the model
by `CHRONO_MODEL`, falling back to `DEEPSEEK_MODEL` / `OPENAI_MODEL`. There is no
built-in default model name — a hardcoded one silently breaks when the provider
retires it.

## Why it works this way

The pipeline used to plan its searches with a model call, then run **three rounds**
of six searches, turning each round's results into structured JSON before deciding
what to search next, then chase citations backward, then synthesise. That was ~27
model calls and four or five separate JSON generations.

Measured on the production box: six searches finished in **14 seconds**, and the
call that turned them into JSON took **104**. Every round overran its output budget
and regenerated from scratch; synthesis then overran the doubled budget and the
trace **failed outright at twenty minutes**.

Generating JSON is the expensive act, so it now happens exactly once. The adaptivity
the round loop bought — deciding what to look for next based on what came back — was
being paid for in the most expensive currency the pipeline has, and the queries a
subject needs turn out to be the strand templates, which are free.

**The known trade:** one pass over gathered prose notices less than four passes did.
The first version of this design lost Paul and the Gospels entirely, and was fixed by
giving the extraction more to read (snippets, more results per search) rather than by
restoring the loop.

## What to watch when a trace looks wrong

- **Chain starts too late** — the precursor strand found the corpus and the
  extraction judged it "not about the subject". The rule is descent, not aboutness.
- **A step with no date** — should now be impossible; if one appears, the demotion in
  Stage 5 failed.
- **Two works in one step** ("New Testament writings, 50–100") — different composition
  dates and different evidence belong in different steps.
- **A real find nothing descends from** — a later church, an object from the right
  region and the wrong century. Real evidence, wrong chain.
- **`truncated at N tokens; retrying`** in the logs — the extraction is overrunning
  again; the budget needs raising, not the output trimming.
- **A trace started within ~3 minutes of a service restart** competes with vLLM and
  CosyVoice loading onto the GPU in the same process, and will look far slower than
  it is.
