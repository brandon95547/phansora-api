# What happens when you start a Chrono Origin trace

Every number here is read from the code as it runs today, not from an older design
note. Where a figure is configurable the setting is named, so it can be checked
rather than trusted.

The short version: **2 model calls, 12 page fetches, one JSON generation.** One
grounded research call in which the model runs its own web searches, then one call
that turns the corpus into a timeline.

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

## Stage 1 — Gather (one grounded call)

**One call.** The model is handed `RESEARCH_PROMPT` and runs as many web searches as
it judges the question needs; the queries it actually ran come back in
`webSearchQueries` and are what the UI reports, rather than a template we wrote.

The prompt asks for two parts, in this order:

| Part | Asks for |
|---|---|
| **1 — what this descends from** | the corpus its texts quote, the tradition they are composed inside, the text they translate, and the surviving objects carrying those today |
| **2 — what survives about the subject** | earliest texts, manuscripts, external sources, documents and records, inscriptions and artifacts, excavated sites |

Part 1 leads, and that ordering is the whole point.

> **Removed: the seven-way fan-out.** Until now this stage fired seven fixed queries,
> one per evidence "strand". That existed because the old provider could not search —
> queries had to be guessed in advance and fired blind to guarantee coverage. It also
> decided where every chain STARTED, and decided it wrongly: **six of the seven asked
> what survives ABOUT the subject and exactly one asked what its evidence DESCENDS
> FROM.** The chain rule says a chain begins with descent, so the deciding half was
> outvoted six to one in every corpus. A trace of Jesus opened at the Dead Sea Scrolls
> and lost the four centuries of scripture the Scrolls are copies *of* — while the
> prompt had named that exact failure by name the whole time.

**Citations come only from grounding metadata**, never from URLs in the model's prose.

**Grounding proxy URLs are resolved.** The API returns
`vertexaisearch.cloud.google.com/grounding-api-redirect/...` for every source, with the
real domain only in the chunk title. Left unresolved that is quietly destructive: every
citation reports the same host, so the five-tier source policy scored a forum post and
a university library identically, `low_authority` pages were never skipped, page-read
ranking was arbitrary, and per-domain diversification saw one domain. Nothing raised.
Each proxy is now followed to the page it points at; when one will not resolve, the
domain from the title is used so the source is still tierable.

**An answer with no grounding at all is discarded** — neither chunks nor queries means
the model answered from memory, and memory is not evidence.

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

It receives: the corpus, up to **60 citations**, the read pages, the five-tier source
doctrine, and the chain rules.

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
  no position in one. Demoted the same way — but read leniently, so `"-400"` or
  `-400.0` still counts as a date, and told in `conclusions` that it lacked a date
  rather than that it "is not a surviving object", which was the wrong reason.
- **Chronological sort**, oldest first, ids assigned before sorting so connections
  survive the reordering. Sorted on the START of a span, falling back to its end — a
  corpus given only a `year_end` used to sort to the *bottom*, which is where the
  oldest material in a trace was landing.
- **A chain starting at a copy is flagged.** If the first step is a manuscript or
  scroll and nothing predates it, that is logged: the work it copies should be an
  earlier step. Warned, never invented — the earlier step has to come from research.
- **If the origin arrives undated** it swaps places with the first dated step, ids
  included, so `origin` keeps naming the head of the chain.
- **Citations are ranked per claim, not per URL** — a 1963 newspaper is primary
  evidence for 1963 and commentary on everything else.
- **Connections are validated** against the ids that actually exist.

---

## The counts

| | Count | Setting |
|---|---|---|
| Model calls | **2** (1 grounded research + 1 extraction) | — |
| Web searches | however many the model runs inside that one call | its own choice |
| Citations kept per answer | 8 | `chrono_max_sources_per_stage` |
| Citations into the prompt | ≤60 | `MAX_CITATIONS_IN_PROMPT` |
| Pages fetched / read | 12 / 6 | `chrono_read_sources` |
| Chain steps | ≤14 | prompt |
| Conclusions / connections | ≤8 / ≤8 | prompt / `chrono_max_connections` |

## Providers

Chosen by **`CHRONO_LLM_PROVIDER`**; the model by `CHRONO_MODEL`, falling back to the
provider-wide variable. There is no built-in default model name — a hardcoded one
silently breaks when the provider retires it.

| Provider | Searching | Needs |
|---|---|---|
| **`gemini`** *(default)* | native Google Search grounding | `GEMINI_API_KEY`, `GEMINI_MODEL` |
| `openai` | native `web_search` tool | `OPENAI_API_KEY`, `OPENAI_MODEL` |
| `deepseek` | **none of its own** — external search required | `DEEPSEEK_API_KEY` + `BRAVE_API_KEY` or `SEARXNG_URL` |

DeepSeek's API rejects anything but `type: "function"` in its tools array, so there
is no hosted search tool to enable — verified against the live API, not inferred.

**On cost.** Grounding is billed per grounded *request*, not per token, with a free
monthly allowance on the 3.x family. That is why `gemini-3.5-flash-lite` is the
default despite a higher token price than 2.5: at this product's volume the free
grounding allowance dominates the bill.

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
being paid for in the most expensive currency the pipeline has.

The searching then went the same way, for a different reason. Fixed templates were a
workaround for a model that could not search; once the model chose its own queries the
templates stopped adding coverage and started removing it, because their *proportions*
silently set where every chain began.

**The known trades.** One pass over gathered prose notices less than four passes did;
the first version of this design lost Paul and the Gospels entirely, and was fixed by
giving synthesis more to read rather than by restoring the loop. And one research call
returns less raw text than seven did — if a chain comes back thin rather than early,
the fix is asking for more angles *within* that call, not restoring the fan-out.

## What to watch when a trace looks wrong

- **Chain starts too late** — the commonest failure, and the log now says so: look for
  `Chain starts at a copy`. It means the research found the copies and not the work
  they copy. The rule is descent, not aboutness.
- **Citations all on one host** — grounding proxies are not being resolved; every
  source will be tiered `unknown` and the source policy is inert. Look for
  `could not be resolved past the redirect`.
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
- **`Gemini answered without searching`** in the logs — grounding did not run. Check
  the key's quota rather than the prompts; the trace has no evidence to work from.
- **`No web search backend is configured`** — only reachable on the `deepseek`
  provider, and it means every search was skipped. Set `BRAVE_API_KEY` or switch to
  `gemini`.
