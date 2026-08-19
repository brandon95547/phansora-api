"""The chain rule: a trace follows surviving evidence and nothing else.

Every step must be a thing that still exists and can be examined — a text, a
manuscript, a scroll, a letter, an inscription, a document, a record, an object, an
excavated find. Anything else is a reading OF the evidence and belongs in
``conclusions``, stated after the chain rather than inside it.

The prompt asks for that. These tests pin that the CODE guarantees it, because the
failure mode is a model under pressure to tell a coherent story reaching for the
connective tissue between documents — an expectation, a movement, a development — and
those arrive looking exactly like findings.
"""
from __future__ import annotations

import pytest

pytest.importorskip("pydantic", reason="pydantic not installed on this host")
pytest.importorskip("pydantic_settings", reason="pydantic-settings not installed on this host")

from phansora.products.chrono_origin.models import EvidenceKind  # noqa: E402
from phansora.products.chrono_origin.pipeline import orchestrator as orch  # noqa: E402

# The worked example from the product brief: the Jesus chain, in order, as it should
# come out — seven surviving things, nothing between them.
JESUS_CHAIN = [
    ("Hebrew scriptures", "text", -1200, -400),
    ("Septuagint", "text", -250, -100),
    ("Dead Sea Scrolls", "scroll", -250, 70),
    ("Authentic letters of Paul", "letter", 50, 65),
    ("The four Gospels", "text", 65, 100),
    ("Josephus, Antiquities", "text", 93, 94),
    ("Tacitus, Annals", "text", 116, 116),
]


def test_every_kind_in_the_reference_chain_is_evidence():
    for title, kind, _a, _b in JESUS_CHAIN:
        assert orch.is_evidence_kind(kind), f"{title} typed as {kind} is not accepted as evidence"


@pytest.mark.parametrize(
    "kind",
    [
        # The interpretive vocabulary the old model allowed as chain steps. Each of
        # these is a conclusion; none is a thing anyone can go and look at.
        "context",
        "event",
        "reconstructed_date",
        "institutional_development",
        "term_history",
        "linguistic_transmission",
        "dating_framework",
        "external_attestation",  # renamed: an outside source is just a "text"
        "messianic_expectation",  # the brief's own example of what must never be a step
        "",
        None,
    ],
)
def test_non_evidence_is_refused_entry_to_the_chain(kind):
    assert not orch.is_evidence_kind(kind)


def test_the_gate_does_not_coerce():
    """An unrecognised kind must not become a valid one.

    The previous implementation mapped anything unknown onto a real type, so an
    interpretation arrived wearing a respectable label and was then indistinguishable
    from an artefact. Refusing is the entire point.
    """
    assert not hasattr(orch, "_node_type"), "the coercing helper must not come back"


def test_evidence_kinds_match_the_declared_model():
    declared = set(EvidenceKind.__args__)
    assert declared == orch._EVIDENCE_KINDS, "model vocabulary and runtime gate have drifted"


# ---------------------------------------------------------------- conclusions
def test_model_conclusions_are_kept_with_their_support():
    out = orch._build_conclusions(
        [
            {
                "statement": "Paul's letters show a resurrection claim circulating within decades.",
                "rests_on": ["t4"],
                "confidence_label": "high",
                "reasoning": "The letters are dated 50-65 and state the claim directly.",
                "dissent": "None identified",
            }
        ],
        [],
        valid_ids={"origin", "t4"},
    )
    assert len(out) == 1
    assert out[0].rests_on == ["t4"]
    assert out[0].confidence_label == "high"


def test_support_pointing_at_a_step_that_does_not_exist_is_dropped():
    """A conclusion citing a step that was never emitted reads as supported when it is not."""
    out = orch._build_conclusions(
        [{"statement": "A claim.", "rests_on": ["t9", "origin"]}],
        [],
        valid_ids={"origin", "t1"},
    )
    assert out[0].rests_on == ["origin"]


def test_a_conclusion_resting_on_nothing_survives_rather_than_being_dropped():
    """Unsupported is a finding, not a reason to hide it — that is the product's whole job."""
    out = orch._build_conclusions(
        [{"statement": "Widely held, unevidenced here."}], [], valid_ids={"origin"}
    )
    assert len(out) == 1
    assert out[0].rests_on == []


def test_a_demoted_step_becomes_a_conclusion_not_a_deletion():
    """The brief's example: an interpretation offered as a step.

    It must survive the demotion — silently dropping it would hide a claim the model
    actually made — and it must land marked as resting on nothing.
    """
    out = orch._build_conclusions(
        None,
        [
            {
                "node_type": "context",
                "claim": "Messianic expectation was widespread in first-century Judaea.",
                "source_title": "Messianic expectation",
            }
        ],
        valid_ids={"origin"},
    )
    assert len(out) == 1
    assert "Messianic expectation was widespread" in out[0].statement
    assert out[0].rests_on == []
    assert out[0].confidence_label == "speculative"
    assert "not a surviving object" in out[0].reasoning


def test_demoted_steps_follow_the_models_own_conclusions():
    out = orch._build_conclusions(
        [{"statement": "A stated conclusion."}],
        [{"node_type": "event", "claim": "An inferred happening."}],
        valid_ids={"origin"},
    )
    assert [c.statement for c in out] == ["A stated conclusion.", "An inferred happening."]


def test_junk_conclusions_are_skipped():
    out = orch._build_conclusions(
        ["not a dict", {"statement": "   "}, {"no_statement": 1}], [], valid_ids=set()
    )
    assert out == []


def test_a_demoted_step_with_no_words_is_skipped():
    out = orch._build_conclusions(None, [{"node_type": "context"}], valid_ids=set())
    assert out == []


# ------------------------------------------------------------------- strands
def test_research_strands_hunt_for_surviving_things():
    """Every strand must be answerable by an object, or the research plans for prose."""
    assert orch._STRANDS == {
        "precursor_evidence",
        "earliest_texts",
        "manuscripts",
        "external_sources",
        "documents_records",
        "inscriptions_artifacts",
        "archaeology",
    }


def test_every_evidence_kind_reports_coverage_of_some_strand():
    """A kind with no strand is a kind the loop can never mark as covered."""
    for kind in orch._EVIDENCE_KINDS:
        assert kind in orch._NODE_TYPE_STRAND, f"{kind} maps to no strand"
        assert orch._NODE_TYPE_STRAND[kind] in orch._STRANDS


def test_every_strand_has_a_query():
    for strand in orch._STRANDS:
        assert strand in orch._STRAND_QUERIES
        assert "{title}" in orch._STRAND_QUERIES[strand]


# ------------------------------------------------- prompt size is bounded
# The synthesis call is the only reasoning-model call in the pipeline and carries
# 16-22k input tokens. Both blocks feeding it used to be unbounded, so a long trace
# paid for its own thoroughness twice: once to gather, once to re-read.
def test_the_citation_list_handed_to_synthesis_is_capped():
    cites = [{"title": f"S{i}", "url": f"https://e.org/{i}"} for i in range(300)]
    block = orch._format_citations_block(cites)
    listed = [ln for ln in block.split("\n") if ln.startswith("[")]
    assert len(listed) == orch.MAX_CITATIONS_IN_PROMPT


def test_a_trimmed_citation_list_says_it_was_trimmed():
    """Synthesis is told to cite from this list; it should know the list is partial."""
    cites = [{"title": f"S{i}", "url": f"https://e.org/{i}"} for i in range(300)]
    assert "further sources gathered" in orch._format_citations_block(cites)


def test_a_short_citation_list_is_untouched():
    cites = [{"title": "One", "url": "https://e.org/1"}]
    block = orch._format_citations_block(cites)
    assert "further sources" not in block
    assert "https://e.org/1" in block


def test_mentions_are_capped_too():
    mentions = [{"node_type": "text", "source_title": f"T{i}", "claim": "c"} for i in range(400)]
    lines = [ln for ln in orch._format_mentions_block(mentions).split("\n") if ln.startswith("- ")]
    assert len(lines) == orch.MAX_MENTIONS_IN_PROMPT


def test_evidence_survives_the_cut_before_interpretation_does():
    """The cut must not drop a manuscript to make room for an inferred development.

    Only surviving objects can become chain steps, so when the block is trimmed the
    mentions still eligible to be steps are the ones that have to survive.
    """
    filler = [
        {"node_type": "context", "source_title": f"Interpretation {i}", "claim": "c"}
        for i in range(orch.MAX_MENTIONS_IN_PROMPT + 50)
    ]
    real = {"node_type": "manuscript", "source_title": "Codex Sinaiticus", "claim": "c"}
    # Arrives last, and would fall off the end of an arrival-ordered trim.
    block = orch._format_mentions_block(filler + [real])
    assert "Codex Sinaiticus" in block


def test_the_extractors_is_evidence_flag_also_counts_as_evidence():
    filler = [
        {"node_type": "context", "source_title": f"X{i}", "claim": "c"}
        for i in range(orch.MAX_MENTIONS_IN_PROMPT + 10)
    ]
    flagged = {"node_type": "unrecognised", "is_evidence": True,
               "source_title": "An excavation report", "claim": "c"}
    assert "An excavation report" in orch._format_mentions_block(filler + [flagged])


def test_mentions_keep_their_research_order_within_a_group():
    """Stable sort: prioritising evidence must not shuffle the rounds' own ordering."""
    mentions = [
        {"node_type": "text", "source_title": "First", "claim": "c"},
        {"node_type": "text", "source_title": "Second", "claim": "c"},
        {"node_type": "text", "source_title": "Third", "claim": "c"},
    ]
    block = orch._format_mentions_block(mentions)
    assert block.index("First") < block.index("Second") < block.index("Third")


# ------------------------------------------------ search concurrency gate
# The keyless DuckDuckGo backend signals throttling by returning an EMPTY result set,
# which web_search cannot tell from a genuinely empty search and therefore retries with
# backoff. So too much concurrency does not surface as an error — it surfaces as a
# slower trace built on fewer sources.
def test_the_keyless_backend_is_gated_harder_than_the_paid_ones():
    from phansora.shared.ai import search as S

    assert S._DEFAULT_CONCURRENCY["duckduckgo"] < S._DEFAULT_CONCURRENCY["brave"]
    assert S._DEFAULT_CONCURRENCY["duckduckgo"] < S._DEFAULT_CONCURRENCY["searxng"]


def test_the_gate_is_shared_per_backend():
    """Two callers must meet the same semaphore, or the limit is per-caller and useless."""
    from phansora.shared.ai import search as S

    S._semaphores.clear()
    assert S._gate("duckduckgo") is S._gate("duckduckgo")
    assert S._gate("duckduckgo") is not S._gate("brave")


def test_the_gate_actually_bounds_parallelism():
    """The limit has to hold across threads — that is the entire point of it."""
    import threading
    import time as _t
    from phansora.shared.ai import search as S

    S._semaphores.clear()
    limit = S._DEFAULT_CONCURRENCY["duckduckgo"]
    live = 0
    peak = 0
    lock = threading.Lock()

    def worker():
        nonlocal live, peak
        with S._gate("duckduckgo"):
            with lock:
                live += 1
                peak = max(peak, live)
            _t.sleep(0.02)
            with lock:
                live -= 1

    threads = [threading.Thread(target=worker) for _ in range(limit * 3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert peak <= limit, f"{peak} searches ran at once against a limit of {limit}"


def test_the_limit_is_overridable_for_a_backend_that_can_take_it(monkeypatch):
    from phansora.shared.ai import search as S

    S._semaphores.clear()
    monkeypatch.setenv("CHRONO_SEARCH_CONCURRENCY", "1")
    gate = S._gate("duckduckgo")
    assert gate.acquire(blocking=False) is True
    assert gate.acquire(blocking=False) is False, "override was not applied"
    gate.release()
    S._semaphores.clear()


# ------------------------------------------------- where the chain starts
# The reported failure: a Jesus trace opened at 30 CE and contained none of the
# Hebrew scriptures, Septuagint or Dead Sea Scrolls. The rounds do find that
# material; synthesis judged it "not about the subject" and left it out, and the
# result looked well-formed. Nothing errored. These pin the detector that makes it
# visible, since a prompt can only ask.
def test_older_evidence_left_out_of_the_chain_is_detected():
    mentions = [
        {"node_type": "text", "source_title": "Hebrew scriptures", "year": -400},
        {"node_type": "text", "source_title": "Septuagint", "year": -250},
        {"node_type": "letter", "source_title": "Paul", "year": 50},
    ]
    dropped = orch._dropped_precursors(mentions, chain_start_year=30)
    assert [m["source_title"] for m in dropped] == ["Septuagint", "Hebrew scriptures"] or \
           [m["source_title"] for m in dropped] == ["Hebrew scriptures", "Septuagint"]
    assert len(dropped) == 2


def test_the_detector_reports_oldest_first():
    mentions = [
        {"node_type": "text", "source_title": "Later", "year": -100},
        {"node_type": "text", "source_title": "Older", "year": -900},
    ]
    assert [m["source_title"] for m in orch._dropped_precursors(mentions, 30)] == ["Older", "Later"]


def test_a_chain_that_starts_where_the_evidence_starts_reports_nothing():
    mentions = [
        {"node_type": "text", "source_title": "Hebrew scriptures", "year": -400},
        {"node_type": "letter", "source_title": "Paul", "year": 50},
    ]
    assert orch._dropped_precursors(mentions, chain_start_year=-400) == []


def test_only_evidence_can_be_reported_as_wrongly_dropped():
    """A mention that could never have been a step was not dropped from the chain."""
    mentions = [
        {"node_type": "context", "source_title": "Messianic expectation", "year": -200},
        {"node_type": "event", "source_title": "An inferred happening", "year": -300},
    ]
    assert orch._dropped_precursors(mentions, chain_start_year=30) == []


def test_the_extractors_flag_also_counts_for_the_detector():
    mentions = [{"is_evidence": True, "node_type": "unrecognised",
                 "source_title": "An excavation report", "year": -500}]
    assert len(orch._dropped_precursors(mentions, 30)) == 1


def test_an_undated_chain_start_cannot_be_judged():
    """No first-step year means no comparison — reporting everything would be noise."""
    mentions = [{"node_type": "text", "source_title": "Anything", "year": -400}]
    assert orch._dropped_precursors(mentions, chain_start_year=None) == []


def test_the_report_is_capped():
    mentions = [
        {"node_type": "text", "source_title": f"T{i}", "year": -1000 + i} for i in range(40)
    ]
    assert len(orch._dropped_precursors(mentions, chain_start_year=30)) == 5


# ------------------------------------------------- the prompt states the rule
def test_the_prompt_says_where_a_chain_starts():
    from phansora.products.chrono_origin.pipeline import prompts as P

    s = P.SYNTHESIZE_PROMPT
    assert "WHERE THE CHAIN STARTS" in s
    # The distinction that failed: descent, not aboutness.
    assert "does NOT start at the earliest evidence ABOUT the subject" in s
    assert "DOCUMENTARY, not thematic" in s


def test_the_planner_no_longer_calls_precursor_evidence_background_only():
    """That phrase belonged to the deleted `context` node type.

    Left in place it told the planner to research the corpus a subject descends from
    and then treat it as scenery — which is exactly what came back.
    """
    from phansora.products.chrono_origin.pipeline import prompts as P

    assert "BACKGROUND ONLY" not in P.DECOMPOSE_PROMPT
