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
        valid_ids={"origin", "t4"},
    )
    assert len(out) == 1
    assert out[0].rests_on == ["t4"]
    assert out[0].confidence_label == "high"


def test_support_pointing_at_a_step_that_does_not_exist_is_dropped():
    """A conclusion citing a step that was never emitted reads as supported when it is not."""
    out = orch._build_conclusions(
        [{"statement": "A claim.", "rests_on": ["t9", "origin"]}],
        valid_ids={"origin", "t1"},
    )
    assert out[0].rests_on == ["origin"]


def test_a_conclusion_resting_on_nothing_survives_rather_than_being_dropped():
    """Unsupported is a finding, not a reason to hide it — that is the product's whole job."""
    out = orch._build_conclusions(
        [{"statement": "Widely held, unevidenced here."}], valid_ids={"origin"}
    )
    assert len(out) == 1
    assert out[0].rests_on == []




def test_junk_conclusions_are_skipped():
    out = orch._build_conclusions(
        ["not a dict", {"statement": "   "}, {"no_statement": 1}], valid_ids=set()
    )
    assert out == []



def test_the_citation_list_is_never_silently_trimmed():
    cites = [{"title": f"S{i}", "url": f"https://e.org/{i}"} for i in range(300)]
    assert "further sources gathered" not in orch._format_citations_block(cites)


def test_no_ceiling_survives_on_the_citation_list():
    """A leftover constant would be a second place that quietly drops evidence."""
    assert not hasattr(orch, "MAX_CITATIONS_IN_PROMPT")


def test_a_short_citation_list_is_untouched():
    cites = [{"title": "One", "url": "https://e.org/1"}]
    block = orch._format_citations_block(cites)
    assert "further sources" not in block
    assert "https://e.org/1" in block






def test_an_unconfigured_search_is_reported_not_guessed(monkeypatch, caplog):
    """No backend must be a stated condition, not a quiet empty list."""
    import logging

    from phansora.shared.ai import search as S

    for var in ("BRAVE_API_KEY", "SEARXNG_URL", "CHRONO_SEARCH_PROVIDER"):
        monkeypatch.delenv(var, raising=False)

    cfg = S.SearchConfig.from_env()
    assert cfg.provider == "", "auto-detect invented a backend that cannot run"
    assert S.search_available(cfg) is False

    with caplog.at_level(logging.WARNING):
        assert S.web_search("dead sea scrolls dating", cfg=cfg) == []
    assert any("No web search backend is configured" in r.message for r in caplog.records)


def test_a_configured_backend_reads_as_available(monkeypatch):
    from phansora.shared.ai import search as S

    monkeypatch.delenv("CHRONO_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    cfg = S.SearchConfig.from_env()
    assert cfg.provider == "brave"
    assert S.search_available(cfg) is True


def test_the_gate_is_shared_per_backend():
    """Two callers must meet the same semaphore, or the limit is per-caller and useless."""
    from phansora.shared.ai import search as S

    S._semaphores.clear()
    assert S._gate("brave") is S._gate("brave")
    assert S._gate("brave") is not S._gate("searxng")


def test_the_gate_actually_bounds_parallelism():
    """The limit has to hold across threads — that is the entire point of it."""
    import threading
    import time as _t
    from phansora.shared.ai import search as S

    S._semaphores.clear()
    limit = S._DEFAULT_CONCURRENCY["brave"]
    live = 0
    peak = 0
    lock = threading.Lock()

    def worker():
        nonlocal live, peak
        with S._gate("brave"):
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
    gate = S._gate("brave")
    assert gate.acquire(blocking=False) is True
    assert gate.acquire(blocking=False) is False, "override was not applied"
    gate.release()
    S._semaphores.clear()



def test_the_research_prompt_does_not_call_descent_background_only():
    """That phrase belonged to the deleted `context` node type.

    Left in place it told the research stage to go and find the corpus a subject
    descends from and then treat it as scenery — which is exactly what came back.
    """
    from phansora.products.chrono_origin.pipeline import prompts as P

    assert "BACKGROUND ONLY" not in P.RESEARCH_PROMPT


def test_a_search_can_still_be_disambiguated():
    """"Mercury" is a planet, a god and an element.

    The context box is a real field on the dashboard form, so the term the model is
    handed has to carry it or the trace silently researches the wrong subject.
    """
    from phansora.products.chrono_origin.pipeline import prompts as P

    out = P.RESEARCH_PROMPT.format(title="Mercury", context_clause=" (the planet)")
    assert '"Mercury" (the planet)' in out




def test_independence_stays_a_dossier_field_not_a_gate():
    """It must remain something a step CARRIES, not something a step must PASS."""
    from phansora.products.chrono_origin.pipeline import prompts as P

    assert '"independent_corroboration"' in P.SYNTHESIZE_PROMPT
    assert "independent_corroboration" not in "".join(
        ln for ln in P.SYNTHESIZE_PROMPT.split("\n") if "RULES FOR THE CHAIN" in ln
    )






def test_the_reasoning_budget_starts_where_the_work_lands():
    """8000 meant three generations per answer, two of them discarded."""
    from phansora.shared.ai.deepseek_research import DeepSeekConfig

    assert DeepSeekConfig.reason_max_tokens >= 32000


# ------------------------------------------------------------ expand modes
# Expanding exists to GROW a timeline. Unaimed, an expansion mostly returns the
# anchor's own neighbours — ask around Paul's letters and the gospels come back,
# which are already the next step along.
def test_the_six_modes_are_the_ones_the_dialog_offers():
    from phansora.products.chrono_origin.models import ExpandMode
    from phansora.products.chrono_origin.pipeline.prompts import EXPAND_MODES

    ids = {"discovery", "preservation", "verification", "related", "earlier", "later"}
    assert set(ExpandMode.__args__) == ids
    assert set(EXPAND_MODES) == ids, "model vocabulary and prompt directives have drifted"


def test_every_mode_aims_both_stages():
    """A search can only be pointed at a topic; extraction can be told what to reject."""
    from phansora.products.chrono_origin.pipeline.prompts import EXPAND_MODES

    for name, spec in EXPAND_MODES.items():
        assert spec["label"], name
        assert len(spec["search"]) > 40, f"{name} has no search directive"
        assert len(spec["extract"]) > 40, f"{name} has no extraction directive"


def test_an_unknown_mode_falls_back_rather_than_failing():
    from phansora.products.chrono_origin.pipeline.prompts import expand_mode

    assert expand_mode("nonsense")["label"] == "Related Evidence"
    assert expand_mode(None)["label"] == "Related Evidence"


def test_what_is_already_shown_is_named_for_the_model():
    """A duplicate cannot be avoided by a model that was never shown it."""
    from phansora.products.chrono_origin.pipeline.prompts import format_existing_block

    block = format_existing_block(["The four Gospels", "Tacitus, Annals"])
    assert "The four Gospels" in block and "Tacitus, Annals" in block


def test_an_empty_board_says_so_rather_than_printing_nothing():
    from phansora.products.chrono_origin.pipeline.prompts import format_existing_block

    assert "nothing else on the timeline" in format_existing_block([])
    assert "nothing else on the timeline" in format_existing_block(["  ", None])


def test_the_existing_list_is_bounded():
    """It rides on every expansion call; a long trace should not turn it into a wall."""
    from phansora.products.chrono_origin.pipeline.prompts import format_existing_block

    assert len(format_existing_block([f"Item {i}" for i in range(200)]).split("\n")) == 40


# ------------------------------------------- the expand search must state its query
# A real expansion came back with queries_run == ["jesus christ"] and citations full
# of wallpaper pages and an article about basilisk lizards. The web query is scraped
# out of the prompt by _derive_queries: it reads a "Search query:" line, and
# EXPAND_SEARCH_PROMPT never had one, so it fell through to "first quoted string" —
# the story title. Every expansion ever run searched the bare subject.
def test_the_expand_prompt_states_a_query_the_client_can_find():
    from phansora.products.chrono_origin.pipeline import prompts as P
    from phansora.shared.ai.deepseek_research import _QUERY_LINE

    m = P.expand_mode("discovery")
    out = P.EXPAND_SEARCH_PROMPT.format(
        story_title="Jesus Christ", context_clause="", when="1200-400 BC",
        parent_source_title="Hebrew scriptures", parent_claim="c",
        search_doctrine=P.SEARCH_DOCTRINE, mode_search=m["search"],
        mode_query=m["query"], existing_block=P.format_existing_block([]),
    )
    found = _QUERY_LINE.search(out)
    assert found, "no 'Search query:' line — the search falls back to the story title"
    query = found.group(1).strip()
    # It must be about the ANCHOR, not the subject of the whole trace.
    assert "Hebrew scriptures" in query
    assert query.strip().lower() != "jesus christ"


def test_the_expand_query_is_aimed_by_the_mode():
    from phansora.products.chrono_origin.pipeline import prompts as P
    from phansora.shared.ai.deepseek_research import _QUERY_LINE

    def query_for(mode):
        m = P.expand_mode(mode)
        out = P.EXPAND_SEARCH_PROMPT.format(
            story_title="S", context_clause="", when="", parent_source_title="Anchor",
            parent_claim="c", search_doctrine=P.SEARCH_DOCTRINE, mode_search=m["search"],
            mode_query=m["query"], existing_block=P.format_existing_block([]),
        )
        return _QUERY_LINE.search(out).group(1).strip()

    # Six modes must produce six different searches, or the choice is decorative.
    queries = {q for q in (query_for(m) for m in P.EXPAND_MODES)}
    assert len(queries) == len(P.EXPAND_MODES)


def test_every_mode_carries_search_keywords():
    """The prose directive instructs the summariser; it is useless as a web query."""
    from phansora.products.chrono_origin.pipeline.prompts import EXPAND_MODES

    for name, spec in EXPAND_MODES.items():
        assert spec.get("query"), f"{name} has no web-query keywords"
        assert len(spec["query"].split()) >= 3, name


def test_the_anchor_is_what_the_fallback_angle_picks_up():
    """_derive_queries adds the first quoted string as a second search.

    That used to be the story title, so even the second angle was the bare subject.
    """
    from phansora.products.chrono_origin.pipeline import prompts as P
    from phansora.shared.ai.deepseek_research import _QUOTED

    m = P.expand_mode("related")
    out = P.EXPAND_SEARCH_PROMPT.format(
        story_title="Jesus Christ", context_clause="", when="", parent_source_title="Dead Sea Scrolls",
        parent_claim="c", search_doctrine=P.SEARCH_DOCTRINE, mode_search=m["search"],
        mode_query=m["query"], existing_block=P.format_existing_block([]),
    )
    assert _QUOTED.search(out).group(1).strip() == "Dead Sea Scrolls"


# --------------------------------------------- the executor must not leak workers
# /trace and /expand run their work in a ThreadPoolExecutor and bound it with
# asyncio.wait_for, which cancels the AWAIT and cannot cancel the thread. When the
# budget is shorter than the work, the handler returns 504 and the thread keeps
# running, holding a worker until it finishes. Four of those exhausted the pool and
# the product went silent: requests arriving, no LLM calls made.
@pytest.mark.parametrize("provider", ["deepseek", "gemini"])
def test_the_request_budget_exceeds_the_client_worst_case(provider):
    """Every provider, not just the one that caused this the first time."""
    from phansora.products.chrono_origin.config import get_settings

    if provider == "deepseek":
        from phansora.shared.ai.deepseek_research import DeepSeekConfig as C
    else:
        from phansora.shared.ai.gemini_research import GeminiConfig as C

    budget = get_settings().chrono_request_timeout_s
    # 3 attempts is what tenacity is configured for on both clients.
    worst_case = C.timeout_s * 3
    assert budget > worst_case, (
        f"budget {budget}s is below the {provider} client's {worst_case}s worst case — a "
        "slow call abandons its thread and leaks an executor worker"
    )


def test_the_pool_is_wide_enough_that_one_product_cannot_starve_the_process():
    """The executor is shared by every product in this process, not just Chrono."""
    import re
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src/phansora/products/chrono_origin/server.py"
    m = re.search(r"ThreadPoolExecutor\(max_workers=(\d+)", src.read_text())
    assert m, "executor construction moved; this guard needs updating"
    assert int(m.group(1)) >= 8, "a handful of slow requests should not take the product down"


# ------------------------------------------- dates at the head of the chain
# Both of these drop the OLDEST material specifically, which is the half a chain is
# ordered by and the half a reader cannot tell is missing.
def test_a_span_dated_step_is_not_sorted_to_the_bottom():
    """Corpora composed across centuries are asked for as a span.

    prompts.py tells the model to give exactly that for the works that open a chain —
    "scriptures composed across centuries take the span of their composition". Keying
    the sort on `year` alone sent any step carrying only `year_end` to the END of the
    timeline: the oldest thing in the trace, sorted last.
    """
    assert orch._sort_key(None, -400) < orch._sort_key(50, 60)
    assert orch._sort_key(None, -400) < orch._sort_key(None, 100)
    # A step with neither still goes last, which is correct — it has no position.
    assert orch._sort_key(-400, None) < orch._sort_key(None, None)


def test_a_date_the_model_wrote_as_text_is_still_a_date():
    """The gate ran on the raw JSON, before pydantic would have coerced anything.

    A step dated "-400" or -400.0 was demoted into `conclusions` and told it was "not
    a surviving object" — untrue, and not the reason.
    """
    assert orch._as_year(-400) == -400
    assert orch._as_year("-400") == -400
    assert orch._as_year(-400.0) == -400
    assert orch._as_year("c. 400 BC") == 400  # a number is there; the sign is the model's job
    assert orch._as_year(None) is None
    assert orch._as_year("") is None
    assert orch._as_year("unknown") is None


def test_a_boolean_is_not_read_as_a_year():
    """bool subclasses int, so a flag would otherwise date a step to year 1."""
    assert orch._as_year(True) is None
    assert orch._as_year(False) is None



def test_a_chain_that_starts_at_a_copy_says_so(caplog):
    """The rule is in the prompt and nothing checked it.

    A trace of Jesus opened at the Dead Sea Scrolls: the copies were there, the
    scriptures they are copies OF were not. Every step was real, so the trace looked
    complete and the reader had no way to see the oldest half was missing.
    """
    import logging

    from phansora.products.chrono_origin.models import OriginResult, TimelineEvent

    origin = OriginResult(id="origin", year=-250, node_type="scroll",
                          source_title="The Dead Sea Scrolls", summary="s")
    later = [TimelineEvent(id="t1", year=50, node_type="letter",
                           source_title="Paul's letters", claim="c")]
    with caplog.at_level(logging.WARNING):
        orch._warn_if_copy_without_work(origin, later)
    assert any("starts at a copy" in r.message for r in caplog.records)


def test_a_chain_whose_copy_has_an_older_step_is_quiet(caplog):
    import logging

    from phansora.products.chrono_origin.models import OriginResult, TimelineEvent

    origin = OriginResult(id="origin", year=-250, node_type="scroll",
                          source_title="The Dead Sea Scrolls", summary="s")
    with_work = [TimelineEvent(id="t1", year=-400, node_type="text",
                               source_title="Hebrew scriptures", claim="c")]
    with caplog.at_level(logging.WARNING):
        orch._warn_if_copy_without_work(origin, with_work)
    assert not [r for r in caplog.records if "starts at a copy" in r.message]


# ------------------------------------------------ the prompts name no subject
def test_no_prompt_is_written_around_one_subject():
    """Chrono Origin traces anything. Its prompts must not teach from one case.

    SYNTHESIZE_PROMPT carried a fully worked example of a first-century religious
    chain — Hebrew scriptures, Septuagint, Dead Sea Scrolls, Paul, the Gospels,
    Josephus, Tacitus — introduced as "the shape wanted. The chain is exactly this."
    That text went out on EVERY trace, so a trace of a transistor, an aircraft, a
    patent or a company was handed a biblical chain as its template of a good answer,
    along with worked examples about censuses and nativities.

    The shape it taught was real and is still taught — as ROLES, which any subject can
    fill. What is gone is the assumption that every subject looks like that one.
    """
    import re

    from phansora.products.chrono_origin.pipeline import prompts as P

    # Proper nouns and terms belonging to ONE subject. Deliberately not generic
    # object kinds: "scroll", "papyrus", "codex" and "manuscript" are categories of
    # physical evidence that any subject may have, and a prompt listing them is
    # describing the world, not a religion.
    subject_terms = re.compile(
        r"jesus|christ|septuagint|dead sea|qumran|josephus|tacitus|gospel|pauline|"
        r"paul's|hebrew scripture|messian|new testament|crucif|nativity|testimonium|"
        r"epistle|bedouin",
        re.I,
    )
    offenders = []
    for name in dir(P):
        if not name.isupper():
            continue
        value = getattr(P, name)
        if not isinstance(value, str):
            continue
        for i, line in enumerate(value.splitlines(), 1):
            if subject_terms.search(line):
                offenders.append(f"{name}:{i}: {line.strip()[:70]}")
    assert not offenders, "a subject leaked into the prompts:\n  " + "\n  ".join(offenders)



def test_a_research_answer_keeps_every_source_end_to_end(monkeypatch, tmp_path):
    """Forty sources found, forty on the trace.

    The gather stage sliced this list to the first eight in ARRIVAL order, before any
    tiering ran, so two thirds of what a search found was discarded unread and which
    third survived was luck. A live trace kept the Israel Museum's Dead Sea Scrolls
    collection only because it happened to arrive seventh.
    """
    from phansora.products.chrono_origin.models import TraceRequest
    from phansora.shared.ai.research import GroundedAnswer

    class Client:
        def grounded_search(self, prompt):
            return GroundedAnswer(
                text="research findings",
                citations=[{"url": f"https://src{i}.example/p", "title": f"S{i}"}
                           for i in range(40)],
                queries=["q"],
            )

        def reason_json(self, prompt, use_reasoning_model=False):
            self.synth = prompt
            return {"origin": {"year": -400, "source_title": "O", "summary": "s",
                               "citations": [], "confidence": 0.6,
                               "evidence": {"claim": "c"}},
                    "timeline": [], "connections": [], "reasoning": "r", "confidence": 0.6}

    monkeypatch.setattr(orch, "get_cached", lambda *a, **k: None)
    monkeypatch.setattr(orch, "save_cached", lambda *a, **k: None)
    monkeypatch.setattr(orch, "read_best", lambda *a, **k: [])

    client = Client()
    o = orch.TraceOrchestrator(client=client)
    result = o.run(TraceRequest(title="Jesus Christ"))

    assert len(result.citations) == 40, "sources were dropped between search and response"
    # and every one of them is offered to synthesis, since that is where citing happens
    for i in (0, 8, 39):
        assert f"https://src{i}.example/p" in client.synth, f"source {i} never reached synthesis"
