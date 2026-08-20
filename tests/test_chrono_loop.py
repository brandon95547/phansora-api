"""The trace pipeline, driven end to end against a scripted model.

This file used to test a research LOOP: plan strands, search, extract, decide
whether to go round again. That loop is gone, and the reason it went is the thing
worth pinning now. Measured on a live trace, a round's six searches finished in 14
seconds and the call that turned them into structured JSON took 104 — every round,
every time, because it overran its output budget and regenerated from scratch.
Synthesis then overran the doubled budget and the whole trace failed at 20 minutes.
Generating JSON was the expensive act, and the loop was buying it once per round.

So the pipeline is one pass: one fixed search per strand, fired together, then a
single JSON call that turns the whole corpus into a timeline. The searches are
template text and cost nothing to produce.

What these tests hold in place is that shape. The fake model is deliberately
unhelpful — it never proposes a query of its own — because everything the pipeline
covers has to come from the pipeline, not from the model volunteering.
"""
from __future__ import annotations

import pytest

from phansora.products.chrono_origin.models import TraceRequest
from phansora.products.chrono_origin.pipeline import orchestrator as orch
from phansora.shared.ai.research import GroundedAnswer


class ScriptedClient:
    """A model that searches when asked and synthesizes once, and counts both."""

    def __init__(self):
        self.searches = []
        self.json_calls = []
        self.synthesize_prompt = None

    def reason_json(self, prompt, use_reasoning_model=False):
        self.json_calls.append(prompt)
        self.synthesize_prompt = prompt
        return {
            "origin": {
                "year": 100, "source_title": "O", "summary": "s",
                "citations": ["https://www.jstor.org/stable/1"], "confidence": 0.6,
                "evidence": {"claim": "c"},
            },
            "timeline": [],
            "connections": [],
            "reasoning": "r",
            "confidence": 0.6,
        }

    def grounded_search(self, prompt):
        query = prompt.split("Search query: ", 1)[-1].split("\n", 1)[0]
        self.searches.append(query)
        return GroundedAnswer(
            text=f"Summary for: {query}",
            citations=[{
                "url": "https://www.jstor.org/stable/1",
                "title": "A paper",
                "snippet": "A detail the summary left out.",
            }],
            queries=[query],
        )


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    """An orchestrator with no network, no cache and no page reads."""
    from phansora.products.chrono_origin.services import cache as cache_mod

    monkeypatch.setattr(orch, "get_cached", lambda *a, **k: None)
    monkeypatch.setattr(orch, "save_cached", lambda *a, **k: None)
    monkeypatch.setattr(orch, "read_best", lambda *a, **k: [])
    monkeypatch.setattr(orch, "mine_references", lambda *a, **k: [])
    assert cache_mod.SCHEMA_VERSION  # imported for the same reason it is patched

    client = ScriptedClient()
    o = orch.TraceOrchestrator(client=client)
    o.settings = o.settings.model_copy(update={"chrono_chase_enabled": False})
    return o, client


class TestEveryStrandIsSearchedFor:
    """Coverage comes from the strand templates now, not from a model's judgement."""

    def test_each_strand_buys_exactly_one_search(self, pipeline):
        o, client = pipeline
        o.run(TraceRequest(title="Jesus Christ"))
        assert len(client.searches) == len(orch._STRANDS)

    def test_no_strand_is_skipped(self, pipeline):
        o, client = pipeline
        o.run(TraceRequest(title="Jesus Christ"))
        searched = " ".join(client.searches).lower()
        # One distinctive phrase per strand template. A strand that stops being
        # searched for stops being able to appear in any trace at all.
        for phrase in (
            "survive from before",           # precursor_evidence
            "earliest surviving texts",      # earliest_texts
            "shelfmark",                     # manuscripts
            "outside the tradition",         # external_sources
            "administrative records",        # documents_records
            "inscriptions, coins, seals",    # inscriptions_artifacts
            "excavation reports",            # archaeology
        ):
            assert phrase in searched, f"nothing searched for {phrase!r}"

    def test_the_subject_is_named_in_every_search(self, pipeline):
        o, client = pipeline
        o.run(TraceRequest(title="Jesus Christ"))
        assert client.searches
        assert all("Jesus Christ" in q for q in client.searches)

    def test_no_search_is_paid_for_twice(self, pipeline):
        o, client = pipeline
        o.run(TraceRequest(title="Jesus Christ"))
        assert len(client.searches) == len(set(client.searches))


class TestTheExpensiveCallHappensOnce:
    """The regression this pipeline was rebuilt around.

    Producing structured JSON is what costs time — roughly seven times a search on
    the same material. One call per trace is the design; anything that reintroduces
    a per-round call brings back the twenty-minute trace.
    """

    def test_exactly_one_json_call_per_trace(self, pipeline):
        o, client = pipeline
        o.run(TraceRequest(title="Jesus Christ"))
        assert len(client.json_calls) == 1

    def test_that_call_is_the_synthesis(self, pipeline):
        o, client = pipeline
        o.run(TraceRequest(title="Jesus Christ"))
        assert client.json_calls[0].startswith("You are assembling the chain of evidence")


class TestSynthesisIsHandedTheResearch:
    def test_the_strand_plan_reaches_synthesis(self, pipeline):
        o, client = pipeline
        o.run(TraceRequest(title="Jesus Christ"))
        assert "RESEARCH PLAN AND WHAT IT COVERED" in client.synthesize_prompt
        assert "earliest_texts: RESEARCHED" in client.synthesize_prompt

    def test_every_search_result_reaches_synthesis(self, pipeline):
        """With no extract stage in between, the corpus IS the research."""
        o, client = pipeline
        o.run(TraceRequest(title="Jesus Christ"))
        for query in client.searches:
            assert f"[{query}]" in client.synthesize_prompt, f"{query!r} never arrived"

    def test_the_raw_snippets_arrive_too_not_just_the_summary(self, pipeline):
        """A summariser writing 300 words about six results drops most of what they
        said, and the dropped detail is exactly what synthesis is looking for. The
        snippets are already fetched and already paid for."""
        o, client = pipeline
        o.run(TraceRequest(title="Jesus Christ"))
        assert "A detail the summary left out." in client.synthesize_prompt


class TestTheResponseIsStillWellFormed:
    def test_a_run_produces_a_serialisable_trace(self, pipeline):
        o, _ = pipeline
        result = o.run(TraceRequest(title="Jesus Christ"))
        assert result.model_dump()["origin"]["source_title"] == "O"

    def test_the_searches_are_reported_back(self, pipeline):
        o, client = pipeline
        result = o.run(TraceRequest(title="Jesus Christ"))
        assert set(result.queries_run) == set(client.searches)

    def test_an_empty_synthesis_fails_instead_of_caching_a_hollow_success(self, pipeline):
        """A trace with no origin and no timeline is a FAILED trace.

        It still serialises into a well-formed response — every field simply takes
        its default — so it used to be stored and cached as a success with the
        user's credit spent. Raising sends it down the refund path.
        """
        o, client = pipeline
        client.reason_json = lambda prompt, use_reasoning_model=False: {}
        with pytest.raises(RuntimeError, match="no timeline and no origin"):
            o.run(TraceRequest(title="Jesus Christ"))
