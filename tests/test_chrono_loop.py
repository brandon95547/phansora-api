"""The trace pipeline, driven end to end against a scripted model.

This file used to test a research LOOP: plan strands, search, extract, decide
whether to go round again. Then it tested a seven-way fan-out. Both are gone, and
the reasons are what is worth pinning.

The loop went because generating JSON is the expensive act and the loop bought it
once per round: a round's six searches finished in 14 seconds and the call that
turned them into JSON took 104, every round, overrunning its budget and starting
over. The trace failed outright at 20 minutes.

The fan-out went because it decided where every chain STARTED, and decided it
wrongly. Seven fixed queries, one per evidence category — six asking what survives
ABOUT the subject, one asking what its evidence DESCENDS FROM. The chain rule says a
chain begins with descent, so the deciding half was outvoted six to one in every
corpus. A trace of Jesus opened at the Dead Sea Scrolls and lost the four centuries
of scripture the Scrolls are copies of. It existed only because the old provider
could not search and the queries had to be guessed in advance.

So a trace is now two model calls: one grounded research call where the model runs
its own searches, and one JSON call that turns the corpus into a timeline.
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
        self.searches.append(prompt)
        return GroundedAnswer(
            text="What the research found.",
            citations=[{
                "url": "https://www.jstor.org/stable/1",
                "title": "A paper",
                "snippet": "A detail the summary left out.",
            }],
            # The queries the MODEL chose, which is what grounding reports back.
            queries=["septuagint earliest manuscripts", "dead sea scrolls dating"],
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


class TestTheResearchIsOneCall:
    """One grounded call, and the model chooses its own queries."""

    def test_exactly_one_search_call_per_trace(self, pipeline):
        o, client = pipeline
        o.run(TraceRequest(title="Jesus Christ"))
        assert len(client.searches) == 1

    def test_the_prompt_asks_for_descent_before_evidence_about_the_subject(self, pipeline):
        """The ordering is the fix, not a formatting choice.

        Descent decides where a chain starts, and it is the half that loses whenever
        anything competes with it. Asking for it first, in its own required part, is
        what stops a trace opening at a copy of something older.
        """
        o, client = pipeline
        o.run(TraceRequest(title="Jesus Christ"))
        prompt = client.searches[0]
        assert "WHAT THIS DESCENDS FROM" in prompt
        assert "WHAT SURVIVES ABOUT THE SUBJECT" in prompt
        assert prompt.index("WHAT THIS DESCENDS FROM") < prompt.index("WHAT SURVIVES ABOUT THE SUBJECT")

    def test_the_prompt_still_carries_the_evidence_vocabulary(self, pipeline):
        """The strand list was worth keeping; the fan-out was not.

        Naming shelfmarks, ostraca and excavation reports is how the model is told
        what a findable object looks like.
        """
        o, client = pipeline
        o.run(TraceRequest(title="Jesus Christ"))
        prompt = client.searches[0].lower()
        for word in ("shelfmark", "ostraca", "excavation report", "repository",
                     "critical edition", "composition"):
            assert word in prompt, f"{word!r} dropped from the research prompt"

    def test_the_subject_is_named(self, pipeline):
        o, client = pipeline
        o.run(TraceRequest(title="Jesus Christ"))
        assert "Jesus Christ" in client.searches[0]

    def test_the_queries_reported_are_the_models_own(self, pipeline):
        """Not the prompt we sent. The user is shown what was actually searched."""
        o, _ = pipeline
        result = o.run(TraceRequest(title="Jesus Christ"))
        assert result.queries_run == ["septuagint earliest manuscripts", "dead sea scrolls dating"]


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
    def test_the_research_reaches_synthesis(self, pipeline):
        """With no extract stage in between, the corpus IS the research."""
        o, client = pipeline
        o.run(TraceRequest(title="Jesus Christ"))
        assert "What the research found." in client.synthesize_prompt

    def test_synthesis_is_not_told_a_strand_plan_that_no_longer_exists(self, pipeline):
        """The old block asserted every strand was RESEARCHED, unconditionally.

        It was built from the planned list, not from what came back, so it said the
        same thing on every trace and told synthesis nothing it could act on.
        """
        o, client = pipeline
        o.run(TraceRequest(title="Jesus Christ"))
        assert "RESEARCH PLAN AND WHAT IT COVERED" not in client.synthesize_prompt

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
        """What the MODEL searched, not the prompt we handed it.

        The prompt is ours and says nothing about what was looked up; grounding
        reports the real queries, and those are what the user is shown.
        """
        o, _ = pipeline
        result = o.run(TraceRequest(title="Jesus Christ"))
        assert result.queries_run == ["septuagint earliest manuscripts", "dead sea scrolls dating"]

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
