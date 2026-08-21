"""The trace pipeline, driven end to end against a scripted model.

This file used to test a research LOOP: plan strands, search, extract, decide whether
to go round again. Then a seven-way fan-out. Then two calls — one grounded search and
one JSON synthesis. Each is gone, and the reasons are what is worth pinning.

The loop went because generating JSON is the expensive act and the loop bought it once
per round: a round's six searches finished in 14 seconds and the call that turned them
into JSON took 104, every round, overrunning its budget and starting over. The trace
failed outright at 20 minutes.

The fan-out went because it decided where every chain STARTED, and decided it wrongly.
Seven fixed queries, six asking what survives ABOUT the subject and one asking what its
evidence DESCENDS FROM, so the deciding half was outvoted six to one in every corpus.

Synthesis went last, and for the same reason as the loop: it was the expensive call, it
overran its budget four times in three days — each time failing a trace whose research
had already succeeded — and being a model it could shorten a list it found long. It was
there to produce JSON, but JSON was never the requirement. RESEARCH_PROMPT asks for
`Title - Date`, one per line, oldest first, and that is already the timeline.

So a trace is ONE model call, and code reads its answer.
"""
from __future__ import annotations

import pytest

from phansora.products.chrono_origin.models import TraceRequest
from phansora.products.chrono_origin.pipeline import orchestrator as orch
from phansora.shared.ai.research import GroundedAnswer


# What the model returns, in the format the research prompt asks for. Deliberately not
# in date order, and with a hyphenated title in it: both are things the real answers do.
SCRIPTED_LIST = """\
Proto-Sinaitic Script - c. 1800 BCE
Cuneiform Script - c. 3400 BCE
Dead Sea Scrolls - 3rd century BCE
King James Bible - c. 1611 CE
"""


class ScriptedClient:
    """A model that answers the research prompt, and counts every call made to it."""

    def __init__(self, text: str = SCRIPTED_LIST, citations=None, queries=None):
        self.searches = []
        self.json_calls = []
        self._text = text
        self._citations = citations if citations is not None else [{
            "url": "https://www.jstor.org/stable/1",
            "title": "A paper",
            "snippet": "A detail the summary left out.",
        }]
        self._queries = queries if queries is not None else [
            "septuagint earliest manuscripts", "dead sea scrolls dating",
        ]

    def reason_json(self, prompt, use_reasoning_model=False):
        self.json_calls.append(prompt)
        return {}

    def grounded_search(self, prompt):
        self.searches.append(prompt)
        return GroundedAnswer(
            text=self._text,
            citations=list(self._citations),
            # The queries the MODEL chose, which is what grounding reports back.
            queries=list(self._queries),
        )


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    """An orchestrator with no network and no cache."""
    monkeypatch.setattr(orch, "get_cached", lambda *a, **k: None)
    monkeypatch.setattr(orch, "save_cached", lambda *a, **k: None)

    client = ScriptedClient()
    return orch.TraceOrchestrator(client=client), client


class TestTheResearchIsOneCall:
    """One grounded call, and the model chooses its own queries."""

    def test_exactly_one_search_call_per_trace(self, pipeline):
        o, client = pipeline
        o.run(TraceRequest(title="Jesus Christ"))
        assert len(client.searches) == 1

    def test_the_subject_is_named(self, pipeline):
        o, client = pipeline
        o.run(TraceRequest(title="Jesus Christ"))
        assert "Jesus Christ" in client.searches[0]

    def test_the_queries_reported_are_the_models_own(self, pipeline):
        """Not the prompt we sent. The user is shown what was actually searched."""
        o, _ = pipeline
        result = o.run(TraceRequest(title="Jesus Christ"))
        assert result.queries_run == ["septuagint earliest manuscripts", "dead sea scrolls dating"]


class TestThereIsNoSecondModelCall:
    """The regression this pipeline was rebuilt around, one architecture later.

    Producing structured JSON is what costs time — roughly seven times a search on the
    same material — and it is also what failed: four traces in three days were researched
    successfully and then lost at the synthesis step's token ceiling. Reading the answer
    with code cannot overrun a budget, cannot time out and cannot decide the list would
    read better shorter.
    """

    def test_no_json_call_is_made(self, pipeline):
        o, client = pipeline
        o.run(TraceRequest(title="Jesus Christ"))
        assert client.json_calls == [], "paid a second model to reformat an answer we can read"

    def test_the_whole_trace_costs_one_call(self, pipeline):
        o, client = pipeline
        o.run(TraceRequest(title="Jesus Christ"))
        assert len(client.searches) + len(client.json_calls) == 1


class TestTheListBecomesTheTimeline:
    def test_every_item_arrives_as_a_node(self, pipeline):
        o, _ = pipeline
        result = o.run(TraceRequest(title="Jesus Christ"))
        titles = [result.origin.source_title] + [e.source_title for e in result.timeline]
        assert titles == [
            "Cuneiform Script", "Proto-Sinaitic Script", "Dead Sea Scrolls", "King James Bible",
        ]

    def test_the_nodes_are_in_date_order_oldest_first(self, pipeline):
        """The model is asked for oldest-first and gives it, but a list that came back a
        little out of order is shown in order rather than shown wrong."""
        o, _ = pipeline
        result = o.run(TraceRequest(title="Jesus Christ"))
        years = [e.year for e in result.timeline]
        assert years == sorted(years)
        assert result.origin.year == -3400

    def test_the_oldest_item_becomes_the_origin_and_is_not_repeated(self, pipeline):
        """The oldest item IS what the trace set out to find, so it moves out of the
        timeline rather than being copied into both and drawn twice."""
        o, _ = pipeline
        result = o.run(TraceRequest(title="Jesus Christ"))
        assert result.origin.source_title == "Cuneiform Script"
        assert "Cuneiform Script" not in [e.source_title for e in result.timeline]

    def test_a_node_carries_a_title_and_a_date_and_claims_nothing_else(self, pipeline):
        """An empty claim says we have no description of this item, which is true. A
        generated one would say we do."""
        o, _ = pipeline
        result = o.run(TraceRequest(title="Jesus Christ"))
        node = result.timeline[0]
        assert node.source_title == "Proto-Sinaitic Script"
        assert node.year == -1800
        assert node.claim == ""
        assert node.citations == []

    def test_a_span_keeps_both_ends(self, pipeline):
        """"3rd century BCE" is a hundred years, not a year."""
        o, _ = pipeline
        result = o.run(TraceRequest(title="Jesus Christ"))
        scrolls = next(e for e in result.timeline if e.source_title == "Dead Sea Scrolls")
        assert (scrolls.year, scrolls.year_end, scrolls.precision) == (-300, -201, "century")


class TestTheResponseIsStillWellFormed:
    def test_a_run_produces_a_serialisable_trace(self, pipeline):
        o, _ = pipeline
        result = o.run(TraceRequest(title="Jesus Christ"))
        assert result.model_dump()["origin"]["source_title"] == "Cuneiform Script"

    def test_sources_are_kept_at_the_top_level_when_the_model_grounded(self, pipeline):
        """Nothing on this path reads them, but discarding sources a search already paid
        for would make adding them back a research problem rather than a display one."""
        o, _ = pipeline
        result = o.run(TraceRequest(title="Jesus Christ"))
        assert [c.url for c in result.citations] == ["https://www.jstor.org/stable/1"]


class TestAnAnswerWithNoListIsAFailedTrace:
    """The 2026-08-21 incident, pinned.

    A trace came back with an empty board, was marked done, CACHED for thirty days and
    charged — and because it was cached, re-running it returned the same emptiness
    without ever calling the model again. The timeline is the product: if there is
    nothing to put on it, the trace failed and the credit goes back.
    """

    def test_an_empty_answer_fails_the_trace(self, pipeline):
        o, client = pipeline
        client._text = ""
        with pytest.raises(RuntimeError, match="came back empty"):
            o.run(TraceRequest(title="Jesus Christ"))

    def test_an_answer_with_no_readable_items_fails_the_trace(self, pipeline):
        """Prose instead of a list. The model answered; it just did not answer this."""
        o, client = pipeline
        client._text = "I could not find reliable information about this subject."
        with pytest.raises(RuntimeError, match="readable list"):
            o.run(TraceRequest(title="Jesus Christ"))

    def test_a_failed_trace_never_reaches_the_cache(self, pipeline, monkeypatch):
        """The part that made it stick: re-running could not clear it."""
        o, client = pipeline
        saved = []
        monkeypatch.setattr(orch, "save_cached", lambda *a, **k: saved.append(a))
        client._text = ""
        with pytest.raises(RuntimeError):
            o.run(TraceRequest(title="Jesus Christ"))
        assert saved == [], "an empty trace was written to the cache"


class TestAnUnsourcedAnswerIsStillATrace:
    """The other half of that incident, and the correction to how it was first fixed.

    The first fix discarded any answer the model had not searched for, on the grounds
    that memory is not evidence. True, and beside the point: it failed the whole trace
    and refunded the credit while the list the user asked for sat in the response,
    complete and in order. A node here is a title and a date — neither of which a
    citation was making truer.
    """

    def test_a_list_with_no_sources_still_produces_a_timeline(self, pipeline):
        o, client = pipeline
        client._citations = []
        client._queries = []
        result = o.run(TraceRequest(title="Jesus Christ"))
        assert len(result.timeline) == 3
        assert result.origin.source_title == "Cuneiform Script"

    def test_an_unsourced_trace_claims_no_sources(self, pipeline):
        """It must never come back looking researched when it was not."""
        o, client = pipeline
        client._citations = []
        client._queries = []
        result = o.run(TraceRequest(title="Jesus Christ"))
        assert result.citations == []
        assert all(e.citations == [] for e in result.timeline)
