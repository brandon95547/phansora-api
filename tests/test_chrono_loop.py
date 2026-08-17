"""The research loop, driven end to end against a scripted model.

The unit tests around this pin single functions. What they cannot show is the
behaviour that actually failed on a real subject: a trace that planned nine
strands, ran twenty-four searches, and came back having researched neither the
texts nor the calendar — because the loop's stopping rule fired on "nothing got
older" while five of its own planned strands were still open, and because the
one channel between the rounds and the report dropped the type off every item on
the way through.

So this runs the whole orchestrator with a fake client, a stubbed reader and a
disabled cache, and asserts on what the loop searched for and what synthesis was
handed. The fake model is deliberately unhelpful — it never volunteers a strand
query of its own — because that is the case the pipeline has to survive.
"""
from __future__ import annotations

import json

import pytest

from phansora.products.chrono_origin.models import TraceRequest
from phansora.products.chrono_origin.pipeline import orchestrator as orch
from phansora.shared.ai.research import GroundedAnswer


class ScriptedClient:
    """A model that plans a wide subject and then stops being helpful.

    It returns one mention per search, typed by whichever strand query prompted
    it, and never proposes a next query. Everything the loop achieves after
    round one is therefore the loop's own doing.
    """

    STRAND_BY_KEYWORD = {
        "calendar": "dating_framework",
        "composition dates": "text_composition",
        "surviving manuscripts": "manuscript_witness",
        "form by form": "linguistic_transmission",
        "outside the tradition": "external_attestation",
    }

    def __init__(self):
        self.searches = []
        self.synthesize_prompt = None

    # -- planning / extraction / synthesis all arrive here
    def reason_json(self, prompt, use_reasoning_model=False):
        if prompt.startswith("You are a research planner"):
            return {
                "entities": ["X"],
                "strands": [
                    {"strand": "text_composition", "why": "…"},
                    {"strand": "manuscript_witness", "why": "…"},
                    {"strand": "linguistic_transmission", "why": "…"},
                    {"strand": "dating_framework", "why": "…"},
                ],
                "queries": ["opening question about the subject"],
            }
        if prompt.startswith("From the research material below"):
            mentions = []
            for i, note in enumerate(self._notes(prompt)):
                strand = next(
                    (s for k, s in self.STRAND_BY_KEYWORD.items() if k in note), "event"
                )
                mentions.append({
                    "year": 100 + len(self.searches) + i,
                    "node_type": strand,
                    "source_title": f"{strand} finding {len(self.searches)}.{i}",
                    "claim": "…",
                    "citations": ["https://www.jstor.org/stable/1"],
                    "source_tier": "academic",
                    "confidence": 0.6,
                })
            return {"mentions": mentions, "next_queries": [], "gaps": []}
        # synthesize
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

    @staticmethod
    def _notes(prompt: str):
        body = prompt.split("Research notes:\n---\n", 1)[-1].split("\n---\n", 1)[0]
        return [chunk for chunk in body.split("### Query: ") if chunk.strip()]

    def grounded_search(self, prompt):
        query = prompt.split("Search query: ", 1)[-1].split("\n", 1)[0]
        self.searches.append(query)
        return GroundedAnswer(
            text=query,
            citations=[{"url": "https://www.jstor.org/stable/1", "title": "A paper"}],
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


class TestOpenStrandsDriveTheSearches:
    def test_a_planned_strand_the_model_never_asks_about_still_gets_searched(self, pipeline):
        o, client = pipeline
        o.run(TraceRequest(title="Jesus Christ"))

        searched = " ".join(client.searches).lower()
        # The four the plan named. The model proposed exactly one query, ever.
        assert "calendar" in searched, "the calendar strand was never searched"
        assert "composition dates" in searched, "no search for when the texts were written"
        assert "surviving manuscripts" in searched
        assert "form by form" in searched, "the name chain was never traced"

    def test_the_subject_is_named_in_every_reserved_search(self, pipeline):
        o, client = pipeline
        o.run(TraceRequest(title="Jesus Christ"))
        reserved = [q for q in client.searches if q != "opening question about the subject"]
        assert reserved
        assert all("Jesus Christ" in q for q in reserved)

    def test_no_search_is_paid_for_twice(self, pipeline):
        o, client = pipeline
        o.run(TraceRequest(title="Jesus Christ"))
        assert len(client.searches) == len(set(client.searches))

    def test_the_loop_still_terminates(self, pipeline):
        o, client = pipeline
        result = o.run(TraceRequest(title="Jesus Christ"))
        assert result.iterations <= o.settings.chrono_max_depth


class TestSynthesisIsHandedTheResearch:
    def test_the_strand_plan_reaches_synthesis(self, pipeline):
        o, client = pipeline
        o.run(TraceRequest(title="Jesus Christ"))
        assert "RESEARCH PLAN AND WHAT IT COVERED" in client.synthesize_prompt
        assert "text_composition: RESEARCHED" in client.synthesize_prompt

    def test_every_item_arrives_with_the_type_the_extractor_gave_it(self, pipeline):
        o, client = pipeline
        o.run(TraceRequest(title="Jesus Christ"))
        block = client.synthesize_prompt
        for strand in ("text_composition", "manuscript_witness", "dating_framework",
                       "linguistic_transmission"):
            assert f"type={strand}" in block, f"{strand} lost its type before synthesis"


class TestTheResponseIsStillWellFormed:
    def test_a_run_produces_a_serialisable_trace(self, pipeline):
        o, _ = pipeline
        result = o.run(TraceRequest(title="Jesus Christ"))
        assert result.origin.year == 100
        assert json.dumps(result.model_dump(mode="json"))

    def test_the_source_list_carries_its_rank_and_role(self, pipeline):
        # The trace-level bibliography used to arrive unranked, so the UI had no
        # way to separate the leads from the evidence in it.
        o, _ = pipeline
        result = o.run(TraceRequest(title="Jesus Christ"))
        assert result.citations
        for c in result.citations:
            assert 1 <= c.tier_rank <= 5
            assert c.role in ("evidence", "discovery")
