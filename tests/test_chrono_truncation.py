"""What happens when the model's answer does not arrive whole.

A real trace on prod spent 62,740 tokens, read four pages and built 36 source chains,
then stored a result with no origin and an empty timeline — marked done, credit spent.
The synthesize call had landed on exactly its 8,000-token cap: the reasoning model spent
the budget thinking, the JSON came back cut off, and nothing in the path noticed.

These pin the four places that let that happen.
"""
from __future__ import annotations

import json

import pytest

from phansora.shared.ai import deepseek_research as dr
from phansora.products.chrono_origin.pipeline.orchestrator import _as_queries


# ── the parser ──────────────────────────────────────────────────────────────

def test_parses_a_clean_object():
    assert dr._parse_json('{"origin": {"year": -1200}}') == {"origin": {"year": -1200}}


def test_still_strips_prose_and_fences_from_a_complete_object():
    # The reason the fallback exists at all.
    assert dr._parse_json('Here you go:\n```json\n{"a": 1}\n```') == {"a": 1}


def test_a_truncated_answer_raises_instead_of_returning_an_inner_object():
    """The exact failure that produced an empty trace.

    `rfind("}")` on a cut-off document lands on the end of some INNER object. That slice
    parses cleanly and returns a dict — with none of the caller's keys — so every field
    fell back to a default and the trace was stored as a success.
    """
    truncated = '{"origin": {"year": -1200, "summary": "earliest attestation"}, "timeline": [{"year": -1'
    with pytest.raises((json.JSONDecodeError, ValueError)):
        dr._parse_json(truncated)


def test_a_json_array_is_not_accepted_as_an_object():
    with pytest.raises(ValueError):
        dr._parse_json("[1, 2, 3]")


def test_empty_input_is_still_an_empty_object():
    assert dr._parse_json("") == {}
    assert dr._parse_json("   ") == {}


# ── the truncation retry ────────────────────────────────────────────────────

class _FakeClient(dr.DeepSeekResearchClient):
    """Drives reason_json without a network, recording the budgets it asked for."""

    def __init__(self, script):
        self._script = list(script)      # [(content, finish_reason), ...]
        self.budgets = []
        self._cfg = dr.DeepSeekConfig(api_key="k", model="m", reasoning_model="r")

    def _chat(self, *, system, user, model, max_tokens, json_mode=False):
        self.budgets.append(max_tokens)
        return self._script.pop(0)


def test_a_truncated_answer_is_retried_with_double_the_budget():
    good = '{"origin": {"year": -1200}}'
    client = _FakeClient([("{\"origin\": {\"year\": -12", "length"), (good, "stop")])
    assert client.reason_json("prompt") == {"origin": {"year": -1200}}
    assert client.budgets[1] == client.budgets[0] * 2


def test_it_gives_up_loudly_rather_than_returning_an_empty_trace():
    # Cut off every time: the honest outcome is an error the job can refund on, not a
    # result with nothing in it.
    cut = ("{\"origin\": {\"year\": -12", "length")
    client = _FakeClient([cut, cut, cut])
    with pytest.raises(RuntimeError, match="cut off"):
        client.reason_json("prompt")


def test_a_complete_answer_costs_no_extra_calls():
    client = _FakeClient([('{"ok": true}', "stop")])
    assert client.reason_json("prompt") == {"ok": True}
    assert len(client.budgets) == 1


def test_the_escalation_has_a_ceiling():
    assert dr.MAX_REASON_TOKENS > dr.DeepSeekConfig().reason_max_tokens


# ── the query shapes ────────────────────────────────────────────────────────

def test_plain_strings_pass_through():
    assert _as_queries(["earliest manuscript", "provenance"]) == ["earliest manuscript", "provenance"]


def test_tiered_objects_are_accepted():
    """The shape that killed a trace at 97%, 'Building response'.

    The decompose prompt asks for queries allocated across the source hierarchy, and the
    model sometimes answers with {tier, query} objects. Both are fair readings, so both
    are accepted — TraceResponse.queries_run is List[str] and used to reject the second.
    """
    got = _as_queries([
        {"tier": 1, "query": "Codex Sinaiticus provenance dating"},
        {"tier": 2, "query": "academic consensus disagreement"},
    ])
    assert got == ["Codex Sinaiticus provenance dating", "academic consensus disagreement"]


def test_mixed_shapes_in_one_list():
    assert _as_queries(["plain", {"tier": 3, "query": "tiered"}]) == ["plain", "tiered"]


def test_unusable_entries_are_dropped_not_searched():
    assert _as_queries([{"tier": 1}, "", "  ", None, 42, {"query": "  keep  "}]) == ["keep"]


def test_missing_or_empty_input():
    assert _as_queries(None) == []
    assert _as_queries([]) == []
