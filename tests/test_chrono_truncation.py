"""What happens when the model's answer does not arrive whole.

A real trace on prod spent 62,740 tokens, read four pages and built 36 source chains,
then stored a result with no origin and an empty timeline — marked done, credit spent.
The synthesize call had landed on exactly its 8,000-token cap: the reasoning model spent
the budget thinking, the JSON came back cut off, and nothing in the path noticed.

These pin the four places that let that happen.
"""
from __future__ import annotations

import json

import httpx
import pytest

from phansora.shared.ai import deepseek as ds
from phansora.shared.ai import deepseek_research as dr


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

@pytest.fixture
def scripted(monkeypatch):
    """A client whose model replies are a fixed list of (content, finish_reason), served
    over the shared helper's HTTP, recording the budget each request asked for."""

    def build(script):
        script = list(script)
        client = dr.DeepSeekResearchClient.__new__(dr.DeepSeekResearchClient)
        client._cfg = dr.DeepSeekConfig(api_key="k", model="m", reasoning_model="r")
        client.budgets = []

        def answer(request: httpx.Request) -> httpx.Response:
            sent = json.loads(request.content)
            client.budgets.append(sent["max_tokens"])
            assert sent["model"] == "r" and "thinking" not in sent
            content, finish = script.pop(0)
            return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": finish}]})

        real_client = httpx.Client
        monkeypatch.setattr(ds.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(answer), **kw))
        return client

    return build


def test_a_truncated_answer_is_retried_with_double_the_budget(scripted):
    good = '{"origin": {"year": -1200}}'
    client = scripted([("{\"origin\": {\"year\": -12", "length"), (good, "stop")])
    assert client.reason_json("prompt") == {"origin": {"year": -1200}}
    assert client.budgets[1] == client.budgets[0] * 2


def test_it_gives_up_loudly_rather_than_returning_an_empty_trace(scripted):
    # Cut off every time: the honest outcome is an error the job can refund on, not a
    # result with nothing in it.
    cut = ("{\"origin\": {\"year\": -12", "length")
    client = scripted([cut, cut, cut])
    with pytest.raises(RuntimeError, match="cut off"):
        client.reason_json("prompt")


def test_a_complete_answer_costs_no_extra_calls(scripted):
    client = scripted([('{"ok": true}', "stop")])
    assert client.reason_json("prompt") == {"ok": True}
    assert len(client.budgets) == 1


def test_the_escalation_has_a_ceiling():
    assert dr.MAX_REASON_TOKENS > dr.DeepSeekConfig().reason_max_tokens


# ── the query shapes ────────────────────────────────────────────────────────





