"""The Gemini research client.

Chrono-Origin's whole claim is that a timeline step is backed by something that
actually survives, and a citation the model produced from memory looks identical to
one it read on a page. So the tests that matter here are about what the client does
with grounding metadata — the only part of the response that says where the words
came from — and about failures being loud rather than empty.
"""
from __future__ import annotations

import json

import pytest

from phansora.shared.ai import gemini_research as G


def cfg(**kw):
    base = dict(api_key="k", model="gemini-3.5-flash-lite", reasoning_model="gemini-3.5-flash-lite")
    base.update(kw)
    return G.GeminiConfig(**base)


def client(monkeypatch, response):
    """A client whose transport returns a canned Gemini response body."""
    c = G.GeminiResearchClient(cfg())
    captured = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return response

    monkeypatch.setattr(c, "_generate", lambda **kw: fake(**kw))
    return c, captured


def grounded_body(text="Summary.", chunks=(), queries=()):
    return {
        "candidates": [{
            "content": {"parts": [{"text": text}]},
            "groundingMetadata": {
                "groundingChunks": [{"web": {"uri": u, "title": t}} for u, t in chunks],
                "webSearchQueries": list(queries),
            },
        }],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20},
    }


# ------------------------------------------------------------------ configuration
def test_it_refuses_to_start_without_a_key():
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        G.GeminiResearchClient(cfg(api_key=""))


def test_it_refuses_to_start_without_a_model():
    """No hardcoded fallback: a retired model name must fail here, not at the API."""
    with pytest.raises(RuntimeError, match="GEMINI_MODEL"):
        G.GeminiResearchClient(cfg(model=""))


def test_the_key_can_come_from_either_env_name(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "from-google-var")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    monkeypatch.delenv("CHRONO_MODEL", raising=False)
    assert G.GeminiConfig.from_env().api_key == "from-google-var"


# ------------------------------------------------------------------ grounded search
def test_search_asks_for_the_search_tool(monkeypatch):
    """Without this the model answers from memory and the trace is fiction."""
    c, captured = client(monkeypatch, grounded_body())
    c.grounded_search("When were the Dead Sea Scrolls found?")
    assert captured["grounded"] is True
    assert captured.get("json_out", False) is False


def test_citations_come_only_from_grounding_metadata(monkeypatch):
    """Sources the response says it used — never ones parsed out of the prose.

    A URL the model wrote into its answer is a URL it may have invented, and
    attributing a claim to a page that never made it is the one failure this product
    cannot survive.
    """
    body = grounded_body(
        text="The scrolls were found in 1947. See https://invented.example/not-a-source",
        chunks=[("https://real.example/a", "A real page")],
    )
    c, _ = client(monkeypatch, body)
    answer = c.grounded_search("q")
    assert [x["url"] for x in answer.citations] == ["https://real.example/a"]


def test_duplicate_sources_are_counted_once(monkeypatch):
    body = grounded_body(chunks=[
        ("https://a.example/x", "A"),
        ("https://a.example/x", "A again"),
        ("https://b.example/y", "B"),
    ])
    c, _ = client(monkeypatch, body)
    assert len(c.grounded_search("q").citations) == 2


def test_a_source_with_no_title_still_carries_its_url(monkeypatch):
    c, _ = client(monkeypatch, grounded_body(chunks=[("https://a.example/x", "")]))
    cite = c.grounded_search("q").citations[0]
    assert cite["title"] == "https://a.example/x"


def test_the_queries_the_model_actually_ran_are_reported(monkeypatch):
    """Shown to the user as what was searched, so it has to be the real list."""
    body = grounded_body(queries=["dead sea scrolls discovery 1947", "qumran cave 1"])
    c, _ = client(monkeypatch, body)
    assert c.grounded_search("q").queries == [
        "dead sea scrolls discovery 1947", "qumran cave 1",
    ]


def test_an_answer_from_memory_is_discarded_not_used(monkeypatch, caplog):
    """No chunks and no queries means the model never searched.

    Keeping that text would put a timeline step on the board whose only source is
    the model's recall, indistinguishable on screen from one backed by a manuscript.
    Empty instead, so it lands in the caller's "search unavailable" path and the user
    is told their search did not run — rather than that no evidence exists.
    """
    import logging

    c, _ = client(monkeypatch, grounded_body(text="I recall that...", chunks=[], queries=[]))
    with caplog.at_level(logging.WARNING):
        answer = c.grounded_search("q")
    assert answer.text == ""
    assert answer.citations == []
    assert any("without searching" in r.message for r in caplog.records)


def test_a_search_that_cited_nothing_keeps_its_text_but_says_so(monkeypatch, caplog):
    """It did search, so the answer is not recall — but the gap gets stated."""
    import logging

    body = grounded_body(text="Nothing conclusive found.", chunks=[], queries=["qumran cave 4"])
    c, _ = client(monkeypatch, body)
    with caplog.at_level(logging.WARNING):
        answer = c.grounded_search("q")
    assert answer.text == "Nothing conclusive found."
    assert answer.citations == []
    assert any("no sources" in r.message for r in caplog.records)


def test_a_failed_search_degrades_instead_of_killing_the_trace(monkeypatch):
    """One dead search is a dead lead, not a dead timeline."""
    c = G.GeminiResearchClient(cfg())

    def boom(**_):
        raise RuntimeError("Gemini 503: upstream unavailable")

    monkeypatch.setattr(c, "_generate", boom)
    answer = c.grounded_search("q")
    assert answer.text == "" and answer.citations == [] and answer.queries == []


# ------------------------------------------------------------------ reason_json
def json_body(payload, finish="STOP"):
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": finish}],
        "usageMetadata": {},
    }


def test_reasoning_asks_for_json_and_not_for_search(monkeypatch):
    """Grounding and a forced JSON mime type cannot be combined on this API."""
    c, captured = client(monkeypatch, json_body({"timeline": []}))
    c.reason_json("prompt")
    assert captured["json_out"] is True
    assert captured.get("grounded", False) is False


def test_it_parses_the_object(monkeypatch):
    c, _ = client(monkeypatch, json_body({"timeline": [{"source_title": "Sinaiticus"}]}))
    assert c.reason_json("p")["timeline"][0]["source_title"] == "Sinaiticus"


def test_it_survives_a_fenced_answer(monkeypatch):
    """Models fence JSON even when the system prompt forbids it."""
    c, _ = client(monkeypatch, json_body('```json\n{"ok": true}\n```'))
    assert c.reason_json("p") == {"ok": True}


def test_it_survives_prose_around_the_object(monkeypatch):
    c, _ = client(monkeypatch, json_body('Here you go:\n{"ok": true}\nHope that helps.'))
    assert c.reason_json("p") == {"ok": True}


def test_a_truncated_answer_raises_rather_than_reading_as_empty(monkeypatch):
    """The failure that has to stay loud.

    Cut-off JSON parses to a dict missing every key the caller wanted, which the
    pipeline then reports as "no evidence found" — the same sentence it uses for a
    subject that genuinely has none.
    """
    c, _ = client(monkeypatch, json_body('{"timeline": [{"source_title": "Sina', finish="MAX_TOKENS"))
    with pytest.raises(RuntimeError, match="cut off"):
        c.reason_json("p")


def test_a_complete_answer_at_the_limit_is_still_accepted(monkeypatch):
    """MAX_TOKENS with a closed object means it finished with nothing to spare."""
    c, _ = client(monkeypatch, json_body({"ok": True}, finish="MAX_TOKENS"))
    assert c.reason_json("p") == {"ok": True}


def test_unparseable_output_is_an_empty_dict_not_a_crash(monkeypatch):
    c, _ = client(monkeypatch, json_body("no json here at all"))
    assert c.reason_json("p") == {}


def test_a_json_array_is_not_passed_off_as_an_object(monkeypatch):
    """Callers index by key; a list would raise somewhere far from the cause."""
    c, _ = client(monkeypatch, json_body("[1, 2, 3]"))
    assert c.reason_json("p") == {}


def test_the_reasoning_model_is_used_for_judgement(monkeypatch):
    c = G.GeminiResearchClient(cfg(model="chat-model", reasoning_model="reason-model"))
    seen = {}
    monkeypatch.setattr(c, "_generate", lambda **kw: seen.update(kw) or json_body({}))

    c.reason_json("p", use_reasoning_model=True)
    assert seen["model"] == "reason-model"
    c.reason_json("p", use_reasoning_model=False)
    assert seen["model"] == "chat-model"


# ------------------------------------------------------------------ the factory
def test_the_factory_returns_this_client_for_gemini(monkeypatch):
    from phansora.shared.ai.research import build_research_client

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    monkeypatch.delenv("CHRONO_MODEL", raising=False)
    monkeypatch.delenv("CHRONO_REASONING_MODEL", raising=False)
    for name in ("gemini", "google"):
        monkeypatch.setenv("CHRONO_LLM_PROVIDER", name)
        assert isinstance(build_research_client(), G.GeminiResearchClient)


def test_gemini_is_the_default_provider(monkeypatch):
    from phansora.shared.ai.research import build_research_client

    monkeypatch.delenv("CHRONO_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    monkeypatch.delenv("CHRONO_MODEL", raising=False)
    monkeypatch.delenv("CHRONO_REASONING_MODEL", raising=False)
    assert isinstance(build_research_client(), G.GeminiResearchClient)


def test_it_satisfies_the_same_contract_as_the_other_clients():
    """The orchestrator calls these two by name and nothing else."""
    import inspect

    for name in ("grounded_search", "reason_json"):
        assert callable(getattr(G.GeminiResearchClient, name))

    search = inspect.signature(G.GeminiResearchClient.grounded_search)
    assert "temperature" in search.parameters
    reason = inspect.signature(G.GeminiResearchClient.reason_json)
    for kw in ("schema", "temperature", "use_reasoning_model"):
        assert kw in reason.parameters, kw
