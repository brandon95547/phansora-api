"""The narration form's conversation (narrava_studio.services.converse, POST /narration/converse).

DeepSeek is stubbed: what these check is ours, namely what the model is shown, and that
whatever it answers comes back as something the editor can use. A question with a few short
answers, or done with a brief, never a broken turn in the middle of a conversation.
"""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed on this host")

from fastapi.testclient import TestClient  # noqa: E402

from phansora.products.narrava_studio import server  # noqa: E402
from phansora.products.narrava_studio.models import ConverseRequest  # noqa: E402
from phansora.products.narrava_studio.services import converse, llm  # noqa: E402


@pytest.fixture()
def model(monkeypatch):
    """DeepSeek, stubbed: set `reply` to what it answers; `seen` holds what it was asked."""
    state = {"reply": {}, "seen": []}

    def fake(system, user, *, max_output_tokens=2000):
        state["seen"].append({"system": system, "user": user})
        return state["reply"]

    monkeypatch.setattr(llm, "deepseek_json", fake)
    return state


def ask(**kw):
    return ConverseRequest(**{"description": "A short film about lighthouse keepers", **kw})


def test_a_question_comes_back_with_a_few_short_answers(model):
    model["reply"] = {
        "done": False,
        "question": "Who is it for?",
        "choices": ["Kids", "kids", "History fans", "x" * 200, "Sailors", "Everyone"],
    }
    out = converse.next_turn(ask())
    assert out["done"] is False
    assert out["question"] == "Who is it for?"
    # No repeats, none longer than a button holds, no more than four.
    assert out["choices"][:3] == ["Kids", "History fans", "x" * converse.CHOICE_CHARS]
    assert len(out["choices"]) == converse.MAX_CHOICES
    assert out["brief"] == ""
    assert "Their description of the narration: A short film about lighthouse keepers" in model["seen"][0]["user"]


def test_the_conversation_so_far_is_shown_every_turn(model):
    model["reply"] = {"done": False, "question": "How should it end?", "choices": []}
    converse.next_turn(ask(turns=[{"question": "Who is it for?", "answer": "Kids"}]))
    user = model["seen"][0]["user"]
    assert "Q: Who is it for?" in user and "A: Kids" in user


def test_a_reply_with_no_question_is_the_end_of_the_conversation(model):
    model["reply"] = {"done": False, "question": "", "brief": "A warm piece for children."}
    out = converse.next_turn(ask())
    assert out["done"] is True
    assert out["question"] is None and out["choices"] == []
    assert out["brief"] == "A warm piece for children."


def test_after_the_last_question_it_finishes_and_their_words_stand_in_for_a_missing_brief(model):
    model["reply"] = {"done": False, "question": "One more thing?"}
    turns = [{"question": f"Q{i}?", "answer": f"A{i}"} for i in range(converse.MAX_QUESTIONS)]
    out = converse.next_turn(ask(turns=turns))
    assert converse.FINISH in model["seen"][0]["user"]
    assert out["done"] is True and out["question"] is None
    # The model gave no brief: the writer gets what they said, not nothing.
    assert out["brief"].startswith("A short film about lighthouse keepers")
    assert "Q0? A0" in out["brief"]


def test_write_now_asks_for_the_brief_at_once(model):
    model["reply"] = {"done": True, "brief": "Told for children, warm, ending at dawn."}
    out = converse.next_turn(ask(finish=True))
    assert converse.FINISH in model["seen"][0]["user"]
    assert out == {"done": True, "question": None, "choices": [], "brief": "Told for children, warm, ending at dawn.", "topics": []}


def topics_ask(**kw):
    topics = [
        {"index": 3, "title": "The storm", "summary": "A ship founders.", "recommended": True},
        {"index": 7, "title": "The rescue", "summary": "The keeper rows out."},
        {"index": 9, "title": "Automation", "summary": "The light is automated.", "recommended": True},
    ]
    return ask(kind="topics", topics=topics, document_title="Keepers", description="Use 2 and 3", **kw)


def test_topics_are_numbered_as_the_editor_shows_them_and_mapped_back(model):
    model["reply"] = {"done": True, "brief": "The rescue, then the end of an era.", "topics": [2, 3, 2, 99, True]}
    out = converse.next_turn(topics_ask())
    user = model["seen"][0]["user"]
    assert "1. The storm (recommended): A ship founders." in user
    assert "2. The rescue: The keeper rows out." in user
    assert "Imported document: Keepers" in user
    # 2 and 3 as shown are chapters 7 and 9; the repeat, the unknown and the bool are dropped.
    assert out["topics"] == [7, 9]


def test_no_topics_named_means_the_recommended_ones(model):
    model["reply"] = {"done": True, "brief": "As they said.", "topics": []}
    assert converse.next_turn(topics_ask())["topics"] == [3, 9]


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(llm, "deepseek_configured", lambda: True)
    return TestClient(server.app, raise_server_exceptions=False)


def test_the_endpoint_answers_with_the_next_turn(client, model):
    model["reply"] = {"done": False, "question": "Who is it for?", "choices": ["Kids"]}
    res = client.post("/narration/converse", json={"description": "Lighthouses", "video": "Video Short · Storytelling"})
    assert res.status_code == 200
    assert res.json() == {"done": False, "question": "Who is it for?", "choices": ["Kids"], "brief": "", "topics": []}
    assert "The video: Video Short · Storytelling" in model["seen"][0]["user"]


def test_choosing_topics_with_none_to_choose_from_is_refused(client, model):
    res = client.post("/narration/converse", json={"kind": "topics", "description": "Use 2"})
    assert res.status_code == 400
    assert model["seen"] == []


def test_without_a_deepseek_key_it_says_so(client, model, monkeypatch):
    monkeypatch.setattr(llm, "deepseek_configured", lambda: False)
    res = client.post("/narration/converse", json={"description": "Lighthouses"})
    assert res.status_code == 503
    assert "DEEPSEEK_API_KEY" in res.text
