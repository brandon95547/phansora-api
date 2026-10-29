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


# ── Neutral ─────────────────────────────────────────────────────────────────
# Asked about a documentary on Pizzagate, the conversation offered four angles that all
# assumed the answer ("how the rumor spread", "fact-checking the claims", ...). It finds out
# what THEIR narration should say; the AI's verdict on the subject has no place in it.

def test_the_model_is_told_to_stay_neutral_and_offer_every_angle(model):
    model["reply"] = {"done": False, "question": "What angle should it take?", "choices": []}
    converse.next_turn(ask(description="A documentary about Pizzagate"))
    system = model["seen"][0]["system"]
    assert "Do not judge the subject in your own voice" in system
    assert "their own take is always among them" in system
    assert "never an angle\nthat assumes the answer" in system


def test_a_reply_that_judges_is_asked_again_and_the_neutral_one_used(model, monkeypatch):
    replies = iter([
        {"done": False, "question": "Which part of the rumor?", "choices": ["How the rumor spread", "Fact-checking the claims"]},
        {"done": False, "question": "What angle should it take?", "choices": ["As believers see it", "The evidence on every side", "Let viewers decide"]},
    ])
    seen = []

    def fake(system, user, *, max_output_tokens=2000):
        seen.append(user)
        return next(replies)

    monkeypatch.setattr(llm, "deepseek_json", fake)
    out = converse.next_turn(ask(description="A documentary about Pizzagate"))
    assert len(seen) == 2
    assert 'judged the subject in your own voice ("fact-checking", "rumor")' in seen[1]
    assert out["question"] == "What angle should it take?"
    assert out["choices"] == ["As believers see it", "The evidence on every side", "Let viewers decide"]


def test_what_still_judges_after_the_second_try_is_dropped(model):
    # The same judging reply both times.
    model["reply"] = {
        "done": False, "question": "How should the hoax be told?",
        "choices": ["How the rumor spread", "Who was hurt", "Why people believed it"],
    }
    out = converse.next_turn(ask(description="A documentary about Pizzagate"))
    assert len(model["seen"]) == 2
    # The question judged: a neutral one stands in, and its answers go with it.
    assert out["question"] == converse.NEUTRAL_QUESTION
    assert out["choices"] == []


def test_judging_answers_are_dropped_and_the_rest_kept(model):
    model["reply"] = {
        "done": False, "question": "What should viewers take away?",
        "choices": ["Why the story was debunked", "What its believers saw in it", "Let them decide"],
    }
    out = converse.next_turn(ask(description="A documentary about Pizzagate"))
    assert out["question"] == "What should viewers take away?"
    assert out["choices"] == ["What its believers saw in it", "Let them decide"]


def test_their_own_words_may_be_echoed(model):
    model["reply"] = {"done": False, "question": "When did the rumor start, in your telling?", "choices": ["2016", "Earlier"]}
    out = converse.next_turn(ask(description="How the Pizzagate rumors spread online"))
    assert len(model["seen"]) == 1
    assert out["question"] == "When did the rumor start, in your telling?"


def test_a_brief_that_judges_gives_way_to_their_words(model):
    model["reply"] = {"done": True, "brief": "A documentary debunking the Pizzagate conspiracy theory."}
    out = converse.next_turn(ask(description="A documentary about Pizzagate", finish=True))
    assert out["done"] is True
    assert out["brief"].startswith("A documentary about Pizzagate")
    assert "debunk" not in out["brief"]


def test_true_and_false_count_only_where_they_judge():
    assert converse.verdicts("A true crime story, told plainly", "") == []
    assert converse.verdicts("The claims are false", "") == ["are false"]
    assert converse.verdicts("Why the false claims spread", "") == ["false claims"]
