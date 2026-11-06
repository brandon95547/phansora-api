"""The editor's guide (narrava_studio.services.guide, POST /narration/guide).

DeepSeek is stubbed: what these check is ours, namely what the model is shown (the articles,
first and in order, and an instruction to answer only from them) and that whatever it answers
comes back as something the guide can show.
"""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed on this host")

from fastapi.testclient import TestClient  # noqa: E402

from phansora.products.narrava_studio import server  # noqa: E402
from phansora.products.narrava_studio.models import GuideRequest  # noqa: E402
from phansora.products.narrava_studio.services import guide, llm  # noqa: E402

ARTICLES = [
    {"title": "Add a narration", "body": "Click **Narration**, then the + button."},
    {"title": "Choose the narrator's voice", "body": "Click the voice button at the top right."},
]


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
    return GuideRequest(**{"question": "How do I add a narration?", "articles": ARTICLES, **kw})


def test_it_answers_from_the_articles_and_names_them(model):
    model["reply"] = {"answer": "Click **Narration**, then the + button.", "articles": [1, 1, 9, True]}
    out = guide.answer(ask())
    assert out == {"answer": "Click **Narration**, then the + button.", "articles": ["Add a narration"]}


def test_the_model_sees_every_article_in_order_and_is_told_to_keep_to_them(model):
    model["reply"] = {"answer": "x", "articles": []}
    guide.answer(ask())
    system = model["seen"][0]["system"]
    assert "Answer only from the articles" in system
    assert system.index("[1] Add a narration") < system.index("[2] Choose the narrator's voice")
    # The question is not in the system prompt, so its long part stays the same between questions.
    assert "How do I add a narration?" not in system
    assert "Question: How do I add a narration?" in model["seen"][0]["user"]


def test_the_conversation_so_far_is_shown(model):
    model["reply"] = {"answer": "x"}
    history = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(6)]
    guide.answer(ask(question="And then?", history=history, video="Video Short · Storytelling"))
    user = model["seen"][0]["user"]
    assert "Their video: Video Short · Storytelling" in user
    # Only the latest few turns.
    assert "Q1" not in user and "Q2" in user and "A5" in user


def test_an_empty_reply_says_it_is_not_covered(model):
    model["reply"] = {}
    assert guide.answer(ask()) == {"answer": guide.NOT_COVERED, "articles": []}


def test_the_endpoint_answers(model, monkeypatch):
    monkeypatch.setattr(server, "_ensure_deepseek", lambda: None)
    model["reply"] = {"answer": "Click the voice button.", "articles": [2]}
    r = TestClient(server.app).post("/narration/guide", json={"question": "Change the voice?", "articles": ARTICLES})
    assert r.status_code == 200
    assert r.json() == {"answer": "Click the voice button.", "articles": ["Choose the narrator's voice"]}


def test_a_question_needs_articles(model, monkeypatch):
    monkeypatch.setattr(server, "_ensure_deepseek", lambda: None)
    r = TestClient(server.app).post("/narration/guide", json={"question": "Hi", "articles": []})
    assert r.status_code == 422
