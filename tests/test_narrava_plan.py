"""Planning scenes before anything is recorded (narrava_studio.services.plan, POST /storyboard/plan).

The model is stubbed: what these check is ours, namely what the model is told (the idea, the
length and the house style), how its answer is cleaned (names, the words, search terms), and
that the scenes' lengths add up to the length asked for, never shorter than their words.
"""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed on this host")

from fastapi.testclient import TestClient  # noqa: E402

from phansora.products.narrava_studio import server  # noqa: E402
from phansora.products.narrava_studio.services import llm, plan  # noqa: E402


@pytest.fixture()
def model(monkeypatch):
    """The model, stubbed: set `reply` to what it answers; `seen` holds what it was asked."""
    state = {"reply": {}, "seen": []}

    def fake(system, user, *, max_output_tokens=2000):
        state["seen"].append({"system": system, "user": user})
        return state["reply"]

    monkeypatch.setattr(llm, "generate_json", fake)
    return state


def scene(text="The lighthouse stands alone.", **kw):
    return {"name": "The lighthouse", "text": text, "visual": "A stone lighthouse at dusk.",
            "media_type": "image", "search_terms": ["lighthouse", "coast at dusk"], "seconds": 5, **kw}


def test_it_asks_with_the_idea_the_length_and_the_style(model):
    model["reply"] = {"scenes": [scene()]}
    plan.plan_from_idea("A short history of lighthouses", 60, style="cinematic")
    asked = model["seen"][0]["user"]
    assert "A short history of lighthouses" in asked
    assert "about 60 seconds" in asked
    assert "VERBATIM" in model["seen"][0]["system"]


def test_the_lengths_add_up_to_the_length_asked_for(model):
    model["reply"] = {"scenes": [scene(seconds=4), scene(seconds=6), scene(seconds=5)]}
    out = plan.plan_from_idea("Lighthouses", 30)
    secs = [s["seconds"] for s in out["scenes"]]
    assert round(sum(secs), 1) == 30
    assert secs[1] > secs[0]


def test_a_scene_is_never_shorter_than_its_words(model):
    long = " ".join(["word"] * 20)  # 8 seconds at 150 words a minute
    model["reply"] = {"scenes": [scene(text=long, seconds=1), scene(seconds=8)]}
    out = plan.plan_from_idea("Lighthouses", 16)
    assert out["scenes"][0]["seconds"] == 8


def test_it_names_a_scene_from_its_words_when_the_model_did_not(model):
    model["reply"] = {"scenes": [scene(name="", text="Keepers lived in the tower for months.")]}
    out = plan.plan_from_idea("Lighthouses", 10)
    assert out["scenes"][0]["name"] == "Keepers lived in"


def test_it_keeps_no_more_scenes_than_allowed_and_drops_empty_ones(model):
    model["reply"] = {"scenes": [scene(), {"text": "", "visual": ""}, "nonsense", scene(), scene()]}
    out = plan.plan_from_idea("Lighthouses", 20, max_scenes=2)
    assert len(out["scenes"]) == 2


def test_nothing_usable_is_an_error_not_a_plan(model):
    model["reply"] = {"scenes": []}
    with pytest.raises(RuntimeError):
        plan.plan_from_idea("Lighthouses", 20)


def test_the_endpoint_answers(model, monkeypatch):
    monkeypatch.setattr(server, "_ensure_llm", lambda: None)
    model["reply"] = {"scenes": [scene(), scene(text="Then came the electric lamp.")]}
    r = TestClient(server.app).post("/storyboard/plan", json={"idea": "Lighthouses", "total_duration_sec": 12})
    assert r.status_code == 200
    body = r.json()
    assert [s["name"] for s in body["scenes"]] == ["The lighthouse", "The lighthouse"]
    assert round(sum(s["seconds"] for s in body["scenes"]), 1) == 12


def test_the_endpoint_fails_cleanly_when_the_model_does(model, monkeypatch):
    monkeypatch.setattr(server, "_ensure_llm", lambda: None)
    model["reply"] = {}
    r = TestClient(server.app).post("/storyboard/plan", json={"idea": "Lighthouses", "total_duration_sec": 12})
    assert r.status_code == 502
    assert "empty" not in r.text  # the exception's text never reaches the caller
