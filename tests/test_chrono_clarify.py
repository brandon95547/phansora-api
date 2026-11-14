"""The subject check before research: which subject was meant, when the words name several."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from phansora.products.chrono_origin import server
from phansora.products.chrono_origin.models import ClarifyRequest
from phansora.products.chrono_origin.pipeline.clarify import clarify, clarify_prompt, parse_clarify

REVELATION = {
    "clear": False,
    "question": "Which Revelation do you mean?",
    "options": [
        {"label": "The Book of Revelation", "context": "the last book of the New Testament"},
        {"label": "Revelation (2014 film)", "context": "the 2014 film"},
    ],
}


class Client:
    def __init__(self, answer):
        self.answer, self.prompts, self.reasoning = answer, [], []

    def reason_json(self, prompt, *, use_reasoning_model=True):
        self.prompts.append(prompt)
        self.reasoning.append(use_reasoning_model)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def test_an_ambiguous_subject_comes_back_with_its_meanings():
    result = clarify(Client(REVELATION), ClarifyRequest(title="Revelation", kind="chronology"))
    assert result.clear is False
    assert result.question == "Which Revelation do you mean?"
    assert [(o.label, o.context) for o in result.options] == [
        ("The Book of Revelation", "the last book of the New Testament"),
        ("Revelation (2014 film)", "the 2014 film"),
    ]


def test_it_is_one_light_call_without_a_search():
    client = Client({"clear": True})
    assert clarify(client, ClarifyRequest(title="The French Revolution")).clear is True
    assert client.reasoning == [False]


def test_the_prompt_names_the_research_and_the_context():
    prompt = clarify_prompt(ClarifyRequest(title="Mercury", context="Roman religion", kind="genealogy"))
    assert 'the genealogy of "Mercury"' in prompt
    assert 'context given: "Roman religion"' in prompt
    assert "Never ask about scope" in prompt
    assert "with no context given" in clarify_prompt(ClarifyRequest(title="Mercury"))


@pytest.mark.parametrize("answer", [
    None, "not json", {"clear": True}, {"clear": "no"},
    {"clear": False, "options": [{"label": "One", "context": "only one"}]},
    {"clear": False, "options": [{"label": "", "context": "x"}, {"label": "B"}]},
    {"clear": False, "options": [{"label": "Same", "context": "a"}, {"label": "same", "context": "b"}]},
])
def test_an_answer_with_fewer_than_two_usable_meanings_is_clear(answer):
    assert parse_clarify(answer).clear is True


def test_meanings_are_trimmed_capped_and_given_a_question():
    many = {"clear": False, "options": [{"label": f"  Meaning {i}  " + "x" * 80, "context": "c " * 60}
                                        for i in range(8)]}
    result = parse_clarify(many)
    assert len(result.options) == 5
    assert all(len(o.label) <= 60 and len(o.context) <= 80 for o in result.options)
    assert result.options[0].label.startswith("Meaning 0")
    assert result.question == "Which one do you mean?"


def test_the_request_is_checked_at_the_boundary():
    with pytest.raises(ValidationError):
        ClarifyRequest(title="x")
    with pytest.raises(ValidationError):
        ClarifyRequest(title="Revelation", kind="horoscope")
    with pytest.raises(ValidationError):
        ClarifyRequest(title="Revelation", context="c" * 301)


def _api(monkeypatch, answer):
    # The app as it starts, with its research client stubbed at the boundary: no keys.
    monkeypatch.setattr(server, "_provider_configured", lambda: True)
    monkeypatch.setattr(server, "TraceOrchestrator", lambda: SimpleNamespace(client=Client(answer)))
    return TestClient(server.app)


def test_the_route_answers_with_the_meanings(monkeypatch):
    with _api(monkeypatch, REVELATION) as api:
        resp = api.post("/clarify", json={"title": "Revelation", "kind": "chronology"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["clear"] is False and len(body["options"]) == 2


def test_a_failed_check_is_a_plain_503_with_no_provider_detail(monkeypatch):
    with _api(monkeypatch, RuntimeError("provider said: secret body")) as api:
        resp = api.post("/clarify", json={"title": "Revelation"})
    assert resp.status_code == 503
    assert "secret" not in resp.text
