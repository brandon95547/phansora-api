"""What POST /users/{id}/sources/classify says about each section: its subject, topic and title.

DeepSeek is replaced where the library asks it (``talari.llm.ask_json``), so these tests see the
prompt that would be sent and decide the answer — no network and no keys.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest
from fastapi.testclient import TestClient

from phansora.products.talari import library
from phansora.products.talari import llm as llm_module
from phansora.products.talari.config import Settings
from phansora.products.talari.errors import ServiceError
from phansora.products.talari.schemas import ClassifyRequest, SectionClass
from phansora.products.talari.server import create_app

Reply = dict | Exception | Callable[[str], dict | Exception]


class FakeAsk:
    """Stands in for ``llm.ask_json``: remembers each call, answers from ``replies`` in turn."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.replies: list[Reply] = []

    async def __call__(self, system: str, user: str, *, max_tokens: int, temperature: float, **options) -> dict:
        self.calls.append({"system": system, "user": user, "max_tokens": max_tokens, "temperature": temperature})
        reply = self.replies.pop(0)
        if callable(reply):
            reply = reply(user)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def ask(monkeypatch) -> FakeAsk:
    fake = FakeAsk()
    monkeypatch.setattr(llm_module, "ask_json", fake)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return fake


@pytest.fixture
def client(tmp_path: Path, ask: FakeAsk):
    settings = Settings(
        comfyui_url="http://comfy.test",
        comfyui_checkpoint="",
        comfyui_workflow="",
        comfyui_base_size=1024,
        comfyui_steps=20,
        comfyui_cfg=7.0,
        comfyui_sampler="euler",
        comfyui_scheduler="normal",
        comfyui_timeout_s=5.0,
        data_dir=tmp_path / "talari",
        comfyui_pipeline="checkpoint",
    )
    with TestClient(create_app(settings, transport=httpx.MockTransport(lambda _r: httpx.Response(404)))) as test_client:
        yield test_client


def classify(client: TestClient, sections: list[dict], **names: Any) -> list[dict]:
    response = client.post("/users/7/sources/classify", json={"sections": sections, **names})
    assert response.status_code == 200, response.text
    return response.json()["sections"]


def section(id: str, title: str = "The Weighing of the Heart") -> dict:
    return {"id": id, "title": title, "text": "Anubis weighs the heart against a feather."}


def entry(id: str, **fields: Any) -> dict:
    return {"id": id, "kind": "text", **fields}


# --- what comes back -----------------------------------------------------------------


def test_a_section_gets_a_subject_a_topic_and_related_topics(client: TestClient, ask: FakeAsk) -> None:
    ask.replies = [{"sections": [
        entry("a", subject="Ancient Egyptian Religion", topic="Osiris and Resurrection", topics=["Osiris and Resurrection", "The Afterlife", "Judgment"]),
        entry("b", kind="exercise", subject="Ancient Egyptian Religion", topic="Judgment", topics=["Judgment"]),
    ]}]
    body = classify(client, [section("a"), section("b", "Questions to answer")])
    # The fields the app reads, under exactly these names.
    assert body == [
        {"id": "a", "kind": "text", "subject": "Ancient Egyptian Religion", "topic": "Osiris and Resurrection",
         "topics": ["Osiris and Resurrection", "The Afterlife", "Judgment"], "title": ""},
        {"id": "b", "kind": "exercise", "subject": "Ancient Egyptian Religion", "topic": "Judgment", "topics": ["Judgment"], "title": ""},
    ]


def test_the_prompt_shows_the_names_in_use_and_which_headings_need_a_title(client: TestClient, ask: FakeAsk) -> None:
    ask.replies = [{"sections": []}]
    classify(
        client,
        [section("a", "Part 3"), section("b", "Mummification"), section("c", "")],
        subjects=["Ancient Egyptian Religion"],
        topics=["Judgment", "Burial"],
    )
    (call,) = ask.calls
    assert 'Subjects already in use: ["Ancient Egyptian Religion"]' in call["user"]
    assert 'Topics already in use: ["Judgment", "Burial"]' in call["user"]
    assert 'also needs a "title": ["a", "c"]' in call["user"]
    assert "subject" in call["system"] and "topic" in call["system"] and "title" in call["system"]
    # Room for the larger answer: more than the 60 tokens a section had before.
    assert call["max_tokens"] >= 110 * 3


def test_the_prompt_leaves_out_what_there_is_none_of(client: TestClient, ask: FakeAsk) -> None:
    ask.replies = [{"sections": []}]
    classify(client, [section("a")])
    (call,) = ask.calls
    assert "already in use" not in call["user"] and "placeholder heading" not in call["user"]


@pytest.mark.parametrize("heading", ["Part 3", "", "Opening", "(untitled)"])
def test_a_placeholder_heading_gets_the_title_the_model_made(client: TestClient, ask: FakeAsk, heading: str) -> None:
    ask.replies = [{"sections": [entry("a", subject="Egypt", topic="Burial", title="How the Dead Were Buried")]}]
    assert classify(client, [section("a", heading)])[0]["title"] == "How the Dead Were Buried"


def test_a_real_heading_is_never_given_a_title(client: TestClient, ask: FakeAsk) -> None:
    ask.replies = [{"sections": [entry("a", subject="Egypt", topic="Burial", title="Something Else Entirely")]}]
    assert classify(client, [section("a", "Mummification")])[0]["title"] == ""


def test_a_made_up_title_is_tidied(client: TestClient, ask: FakeAsk) -> None:
    ask.replies = [{"sections": [
        entry("a", title='Part 2: "The Judgment of Osiris"'),
        entry("b", title="3. Burial Customs"),
        entry("c", title="Part 4"),          # only another placeholder
        entry("d", title="x" * 300),
        entry("e", title=["not", "text"]),
    ]}]
    titles = [s["title"] for s in classify(client, [section(i, "Opening") for i in "abcde"])]
    assert titles == ["The Judgment of Osiris", "Burial Customs", "", "x" * 100, ""]


def test_names_lose_their_quotes_and_extra_spaces_and_are_capped(client: TestClient, ask: FakeAsk) -> None:
    ask.replies = [{"sections": [entry("a", subject=' "Ancient   Egyptian\nReligion" ', topic="T" * 90, topics=["  ", "Pharaoh’s Tomb"])]}]
    (one,) = classify(client, [section("a")])
    assert one["subject"] == "Ancient Egyptian Religion"
    assert one["topic"] == "T" * 60
    assert one["topics"] == ["T" * 60, "Pharaoh’s Tomb"]


# --- one spelling for one name --------------------------------------------------------


def test_a_known_subject_is_reused_whatever_the_case(client: TestClient, ask: FakeAsk) -> None:
    ask.replies = [{"sections": [
        entry("a", subject="ancient egyptian religion", topic="judgment", topics=["JUDGMENT", "burial"]),
        entry("b", subject="ANCIENT EGYPTIAN RELIGION", topic="Funerary Rites"),
    ]}]
    body = classify(client, [section("a"), section("b")], subjects=["Ancient Egyptian Religion"], topics=["Judgment"])
    assert [s["subject"] for s in body] == ["Ancient Egyptian Religion"] * 2
    assert body[0]["topic"] == "Judgment" and body[0]["topics"] == ["Judgment", "burial"]


def test_a_new_name_keeps_the_spelling_first_seen_in_the_call(client: TestClient, ask: FakeAsk) -> None:
    ask.replies = [{"sections": [
        entry("a", subject="Nile Agriculture", topic="Flood Season"),
        entry("b", subject="nile agriculture", topic="flood season", topics=["Irrigation"]),
        entry("c", subject="NILE AGRICULTURE", topic="Flood season"),
    ]}]
    body = classify(client, [section(i) for i in "abc"])
    assert {s["subject"] for s in body} == {"Nile Agriculture"}
    assert {s["topic"] for s in body} == {"Flood Season"}
    assert body[1]["topics"] == ["Flood Season", "Irrigation"]


def test_names_are_made_one_across_batches_in_section_order(client: TestClient, ask: FakeAsk, monkeypatch) -> None:
    monkeypatch.setattr(library, "SECTION_BATCH", 1)

    def answer(user: str) -> dict:
        # Each batch spells the names its own way; the first section's spelling is the one kept.
        if "[a]" in user:
            return {"sections": [entry("a", subject="Nile Agriculture", topic="Flood Season")]}
        return {"sections": [entry("b", subject="nile agriculture", topic="flood season")]}

    ask.replies = [answer, answer]
    body = classify(client, [section("a"), section("b")])
    assert len(ask.calls) == 2
    assert [(s["id"], s["subject"], s["topic"]) for s in body] == [
        ("a", "Nile Agriculture", "Flood Season"),
        ("b", "Nile Agriculture", "Flood Season"),
    ]


def test_a_topic_may_be_the_subject(client: TestClient, ask: FakeAsk) -> None:
    ask.replies = [{"sections": [entry("a", subject="Mummification", topic="mummification")]}]
    (one,) = classify(client, [section("a")])
    assert one["subject"] == "Mummification" and one["topic"] == "mummification"


def test_the_topic_leads_the_topics_and_they_are_three_at_most(client: TestClient, ask: FakeAsk) -> None:
    ask.replies = [{"sections": [entry("a", subject="Egypt", topic="Burial", topics=["Tombs", "Grave Goods", "burial", "Amulets"])]}]
    (one,) = classify(client, [section("a")])
    assert one["topic"] == "Burial" and one["topics"] == ["Burial", "Tombs", "Grave Goods"]


def test_an_answer_with_only_topics_uses_the_first_as_the_topic(client: TestClient, ask: FakeAsk) -> None:
    ask.replies = [{"sections": [{"id": "a", "kind": "exercise", "topics": ["Soil testing", "pH"]}]}]
    (one,) = classify(client, [section("a", "Try it yourself")])
    assert one == {"id": "a", "kind": "exercise", "subject": "", "topic": "Soil testing", "topics": ["Soil testing", "pH"], "title": ""}


def test_topics_that_are_not_a_list_are_ignored(client: TestClient, ask: FakeAsk) -> None:
    ask.replies = [{"sections": [entry("a", subject="Egypt", topics="Burial")]}]
    (one,) = classify(client, [section("a")])
    assert one["topic"] == "" and one["topics"] == []


# --- when there is no answer ----------------------------------------------------------


def test_a_failed_call_leaves_the_new_fields_empty(client: TestClient, ask: FakeAsk) -> None:
    ask.replies = [ServiceError("DeepSeek's reply could not be read. Try again.")]
    body = classify(client, [section("a", "Part 1"), section("b", "Practice problems")], subjects=["Egypt"])
    assert body == [
        {"id": "a", "kind": "text", "subject": "", "topic": "", "topics": [], "title": ""},
        {"id": "b", "kind": "exercise", "subject": "", "topic": "", "topics": [], "title": ""},
    ]


def test_a_section_the_model_skipped_leaves_the_new_fields_empty(client: TestClient, ask: FakeAsk) -> None:
    ask.replies = [{"sections": [entry("a", subject="Egypt", topic="Burial"), "junk", {"kind": "text"}]}]
    body = classify(client, [section("a"), section("b", "Opening")])
    assert (body[0]["subject"], body[0]["topic"]) == ("Egypt", "Burial")
    assert body[1] == {"id": "b", "kind": "text", "subject": "", "topic": "", "topics": [], "title": ""}


def test_one_failed_batch_does_not_spoil_the_other(client: TestClient, ask: FakeAsk, monkeypatch) -> None:
    monkeypatch.setattr(library, "SECTION_BATCH", 1)
    ask.replies = [
        lambda user: ServiceError("no") if "[a]" in user else {"sections": [entry("b", subject="Egypt", topic="Burial")]}
    ] * 2
    body = classify(client, [section("a"), section("b")])
    assert [(s["id"], s["subject"]) for s in body] == [("a", ""), ("b", "Egypt")]


def test_no_sections_asks_nothing(client: TestClient, ask: FakeAsk) -> None:
    assert classify(client, []) == []
    assert ask.calls == []


# --- the request ----------------------------------------------------------------------


def test_an_old_request_without_the_lists_is_still_valid() -> None:
    request = ClassifyRequest.model_validate({"sections": [{"id": "a", "title": "T", "text": "x"}]})
    assert request.subjects == [] and request.topics == []


def test_the_lists_are_trimmed_not_refused(client: TestClient, ask: FakeAsk) -> None:
    request = ClassifyRequest.model_validate({
        "subjects": [f"Subject {n}" for n in range(80)] + [None],
        "topics": ["  A   long   one " + "x" * 200, "", "   ", 7, None, "Short"],
        "sections": [],
    })
    assert len(request.subjects) == 60 and request.subjects[0] == "Subject 0"
    assert request.topics == ["A long one " + "x" * 69, "Short"]
    assert len(request.topics[0]) == 80
    # And over HTTP: too long is a 200, not a 422.
    ask.replies = [{"sections": []}]
    response = client.post("/users/7/sources/classify", json={
        "sections": [section("a")], "subjects": ["s" * 500] * 100, "topics": None,
    })
    assert response.status_code == 200, response.text


def test_the_wire_names_are_the_ones_the_app_reads() -> None:
    wire = SectionClass(id="a", kind="text", subject="S", topic="T", topics=["T"], title="X").model_dump(by_alias=True)
    assert list(wire) == ["id", "kind", "subject", "topic", "topics", "title"]
    request = ClassifyRequest.model_validate({"subjects": ["S"], "topics": ["T"]})
    assert list(request.model_dump(by_alias=True)) == ["pictures", "sections", "subjects", "topics"]


def test_library_sections_can_be_called_without_the_app(ask: FakeAsk) -> None:
    ask.replies = [{"sections": [entry("a", subject="Egypt", topic="Burial", title="Burial Customs")]}]
    request = ClassifyRequest.model_validate({"sections": [{"id": "a", "title": "Part 1", "text": "x"}], "subjects": ["egypt"]})
    (found,) = asyncio.run(library.sections(request))
    assert (found.subject, found.topic, found.title) == ("egypt", "Burial", "Burial Customs")
    assert json.loads(found.model_dump_json())["topics"] == ["Burial"]
