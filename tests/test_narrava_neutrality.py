"""Narrava Studio: the narration writer passes no verdict on the subject in its own voice.

The writer's half of the conversation's neutrality (test_narrava_converse.py). Only the LLM
boundary is stubbed, so these run without an API key.
"""
from __future__ import annotations

import pytest

from phansora.products.narrava_studio.services import neutrality, script

# The opening a documentary on Pizzagate was written with: the narrator settling it.
JUDGING = (
    "He walked into the pizza place with a rifle. The emails were real. "
    "A basement that didn't exist. A conspiracy that did."
)
NEUTRAL = (
    "In 2016, posts online claimed a Washington pizzeria hid a child-abuse ring. "
    "Believers pointed to code words in leaked emails. The restaurant's owner denied it. "
    "Critics called the theory a hoax."
)


@pytest.fixture
def writer(monkeypatch):
    """The writer, answering each call with the next of ``replies``; and its reviewer, with
    the next of ``reviews`` (by default, finding nothing)."""
    state = {"replies": [], "seen": [], "reviews": [], "reviewed": []}

    def next_of(replies, default=None):
        reply = replies.pop(0) if replies else default
        if isinstance(reply, Exception):
            raise reply
        return reply

    def write(system, user, **kwargs):
        state["seen"].append({"system": system, "user": user})
        return next_of(state["replies"])

    def review(system, user, **kwargs):
        state["reviewed"].append(user)
        return next_of(state["reviews"], {"opening": "other", "sentences": []})

    monkeypatch.setattr(script.llm, "generate_text", write)
    monkeypatch.setattr(script.llm, "deepseek_json", review)
    return state


def test_a_verdict_in_the_narrators_voice_is_found():
    assert neutrality.judging_sentences(JUDGING, "A documentary about Pizzagate") == ["A conspiracy that did."]


def test_a_verdict_that_says_whose_it_is_is_reporting():
    assert neutrality.judging_sentences(NEUTRAL, "A documentary about Pizzagate") == []
    assert neutrality.judging_sentences('Police: "It was a hoax."', "") == []


def test_their_own_framing_may_be_echoed():
    assert neutrality.judging_sentences(JUDGING, "How the Pizzagate conspiracy theory spread") == []


def test_words_that_are_not_verdicts_are_left_alone():
    text = "The town lies in a valley. He lived on the fringe of the city. He was charged with conspiracy."
    assert neutrality.judging_sentences(text, "") == []


def test_the_writer_is_told_to_stay_neutral(writer):
    writer["replies"] = [NEUTRAL]
    script.generate_script("A documentary about Pizzagate")
    assert "Stay neutral about real people and events" in writer["seen"][0]["system"]
    assert "the first lines are the claim itself" in writer["seen"][0]["system"]


def test_a_narration_that_judges_is_written_again(writer):
    writer["replies"] = [JUDGING, NEUTRAL]
    out = script.generate_script("A documentary about Pizzagate")
    assert len(writer["seen"]) == 2
    assert "A conspiracy that did." in writer["seen"][1]["user"]
    assert out.full_text == NEUTRAL


def test_a_neutral_narration_is_written_once(writer):
    writer["replies"] = [NEUTRAL]
    out = script.generate_script("A documentary about Pizzagate")
    assert len(writer["seen"]) == 1
    assert out.full_text == NEUTRAL


@pytest.mark.parametrize("redraft", [JUDGING, "A conspiracy.", RuntimeError("timeout")])
def test_the_first_draft_stands_when_the_redraft_is_no_better(writer, redraft):
    writer["replies"] = [JUDGING, redraft]
    out = script.generate_script("A documentary about Pizzagate")
    assert out.full_text == JUDGING


def test_a_verdict_without_a_verdict_word_is_found_by_the_review(writer):
    draft = "In 2016, posts claimed a pizzeria hid a crime. No such ring was found."
    writer["replies"] = [draft, NEUTRAL]
    writer["reviews"] = [{"opening": "claim", "sentences": ["No such ring was found."]}]
    out = script.generate_script("A documentary about Pizzagate")
    assert "- No such ring was found." in writer["seen"][1]["user"]
    assert "The brief:\nA documentary about Pizzagate" in writer["reviewed"][0]
    assert out.full_text == NEUTRAL


def test_an_opening_on_the_aftermath_is_written_again(writer):
    draft = "He walked into the pizza place with a rifle. Posts had claimed it hid a crime."
    writer["replies"] = [draft, NEUTRAL]
    writer["reviews"] = [{"opening": "aftermath", "sentences": []}]
    out = script.generate_script("A documentary about Pizzagate")
    assert "Open on the claim itself" in writer["seen"][1]["user"]
    assert out.full_text == NEUTRAL


def test_a_sentence_the_reviewer_misquoted_is_not_a_fault(writer):
    writer["replies"] = [NEUTRAL]
    writer["reviews"] = [{"opening": "claim", "sentences": ["Nothing like this is in the narration."]}]
    script.generate_script("A documentary about Pizzagate")
    assert len(writer["seen"]) == 1


def test_without_the_review_the_words_are_still_checked(writer):
    writer["replies"] = [JUDGING, NEUTRAL]
    writer["reviews"] = [RuntimeError("no DeepSeek"), RuntimeError("no DeepSeek")]
    out = script.generate_script("A documentary about Pizzagate")
    assert out.full_text == NEUTRAL


def test_a_story_is_not_checked(writer):
    writer["replies"] = ["Everything she told him was lies."]
    out = script.generate_script("A fable about a fox", style="story")
    assert len(writer["seen"]) == 1
    assert writer["reviewed"] == []
    assert out.full_text == "Everything she told him was lies."


def test_a_document_imported_frames_it_too(writer, monkeypatch):
    """Words the imported document used are its framing, not the writer's verdict."""
    writer["replies"] = ["It was a hoax from the start."]
    script.from_chapters("Hoaxes of the Century", [{"title": "The pizzeria hoax", "summary": "How it began."}])
    assert len(writer["seen"]) == 1
