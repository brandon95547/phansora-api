r"""Salvaging JSON the model wrote while quoting messy source.

Project 49 — a Coptic lexicon read by OCR — died in the curriculum phase with
"Could not parse JSON from DeepSeek response". The response was not malformed by the
model's own doing: the source contains literal backslashes, and DeepSeek copied one
into a string it was quoting (``'-- EBOI\ ZN-'``). ``\ `` is not a JSON escape, so
json.loads rejected the whole 33KB plan over one character, and 19 chunks of the same
book had already been skipped the same way earlier in the run.

These pin that a stray backslash costs nothing, and that nothing valid was traded for it.
"""
from __future__ import annotations

import json

import pytest

from phansora.shared.ai.json_repair import (
    escape_stray_backslashes,
    parse_json_loose,
    repair_truncated_json,
)


# ── the failure that lost the book ──────────────────────────────────────────

def test_a_literal_backslash_in_quoted_source_no_longer_fails_the_parse():
    raw = r'{"teachable": true, "concepts": [{"title": "t", "body": "prepositions: EBOI\ ZN-"}]}'
    assert parse_json_loose(raw)["concepts"][0]["body"] == r"prepositions: EBOI\ ZN-"


def test_a_response_that_is_both_truncated_and_stray_escaped_still_salvages():
    # The prod shape exactly: finish_reason=="length" AND a bad escape before the cut.
    raw = (
        '{"work_title": "Nag Hammadi Codices",\n "sessions": [\n'
        r'  {"ordinal": 1, "start_segment": 0, "topics": ["elA)\.IIA)\; elMe"]},' "\n"
        '  {"ordinal": 2, "start_segment": 6, "title": "Descr'
    )
    with pytest.raises(ValueError):
        parse_json_loose(raw)          # not valid on its own — it was cut off
    plan = parse_json_loose(repair_truncated_json(raw))
    assert plan["work_title"] == "Nag Hammadi Codices"
    assert plan["sessions"][0]["start_segment"] == 0
    assert plan["sessions"][0]["topics"] == [r"elA)\.IIA)\; elMe"]
    # The cut-off session comes back as an empty stub, which _entries_from_plan
    # drops for having no start_segment — a lesson boundary is never guessed.
    assert plan["sessions"][-1] == {}


def test_a_raw_newline_inside_a_string_is_tolerated_too():
    assert parse_json_loose('{"body": "line one\nline two"}')["body"] == "line one\nline two"


# ── and nothing valid was traded away ───────────────────────────────────────

@pytest.mark.parametrize("doc", [
    {"body": "a\nb"},                       # a real \n escape
    {"body": 'she said "hi"'},              # a real \" escape
    {"path": r"C:\Users\book"},             # a real \\ escape
    {"body": "\u00e9\u2014"},               # a real \uXXXX escape
    {"body": "a/b"},
    {"n": [1, 2.5, -3], "t": True, "z": None},
])
def test_valid_json_round_trips_unchanged(doc):
    assert parse_json_loose(json.dumps(doc)) == doc


def test_text_that_already_parses_is_never_rewritten():
    raw = json.dumps({"path": r"C:\Users", "q": 'say "hi"', "nl": "a\nb"})
    assert escape_stray_backslashes(raw) == raw


def test_fences_and_prose_are_still_stripped():
    assert parse_json_loose('Here you go:\n```json\n{"a": 1}\n```') == {"a": 1}


def test_an_empty_response_is_still_an_error():
    with pytest.raises(ValueError):
        parse_json_loose("   ")
