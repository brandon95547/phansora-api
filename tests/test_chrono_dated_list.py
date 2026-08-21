"""Reading the research answer onto the timeline.

This is the step that replaced a model call, so what it must not do is what the model
did: shorten the list, drop an item it finds odd, or invent one. Everything shaped like
an item is kept; everything else is left alone.
"""
from __future__ import annotations

import pytest

from phansora.products.chrono_origin.pipeline.dated_list import parse_dated_list, parse_date


# The shape prod actually returns, taken from a live answer.
LIVE_ANSWER = """\
Cuneiform Script - c. 3400 BCE
Egyptian Hieroglyphs - c. 3200 BCE
Proto-Sinaitic Script - c. 1800 BCE
Code of Hammurabi - c. 1750 BCE
Mesha Stele - c. 840 BCE
Codex Sinaiticus - c. 350 CE
Wycliffe Bible - c. 1382 CE
Gutenberg Bible - c. 1455 CE
King James Bible - c. 1611 CE
"""


class TestTheListSurvivesIntact:
    def test_every_item_is_kept(self):
        assert len(parse_dated_list(LIVE_ANSWER)) == 9

    def test_the_order_given_is_the_order_returned(self):
        titles = [i.title for i in parse_dated_list(LIVE_ANSWER)]
        assert titles[0] == "Cuneiform Script"
        assert titles[-1] == "King James Bible"

    def test_a_hyphenated_title_is_not_cut_in_half(self):
        """Proto-Sinaitic, Al-Andalus, Sub-Saharan. Splitting on a bare "-" loses them."""
        items = parse_dated_list("Proto-Sinaitic Script - c. 1800 BCE")
        assert items[0].title == "Proto-Sinaitic Script"
        assert items[0].year == -1800

    def test_an_em_dash_separator_also_works(self):
        assert parse_dated_list("Codex Sinaiticus — c. 350 CE")[0].title == "Codex Sinaiticus"

    def test_bullets_and_numbering_are_stripped(self):
        """The prompt asks for neither. Models add them anyway."""
        text = "- Cuneiform Script - c. 3400 BCE\n2. Mesha Stele - c. 840 BCE"
        assert [i.title for i in parse_dated_list(text)] == ["Cuneiform Script", "Mesha Stele"]

    def test_bold_markers_are_stripped(self):
        assert parse_dated_list("**Mesha Stele** - c. 840 BCE")[0].title == "Mesha Stele"

    def test_the_same_item_twice_is_kept_once(self):
        text = "Mesha Stele - c. 840 BCE\nmesha stele - c. 840 BCE"
        assert len(parse_dated_list(text)) == 1


class TestWhatIsNotAnItem:
    def test_prose_without_a_separator_is_skipped(self):
        text = "Here is the chronological list you asked for:\nMesha Stele - c. 840 BCE"
        assert [i.title for i in parse_dated_list(text)] == ["Mesha Stele"]

    def test_a_sentence_containing_a_dash_is_not_an_item(self):
        """A separator alone does not make a line an item — something date-shaped has
        to be on the right of it."""
        text = ("The Septuagint - a Greek translation made in Alexandria - remains the "
                "principal witness to a Hebrew text older than the Masoretic tradition "
                "and is cited throughout the New Testament writings.")
        assert parse_dated_list(text) == []

    def test_an_empty_answer_yields_nothing(self):
        assert parse_dated_list("") == []
        assert parse_dated_list("   \n\n  ") == []


class TestDates:
    @pytest.mark.parametrize("text,expected", [
        ("c. 3400 BCE", -3400),
        ("3400 BC", -3400),
        ("c. 1611 CE", 1611),
        ("1611 AD", 1611),
        ("1611", 1611),
    ])
    def test_a_year_lands_on_the_right_side_of_zero(self, text, expected):
        assert parse_date(text)[0] == expected

    def test_a_century_becomes_the_span_it_is(self):
        """"3rd century BCE" is a hundred years. Pinning it to one would be a claim the
        answer never made."""
        assert parse_date("3rd century BCE")[:3] == (-300, -201, "century")
        assert parse_date("4th century CE")[:3] == (301, 400, "century")

    def test_a_millennium_becomes_a_span_too(self):
        assert parse_date("3rd millennium BCE")[:3] == (-3000, -2001, "millennium")

    def test_a_range_keeps_both_ends_and_starts_with_the_earlier(self):
        """A BCE range counts DOWN — 1400-1200 BCE — so ordering by value keeps the
        earlier end first whichever era it is in."""
        assert parse_date("c. 1400-1200 BCE")[:2] == (-1400, -1200)
        assert parse_date("1400–1200 BCE")[:2] == (-1400, -1200)
        assert parse_date("1500 to 1600 CE")[:2] == (1500, 1600)

    def test_an_era_with_no_number_keeps_its_words(self):
        """"Bronze Age" is a real position on a timeline. Dropping it would drop the item."""
        year, _, precision, label = parse_date("Late Bronze Age")
        assert year is None and precision == "era" and label == "Late Bronze Age"

    def test_present_day_is_an_era_not_a_year(self):
        assert parse_date("present day")[3] == "present day"

    def test_a_dateless_line_is_still_an_item(self):
        items = parse_dated_list("Modern critical editions - present day")
        assert items[0].title == "Modern critical editions"
        assert items[0].year is None and items[0].era_label == "present day"


# --------------------------------------------------------- the subject reaches the model
def test_a_prompt_that_lost_its_title_placeholder_fails_loudly():
    """The failure this prevents is silent, which is what makes it expensive.

    `str.format()` ignores a keyword the template does not use. A RESEARCH_PROMPT with
    the subject hardcoded — easily done while tuning against one subject — therefore
    raises nothing, and every trace of every subject researches that one subject with
    nothing anywhere reporting it.
    """
    import pytest

    from phansora.products.chrono_origin.pipeline import orchestrator as orch

    original = orch.RESEARCH_PROMPT
    orch.RESEARCH_PROMPT = 'Trace the lineage of "Jesus Christ" from the beginning.'
    try:
        with pytest.raises(RuntimeError, match="never mentions"):
            orch.build_research_prompt("Mount Everest", "")
    finally:
        orch.RESEARCH_PROMPT = original


def test_the_shipped_prompt_carries_the_subject():
    from phansora.products.chrono_origin.pipeline.orchestrator import build_research_prompt

    assert "Mount Everest" in build_research_prompt("Mount Everest", "")


def test_a_prompt_containing_json_braces_still_renders():
    """The output shape moved to a JSON array, and every brace in the example read as a
    format field: the template died on `KeyError: '\\n    "title"'` before it reached the
    model. Substitution has no opinion about braces."""
    from phansora.products.chrono_origin.pipeline import orchestrator as orch

    original = orch.RESEARCH_PROMPT
    orch.RESEARCH_PROMPT = 'Trace "{title}". Return [{"title": "x", "date": "y"}] and nothing else.'
    try:
        out = orch.build_research_prompt("Mount Everest", "")
        assert "Mount Everest" in out
        assert '[{"title": "x", "date": "y"}]' in out
    finally:
        orch.RESEARCH_PROMPT = original


# ------------------------------------------------------------------ the JSON array
# What the prompt asks for now. It is asked for in the PROMPT, not enforced with
# responseMimeType, because the API refuses a forced JSON mime type alongside the search
# tool — so the answer arrives however the model felt like sending it, and reading past
# a fence or a stray sentence is cheaper than failing a trace that was researched fine.
FENCED = """Here is the chronological list you asked for.

```json
[
  {"title": "Cuneiform Script", "date": "c. 3400 BCE", "origin": "Mesopotamia",
   "material": "Clay tablets; Sumerian", "authorship": "Sumerian scribes",
   "significance": "Earliest known writing system."},
  {"title": "Dead Sea Scrolls", "date": "3rd century BCE", "origin": "Qumran",
   "material": "Parchment; Hebrew", "authorship": "", "significance": "Oldest witnesses."}
]
```

Let me know if you would like more detail."""


class TestTheJsonArray:
    def test_a_fenced_array_wrapped_in_prose_is_read(self):
        items = parse_dated_list(FENCED)
        assert [i.title for i in items] == ["Cuneiform Script", "Dead Sea Scrolls"]

    def test_dates_are_parsed_the_same_way_as_ever(self):
        items = parse_dated_list(FENCED)
        assert items[0].year == -3400
        assert (items[1].year, items[1].year_end, items[1].precision) == (-300, -201, "century")

    def test_the_metadata_arrives_labelled_and_in_order(self):
        """"Mesopotamia" and "clay tablet" answer different questions; run together into
        one string they answer neither, and nothing downstream can lay them out."""
        assert parse_dated_list(FENCED)[0].details == (
            ("Origin", "Mesopotamia"),
            ("Material", "Clay tablets; Sumerian"),
            ("Authorship", "Sumerian scribes"),
            ("Significance", "Earliest known writing system."),
        )

    def test_a_value_written_across_lines_is_flattened(self):
        """A line break inside a value breaks every consumer that shows these as rows,
        and nothing in a field this size needs one."""
        items = parse_dated_list('[{"title": "A", "date": "1 CE", "origin": "Rome\nand Ostia"}]')
        assert items[0].details == (("Origin", "Rome and Ostia"),)

    def test_an_empty_field_is_left_out_not_shown_as_a_blank(self):
        """The prompt tells the model to leave a field empty rather than guess. Printing
        the gap back would undo the point of asking."""
        labels = [label for label, _ in parse_dated_list(FENCED)[1].details]
        assert "Authorship" not in labels
        assert labels[0] == "Origin"

    def test_the_keys_are_read_leniently(self):
        """A trace should not fail because the model wrote "name" for "title"."""
        items = parse_dated_list('[{"name": "King James Bible", "date_range": "1611 CE"}]')
        assert items[0].title == "King James Bible" and items[0].year == 1611

    def test_an_object_with_no_title_is_skipped(self):
        items = parse_dated_list('[{"date": "1611 CE"}, {"title": "Real", "date": "1612 CE"}]')
        assert [i.title for i in items] == ["Real"]

    def test_the_same_title_twice_is_kept_once(self):
        items = parse_dated_list('[{"title": "A", "date": "1 CE"}, {"title": "a", "date": "1 CE"}]')
        assert len(items) == 1

    def test_an_item_with_no_metadata_simply_has_none(self):
        items = parse_dated_list('[{"title": "A", "date": "1611 CE"}]')
        assert items[0].details == ()


class TestTheOlderShapeStillReads:
    """The prompt is retuned by hand and often. A trace should not die because the output
    shape moved back."""

    def test_lines_are_read_when_there_is_no_array(self):
        items = parse_dated_list("Cuneiform Script - c. 3400 BCE\nMesha Stele - c. 840 BCE")
        assert [i.title for i in items] == ["Cuneiform Script", "Mesha Stele"]

    def test_malformed_json_falls_back_to_lines_rather_than_failing(self):
        text = "[{'title': not json at all\nCuneiform Script - c. 3400 BCE"
        assert [i.title for i in parse_dated_list(text)] == ["Cuneiform Script"]

    def test_an_array_of_strings_is_not_mistaken_for_items(self):
        assert parse_dated_list('["Cuneiform Script", "Mesha Stele"]') == []


class TestATruncatedAnswerKeepsWhatItGot:
    """Six fields per item makes running out of output budget the expected failure, and
    an array cut off mid-object is not valid JSON. Losing thirty researched items
    because the thirty-first was cut in half is a bug, not a safeguard."""

    TRUNCATED = (
        '[\n'
        '  {"title": "Cuneiform Script", "date": "c. 3400 BCE",'
        '   "significance": "Contains a } brace, as prose does"},\n'
        '  {"title": "Mesha Stele", "date": "c. 840 BCE"},\n'
        '  {"title": "King James Bib'
    )

    def test_the_complete_items_survive(self):
        items = parse_dated_list(self.TRUNCATED)
        assert [i.title for i in items] == ["Cuneiform Script", "Mesha Stele"]

    def test_the_half_written_item_is_dropped_not_guessed_at(self):
        assert "King James" not in [i.title for i in parse_dated_list(self.TRUNCATED)]

    def test_a_brace_inside_prose_is_not_mistaken_for_structure(self):
        items = parse_dated_list(self.TRUNCATED)
        assert items[0].details == (("Significance", "Contains a } brace, as prose does"),)

    def test_it_says_so_in_the_log(self, caplog):
        """Otherwise a list quietly gets shorter and nothing reports why."""
        import logging

        with caplog.at_level(logging.WARNING):
            parse_dated_list(self.TRUNCATED)
        assert any("GEMINI_SEARCH_MAX_TOKENS" in r.message for r in caplog.records)
