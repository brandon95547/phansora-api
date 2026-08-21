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
