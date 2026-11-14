"""The seven kinds of research on the start screen, each asking its own question.

The failure these pin is the one the screen was most likely to ship with: a card per kind
over one trace prompt, so that Chronology, Parallels, Translations and Evidence all came
back as the same lineage under different headings.
"""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from phansora.products.chrono_origin.models import ResearchSpec, TraceRequest
from phansora.products.chrono_origin.pipeline import orchestrator as orch
from phansora.products.chrono_origin.pipeline import research_modes as rm
from phansora.products.chrono_origin.pipeline.dated_list import parse_dated_list
from phansora.products.chrono_origin.services.cache import request_key
from phansora.shared.ai.research import GroundedAnswer


class Client:
    def __init__(self, items):
        self.searches = []
        self._text = json.dumps(items)

    def reason_json(self, prompt, use_reasoning_model=False):  # pragma: no cover
        raise AssertionError("a trace makes no JSON call")

    def grounded_search(self, prompt):
        self.searches.append(prompt)
        return GroundedAnswer(text=self._text, citations=[], queries=["q"])


@pytest.fixture
def run(monkeypatch):
    monkeypatch.setattr(orch, "get_cached", lambda *a, **k: None)
    monkeypatch.setattr(orch, "save_cached", lambda *a, **k: None)

    def go(items, title="The Flood", research=None, context=None):
        client = Client(items)
        req = TraceRequest(title=title, context=context,
                           research=ResearchSpec(**research) if research else None)
        return orch.TraceOrchestrator(client=client).run(req), client

    return go


TWO = [{"title": "A", "date": "c. 2000 BCE"}, {"title": "B", "date": "c. 1000 BCE"}]


# ------------------------------------------------------------------ each its own question
class TestEachKindAsksItsOwnQuestion:
    def test_a_plain_trace_still_sends_the_trace_prompt(self, run):
        _, client = run(TWO)
        assert client.searches[0].startswith("Using live web search, trace the complete historical lineage")

    def test_origins_asks_how_it_came_to_be_and_leaves_older_versions_out(self, run):
        _, plain = run(TWO)
        _, origins = run(TWO, research={"kind": "origins"})
        prompt = origins.searches[0]
        assert prompt.startswith(plain.searches[0].rstrip())
        assert "how did" in prompt and "come to be?" in prompt
        assert "Origin myth:" in prompt
        assert "Older versions of the same story" in prompt and "leave them out" in prompt
        assert "how did" not in plain.searches[0]

    def test_origins_appends_its_focus_and_perspectives(self, run):
        _, client = run(TWO, research={"kind": "origins", "focus": "words",
                                       "perspectives": ["linguistic", "textual"]})
        prompt = client.searches[0]
        assert prompt.startswith("Using live web search, trace the complete historical lineage")
        assert "etymology" in prompt
        assert "Linguistic: words, names and their roots" in prompt
        assert "Textual:" in prompt

    @pytest.mark.parametrize("kind,marker", [
        ("chronology", "build the chronology of"),
        ("genealogy", "trace the genealogy of"),
        ("interpretations", "has been read through this interpretation"),
        ("parallels", "find the parallels to"),
        ("translations", "is worded across languages and versions"),
        ("establishment", "map the power structure of"),
    ])
    def test_every_other_kind_sends_a_prompt_of_its_own(self, run, kind, marker):
        _, client = run(TWO, research={"kind": kind})
        prompt = client.searches[0]
        assert marker in prompt
        assert "trace the complete historical lineage" not in prompt
        assert '"The Flood"' in prompt

    def test_the_context_reaches_every_kind(self, run):
        for kind in ("chronology", "genealogy", "interpretations", "parallels", "translations", "establishment"):
            _, client = run(TWO, research={"kind": kind}, context="Mesopotamian myth")
            assert "Mesopotamian myth" in client.searches[0], kind


# ------------------------------------------------------------- correlated across sources
class TestEveryKindCorrelatesItsSources:
    """One unified result per kind, not a list per source, with its agreement reported."""

    @pytest.mark.parametrize("research,unified", [
        ({"kind": "chronology", "perspectives": ["narrative"], "sequence": "narrative"},
         "one timeline of the subject, not a list per source"),
        ({"kind": "genealogy", "focus": "families"}, "Build ONE tree"),
        ({"kind": "origins"}, "Reconstruct ONE path"),
        ({"kind": "interpretations", "focus": "numerical"}, "give each reading its own entry"),
        ({"kind": "parallels", "focus": "stories"}, "one entry naming them all"),
        ({"kind": "translations", "focus": "wording"}, "identical wording are one entry"),
        ({"kind": "establishment", "focus": "religions"}, "ONE structure"),
    ])
    def test_each_kind_merges_its_sources_into_one_result(self, run, research, unified):
        _, client = run(TWO, research=research)
        prompt = client.searches[0]
        assert "CORRELATE ACROSS SOURCES" in prompt
        assert unified in prompt
        for field in ('"sources"', '"agreement"', '"conflicts"'):
            assert field in prompt, (research["kind"], field)

    def test_a_plain_trace_is_left_as_it_was(self, run):
        _, client = run(TWO)
        assert "CORRELATE ACROSS SOURCES" not in client.searches[0]

    def test_the_correlation_becomes_rows_on_the_card(self):
        text = json.dumps([{"title": "David anointed king", "date": "c. 1000 BCE",
                            "sources": "2 Samuel 5:1-5; 1 Chronicles 11:1-3",
                            "agreement": "differ", "conflicts": "Chronicles omits the seven years at Hebron"}])
        details = parse_dated_list(text)[0].details
        assert ("Sources", "2 Samuel 5:1-5; 1 Chronicles 11:1-3") in details
        assert ("Agreement", "Differ") in details
        assert ("Conflicts", "Chronicles omits the seven years at Hebron") in details
        single = parse_dated_list(json.dumps([{"title": "X", "date": "1", "agreement": "single source"}]))[0]
        assert ("Agreement", "Single source") in single.details


class TestChronology:
    def test_each_sequence_is_its_own_instruction(self):
        spec = lambda seq: ResearchSpec(kind="chronology", perspectives=["historical"], sequence=seq)
        assert "NARRATIVE ORDER" in rm.chronology_prompt("X", None, spec("narrative"))
        narrative = rm.chronology_prompt("X", None, spec("narrative"))
        assert "Reconstruct ONE sequence of events from every account" in narrative
        assert "align the parallel passages" in narrative
        assert "where that evidence converges" in rm.chronology_prompt("X", None, spec("historical"))
        assert "Mark its uncertainty" in rm.chronology_prompt("X", None, spec("historical"))
        # The order of an interpretation's own stages left with it.
        with pytest.raises(ValidationError):
            spec("symbolic")

    def test_only_the_chosen_perspective_is_asked_for(self):
        prompt = rm.chronology_prompt("X", None, ResearchSpec(
            kind="chronology", perspectives=["narrative"]))
        assert "Narrative:" in prompt
        assert "Evidence:" not in prompt

    def test_earlier_versions_are_gone(self):
        # Older tellings of a story by other cultures are Parallels' to find.
        with pytest.raises(ValidationError):
            ResearchSpec(kind="chronology", perspectives=["comparative"])

    def test_the_readings_are_no_longer_a_chronology_lens(self):
        # They are Interpretations' focus now; a chronology asking for one is refused.
        with pytest.raises(ValidationError):
            ResearchSpec(kind="chronology", perspectives=["solar_astronomical"])

    def test_a_chronology_takes_one_perspective(self):
        with pytest.raises(ValidationError):
            ResearchSpec(kind="chronology", perspectives=["narrative", "historical"])
        # Other kinds still search through several.
        assert len(ResearchSpec(kind="origins", perspectives=["textual", "linguistic"]).perspectives) == 2

    def test_a_layout_from_an_older_request_is_ignored(self):
        spec = ResearchSpec(kind="chronology", perspectives=["narrative"], layout="grouped")
        assert not hasattr(spec, "layout")
        prompt = rm.chronology_prompt("X", None, spec)
        assert '"perspective"' not in prompt and '"readings"' not in prompt

    def test_a_chronology_asks_for_no_intent(self):
        prompt = rm.chronology_prompt("X", None, ResearchSpec(
            kind="chronology", perspectives=["narrative"]))
        assert "INTERPRETATION" not in prompt and '"intent"' not in prompt

    def test_interpretations_are_reported_not_asserted(self):
        prompt = rm.chronology_prompt("X", None, ResearchSpec(kind="chronology"))
        assert "do not present an interpretation as historical fact" in prompt

    def test_a_narrative_order_is_kept_as_given(self, run):
        items = [
            {"title": "Later in the story", "stage": "1", "date": "c. 500 CE"},
            {"title": "Earlier in the story", "stage": "2", "date": "c. 900 BCE"},
            {"title": "End", "stage": "3", "date": ""},
        ]
        trace, _ = run(items, research={"kind": "chronology", "perspectives": ["narrative"],
                                        "sequence": "narrative"})
        assert trace.ordering == "given"
        order = [trace.origin.source_title] + [e.source_title for e in trace.timeline]
        assert order == ["Later in the story", "Earlier in the story", "End"]
        # Labelled by its place in the sequence; the date is kept, in the details.
        assert trace.origin.year is None and trace.origin.era_label == "1"
        assert {"label": "Date", "value": "c. 500 CE"} in [d.model_dump() for d in trace.origin.details]

    def test_a_historical_order_is_sorted_by_date(self, run):
        items = [{"title": "Late", "date": "1200 CE"}, {"title": "Early", "date": "500 BCE"}]
        trace, _ = run(items, research={"kind": "chronology", "sequence": "historical"})
        assert trace.ordering == "date"
        assert trace.origin.source_title == "Early"


class TestInterpretations:
    def test_the_reading_chosen_is_the_one_asked_for(self):
        prompt = rm.interpretations_prompt("Revelation 13", None, ResearchSpec(
            kind="interpretations", focus="numerical"))
        assert "Symbolic & Numerical:" in prompt
        assert "Political & Coded History:" not in prompt

    def test_every_entry_is_graded_by_the_writers_intent(self):
        prompt = rm.interpretations_prompt("X", None, ResearchSpec(
            kind="interpretations", focus="coded_political"))
        assert "whether the people who made the subject meant it this way" in prompt
        assert '"intent"' in prompt and "no tradition" in prompt
        assert "Political & Coded History" in prompt.split("THIS IS AN INTERPRETATION")[1]

    def test_a_subject_with_no_sequence_goes_element_by_element(self):
        prompt = rm.interpretations_prompt("666", None, ResearchSpec(
            kind="interpretations", focus="numerical"))
        assert "element by element where it has none" in prompt
        assert '"element"' in prompt and '"reading"' in prompt

    def test_the_element_reading_and_intent_become_rows_on_the_card(self):
        text = json.dumps([{"title": "Babylon as Rome", "date": "c. 95 CE", "element": "Babylon",
                            "reading": "Rome", "intent": "intended"}])
        details = parse_dated_list(text)[0].details
        assert ("Element", "Babylon") in details and ("Read as", "Rome") in details
        assert ("Writers' intent", "Intended") in details

    def test_its_entries_keep_the_order_given(self, run):
        items = [{"title": "Spring", "stage": "1", "date": "1890s"},
                 {"title": "Winter", "stage": "2", "date": "c. 200 CE"}]
        trace, _ = run(items, research={"kind": "interpretations", "focus": "ritual"})
        assert trace.ordering == "given"
        assert [trace.origin.source_title] + [e.source_title for e in trace.timeline] == ["Spring", "Winter"]

    def test_a_reading_is_a_focus_of_interpretations_only(self):
        with pytest.raises(ValidationError):
            ResearchSpec(kind="interpretations", perspectives=["numerical"])


class TestParallels:
    def test_one_subject_is_searched_for_parallels(self):
        prompt = rm.parallels_prompt("Osiris", None, ResearchSpec(kind="parallels", focus="people"))
        assert 'find the parallels to "Osiris"' in prompt
        assert "people and figures" in prompt

    def test_several_subjects_are_compared_with_each_other(self):
        prompt = rm.parallels_prompt("Osiris · Tammuz", None, ResearchSpec(
            kind="parallels", focus="stories", subjects=["Osiris", "Tammuz"]))
        assert 'compare "Osiris" and "Tammuz"' in prompt
        assert "EACH SUBJECT ITSELF" in prompt

    def test_similarities_and_differences_reach_the_card(self):
        text = json.dumps([{"title": "Dying and rising", "date": "c. 2400 BCE",
                            "similarities": "Death and return", "differences": "Seasonal",
                            "relation": "independent_parallel"}])
        item = parse_dated_list(text)[0]
        assert ("Similarities", "Death and return") in item.details
        assert ("Differences", "Seasonal") in item.details
        assert ("Relation", "Independent parallel") in item.details


class TestTranslations:
    def test_the_languages_versions_and_focus_are_all_asked_for(self):
        prompt = rm.translations_prompt("In the beginning", None, ResearchSpec(
            kind="translations", focus="manuscripts", languages=["Hebrew", "Greek"],
            versions=["Septuagint", "Vulgate"]))
        assert "MANUSCRIPT DIFFERENCES" in prompt
        assert "Hebrew and Greek" in prompt
        assert "Septuagint and Vulgate" in prompt

    def test_quotation_is_kept_short(self):
        prompt = rm.translations_prompt("X", None, ResearchSpec(kind="translations"))
        assert "at most about 25 words" in prompt


class TestEstablishment:
    def test_the_kind_of_body_decides_what_is_searched(self):
        spec = lambda focus: ResearchSpec(kind="establishment", focus=focus)
        assert "dioceses" in rm.establishment_prompt("X", None, spec("religions"))
        assert "lodges" in rm.establishment_prompt("X", None, spec("societies"))
        assert "legislature" in rm.establishment_prompt("X", None, spec("governments"))
        assert "subsidiaries" in rm.establishment_prompt("X", None, spec("companies"))
        assert "youth" in rm.establishment_prompt("X", None, spec("movements"))
        # The top is named in the words that kind of body uses.
        assert "its supreme authority, then every level" in rm.establishment_prompt("X", None, spec("religions"))
        assert "its head of state or highest authority, then" in rm.establishment_prompt("X", None, spec("governments"))

    def test_it_maps_the_official_structure_and_everything_outside_it(self):
        prompt = rm.establishment_prompt("The Catholic Church", None, ResearchSpec(kind="establishment", focus="religions"))
        assert "THE OFFICIAL STRUCTURE, from the top down" in prompt
        assert "EVERYTHING OUTSIDE THE OFFICIAL STRUCTURE" in prompt
        assert "splinter groups, breakaway factions" in prompt
        assert "Do not take a side" in prompt
        assert '"parent_body"' in prompt and '"standing"' in prompt

    def test_the_structure_is_kept_top_down(self, run):
        items = [{"title": "Pope", "date": "c. 33 CE", "standing": "official"},
                 {"title": "College of Bishops", "date": "", "parent_body": "Pope"},
                 {"title": "Society of Saint Pius X", "date": "1970", "parent_body": "Pope",
                  "standing": "Irregular"}]
        trace, _ = run(items, title="The Catholic Church",
                       research={"kind": "establishment", "focus": "religions"})
        assert trace.ordering == "given"
        assert trace.origin.source_title == "Pope"
        assert trace.origin.standing == "official"
        assert [e.source_title for e in trace.timeline] == ["College of Bishops", "Society of Saint Pius X"]
        assert [e.parent_body for e in trace.timeline] == ["Pope", "Pope"]
        assert [e.standing for e in trace.timeline] == ["", "disputed"]

    def test_level_head_and_standing_reach_the_card(self):
        [item] = parse_dated_list(json.dumps([{
            "title": "Society of Saint Pius X", "date": "1970", "parent": "Pope",
            "standing": "schismatic", "level": "priestly society", "head": "Superior General",
            "authority": "its priories and seminaries"}]))
        assert item.parent_body == "Pope"
        assert item.standing == "breakaway"
        details = dict(item.details)
        assert details["Level"] == "priestly society"
        assert details["Led by"] == "Superior General"
        assert details["Authority"] == "its priories and seminaries"
        assert details["Standing"] == "Schismatic"


class TestGenealogy:
    def test_the_kind_of_lineage_decides_what_is_searched(self):
        spec = lambda focus: ResearchSpec(kind="genealogy", focus=focus)
        assert "people and families" in rm.genealogy_prompt("X", None, spec("families"))
        assert "lines of succession" not in rm.genealogy_prompt("X", None, spec("families"))
        assert "who succeeded whom" in rm.genealogy_prompt("X", None, spec("dynasties"))
        assert "divine, heroic and legendary parentage" in rm.genealogy_prompt("X", None, spec("deities"))
        assert "teachers and students" in rm.genealogy_prompt("X", None, spec("schools"))
        assert "proto-languages" in rm.genealogy_prompt("X", None, spec("languages"))

    def test_each_direction_is_its_own_instruction(self):
        spec = lambda d: ResearchSpec(kind="genealogy", focus="families", direction=d)
        assert "follow the line BACK" in rm.genealogy_prompt("X", None, spec("ancestors"))
        assert "FORWARD" not in rm.genealogy_prompt("X", None, spec("ancestors"))
        assert "follow the line FORWARD" in rm.genealogy_prompt("X", None, spec("descendants"))
        both = rm.genealogy_prompt("X", None, spec("both"))
        assert "BACK" in both and "FORWARD" in both
        # With no direction sent, a genealogy goes back, as the dashboard defaults it.
        assert "follow the line BACK" in rm.genealogy_prompt("X", None, ResearchSpec(kind="genealogy"))

    def test_the_parent_is_named_for_the_kind_of_lineage(self):
        prompt = rm.genealogy_prompt("Latin", None, ResearchSpec(kind="genealogy", focus="languages"))
        assert "every parent language the research names" in prompt
        assert "-1 for a parent language" in prompt

    def test_only_the_chosen_perspectives_are_asked_for(self):
        prompt = rm.genealogy_prompt("Zeus", None, ResearchSpec(
            kind="genealogy", focus="deities", perspectives=["traditional"]))
        assert "Traditional & Mythic:" in prompt
        assert "Historical:" not in prompt

    def test_traditional_descent_is_reported_not_asserted(self):
        prompt = rm.genealogy_prompt("X", None, ResearchSpec(kind="genealogy"))
        assert "do not present it as historical fact" in prompt
        assert "Never invent a link" in prompt

    def test_the_line_is_kept_in_generation_order(self, run):
        items = [{"title": "Pepin the Short", "date": "714-768 CE", "generation": -1, "relationship": "father"},
                 {"title": "Charlemagne", "date": "742-814 CE", "generation": 0},
                 {"title": "Charles Martel", "date": "c. 688-741 CE", "generation": -2}]
        trace, _ = run(items, title="Charlemagne",
                       research={"kind": "genealogy", "focus": "dynasties", "direction": "ancestors"})
        assert trace.origin.source_title == "Pepin the Short"
        assert [e.source_title for e in trace.timeline] == ["Charlemagne", "Charles Martel"]

    def test_generation_and_relationship_reach_the_card(self):
        [item] = parse_dated_list(json.dumps([{
            "title": "Pepin the Short", "date": "714-768 CE", "generation": -1,
            "relationship": "father", "line": "Frankish annals"}]))
        details = dict(item.details)
        assert details["Generation"] == "-1"
        assert details["Relationship"] == "father"
        assert details["Line"] == "Frankish annals"


# ----------------------------------------------------------------------------- the rest
class TestTheRequestAndTheCache:
    def test_an_unknown_choice_is_refused(self):
        with pytest.raises(ValidationError):
            ResearchSpec(kind="horoscope")
        with pytest.raises(ValidationError):
            ResearchSpec(kind="chronology", perspectives=["vibes"])
        with pytest.raises(ValidationError):
            ResearchSpec(kind="genealogy", direction="sideways")

    def test_the_research_choices_are_echoed_back(self, run):
        trace, _ = run(TWO, research={"kind": "establishment", "focus": "societies"})
        assert trace.research.kind == "establishment"
        assert trace.research.focus == "societies"

    def test_a_plain_trace_keeps_the_cache_key_it_always_had(self):
        assert request_key("The Flood", language="en") == request_key("The Flood", language="en", research=None)
        assert orch._research_key(TraceRequest(title="The Flood")) is None
        # Origins asks its own question now, so it is never the plain trace's entry.
        assert orch._research_key(TraceRequest(title="The Flood", research=ResearchSpec())) is not None

    def test_a_request_from_before_genealogy_keeps_its_cache_key(self):
        # `direction` came later; a request that does not set it is keyed as it always was.
        key = orch._research_key(TraceRequest(title="Troy", research=ResearchSpec(kind="parallels")))
        assert "direction" not in json.loads(key)

    def test_different_choices_are_different_cache_entries(self):
        a = orch._research_key(TraceRequest(title="Troy", research=ResearchSpec(kind="establishment")))
        up = orch._research_key(TraceRequest(title="Troy", research=ResearchSpec(kind="genealogy", direction="ancestors")))
        down = orch._research_key(TraceRequest(title="Troy", research=ResearchSpec(kind="genealogy", direction="descendants")))
        assert up != down
        b = orch._research_key(TraceRequest(title="Troy", research=ResearchSpec(kind="parallels")))
        assert a and b and a != b


class TestTheIdsMatchTheDashboard:
    """Pinned on both sides: phansora's tests/chrono-research.test.js pins the same lists.

    An id the dashboard sends that is not here is refused; an id here the dashboard never
    sends is a prompt nobody can reach.
    """

    def test_the_kinds_and_choices(self):
        from typing import get_args

        from phansora.products.chrono_origin import models as m

        assert get_args(m.ResearchKind) == (
            "chronology", "genealogy", "origins", "interpretations", "parallels", "translations",
            "establishment")
        assert get_args(m.ResearchDirection) == ("ancestors", "descendants", "both")
        assert get_args(m.ResearchSequence) == ("narrative", "historical")
        assert list(rm.CHRONOLOGY_PERSPECTIVES) == ["narrative", "historical"]
        assert list(rm.INTERPRETATION_READINGS) == [
            "coded_political", "numerical", "ritual", "mystical", "solar_astronomical", "initiatory"]
        assert list(rm.GENEALOGY_FOCUS) == ["families", "dynasties", "deities", "schools", "languages"]
        assert list(rm.GENEALOGY_PERSPECTIVES) == ["historical", "traditional", "textual"]
        assert list(rm._DIRECTIONS) == ["ancestors", "descendants", "both"]
        assert list(rm.ORIGINS_FOCUS) == ["inventions", "customs", "words", "ideas", "texts"]
        assert list(rm.ORIGINS_PERSPECTIVES) == ["historical", "textual", "archaeological", "linguistic"]
        assert list(rm.PARALLEL_FOCUS) == ["stories", "people", "symbols", "practices"]
        assert list(rm.PARALLEL_PERSPECTIVES) == ["historical", "literary", "symbolic", "cultural", "natural"]
        assert list(rm.ESTABLISHMENT_FOCUS) == ["religions", "societies", "governments", "companies", "movements"]
        assert list(rm.TRANSLATION_FOCUS) == ["wording", "choices", "manuscripts"]
