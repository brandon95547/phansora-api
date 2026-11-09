"""The five kinds of research on the start screen, each asking its own question.

The failure these pin is the one the screen was most likely to ship with: five cards
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

    def test_origins_with_nothing_chosen_is_the_plain_trace(self, run):
        _, plain = run(TWO)
        _, origins = run(TWO, research={"kind": "origins"})
        assert plain.searches == origins.searches

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
        ("parallels", "find the parallels to"),
        ("translations", "is worded across languages and versions"),
        ("evidence", "find the evidence bearing on"),
    ])
    def test_every_other_kind_sends_a_prompt_of_its_own(self, run, kind, marker):
        _, client = run(TWO, research={"kind": kind})
        prompt = client.searches[0]
        assert marker in prompt
        assert "trace the complete historical lineage" not in prompt
        assert '"The Flood"' in prompt

    def test_the_context_reaches_every_kind(self, run):
        for kind in ("chronology", "genealogy", "parallels", "translations", "evidence"):
            _, client = run(TWO, research={"kind": kind}, context="Mesopotamian myth")
            assert "Mesopotamian myth" in client.searches[0], kind


class TestChronology:
    def test_each_sequence_is_its_own_instruction(self):
        spec = lambda seq: ResearchSpec(kind="chronology", perspectives=["historical"], sequence=seq)
        assert "NARRATIVE ORDER" in rm.chronology_prompt("X", None, spec("narrative"))
        assert "bring the related passages" in rm.chronology_prompt("X", None, spec("narrative"))
        assert "mark its uncertainty" in rm.chronology_prompt("X", None, spec("historical"))
        assert "SYMBOLIC ORDER" in rm.chronology_prompt("X", None, spec("symbolic"))

    def test_only_the_chosen_perspectives_are_asked_for(self):
        prompt = rm.chronology_prompt("X", None, ResearchSpec(
            kind="chronology", perspectives=["solar_astronomical", "historical"]))
        assert "Solar & Astronomical:" in prompt and "Historical:" in prompt
        assert "Bodily & Spiritual:" not in prompt

    def test_separate_timelines_tag_each_entry_with_its_perspective(self):
        prompt = rm.chronology_prompt("X", None, ResearchSpec(
            kind="chronology", perspectives=["narrative", "historical"], layout="separate"))
        assert "ONE ENTRY PER EVENT PER PERSPECTIVE" in prompt
        assert '"perspective"' in prompt and '"readings"' not in prompt

    def test_grouped_stages_carry_every_perspectives_reading(self):
        prompt = rm.chronology_prompt("X", None, ResearchSpec(
            kind="chronology", perspectives=["narrative", "historical"], layout="grouped"))
        assert "ONE ENTRY PER STAGE" in prompt
        assert '"readings"' in prompt

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

    def test_each_entry_keeps_its_perspective(self, run):
        items = [{"title": "A", "date": "1 CE", "perspective": "solar & astronomical"},
                 {"title": "B", "date": "2 CE", "perspective": "historical"}]
        trace, _ = run(items, research={"kind": "chronology",
                                        "perspectives": ["solar_astronomical", "historical"]})
        assert trace.origin.perspective == "Solar & Astronomical"
        assert trace.timeline[0].perspective == "Historical"

    def test_grouped_readings_become_rows_on_the_card(self):
        text = json.dumps([{"title": "Stage", "date": "", "readings": {
            "Narrative": "The waters rise.", "Historical": "A regional flood c. 2900 BCE."}}])
        item = parse_dated_list(text)[0]
        assert ("Narrative", "The waters rise.") in item.details
        assert ("Historical", "A regional flood c. 2900 BCE.") in item.details


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


class TestEvidence:
    def test_it_asks_for_challenges_and_limitations_as_hard_as_support(self):
        prompt = rm.evidence_prompt("The Trojan War happened", None, ResearchSpec(kind="evidence"))
        assert "CHALLENGE" in prompt and "LIMITATIONS" in prompt
        assert "Do not decide the question" in prompt

    def test_only_the_chosen_source_types_are_named(self):
        prompt = rm.evidence_prompt("X", None, ResearchSpec(kind="evidence", source_types=["inscriptions"]))
        assert "inscriptions and artifacts" in prompt
        assert "historical records" not in prompt

    def test_each_source_carries_its_stance(self, run):
        items = [{"title": "A", "date": "700 BCE", "stance": "supports"},
                 {"title": "B", "date": "600 BCE", "stance": "Against"},
                 {"title": "C", "date": "1900 CE", "stance": "limitation"}]
        trace, _ = run(items, research={"kind": "evidence"})
        assert trace.origin.stance == "supports"
        assert [e.stance for e in trace.timeline] == ["challenges", "limitation"]


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
        trace, _ = run(TWO, research={"kind": "evidence", "source_types": ["texts"]})
        assert trace.research.kind == "evidence"
        assert trace.research.source_types == ["texts"]

    def test_a_plain_trace_keeps_the_cache_key_it_always_had(self):
        assert request_key("The Flood", language="en") == request_key("The Flood", language="en", research=None)
        assert orch._research_key(TraceRequest(title="The Flood")) is None
        assert orch._research_key(TraceRequest(title="The Flood", research=ResearchSpec())) is None

    def test_a_request_from_before_genealogy_keeps_its_cache_key(self):
        # `direction` came later; a request that does not set it is keyed as it always was.
        key = orch._research_key(TraceRequest(title="Troy", research=ResearchSpec(kind="evidence")))
        assert "direction" not in json.loads(key)

    def test_different_choices_are_different_cache_entries(self):
        a = orch._research_key(TraceRequest(title="Troy", research=ResearchSpec(kind="evidence")))
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
            "chronology", "genealogy", "origins", "parallels", "translations", "evidence")
        assert get_args(m.ResearchDirection) == ("ancestors", "descendants", "both")
        assert get_args(m.ResearchSequence) == ("narrative", "historical", "symbolic")
        assert get_args(m.ResearchLayout) == ("separate", "grouped")
        assert get_args(m.ResearchSourceType) == ("all", "texts", "records", "inscriptions")
        assert list(rm.CHRONOLOGY_PERSPECTIVES) == [
            "narrative", "bodily_spiritual", "solar_astronomical", "historical", "comparative"]
        assert list(rm.GENEALOGY_FOCUS) == ["families", "dynasties", "deities", "schools", "languages"]
        assert list(rm.GENEALOGY_PERSPECTIVES) == ["historical", "traditional", "textual"]
        assert list(rm._DIRECTIONS) == ["ancestors", "descendants", "both"]
        assert list(rm.ORIGINS_FOCUS) == ["stories", "beliefs", "words", "practices"]
        assert list(rm.ORIGINS_PERSPECTIVES) == ["historical", "textual", "archaeological", "linguistic", "comparative"]
        assert list(rm.PARALLEL_FOCUS) == ["stories", "people", "symbols", "practices"]
        assert list(rm.PARALLEL_PERSPECTIVES) == ["historical", "literary", "symbolic", "cultural", "natural"]
        assert list(rm.TRANSLATION_FOCUS) == ["wording", "choices", "manuscripts"]
        assert list(rm.EVIDENCE_SOURCE_TYPES) == ["texts", "records", "inscriptions"]
