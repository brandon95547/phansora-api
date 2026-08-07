"""Research Atlas's one hard promise: it reports what sources say, it does not judge.

The examples in the first class are taken VERBATIM from the product specification --
the BAD/GOOD pairs it was written around. They are the acceptance criteria for the
whole product, so they are asserted directly rather than paraphrased.

Worth knowing if these ever fail: an earlier version of the checker passed both GOOD
cases and failed both BAD ones, because it accepted a bare speech verb as attribution.
"This conspiracy theory ALLEGES that ..." and "This CLAIM has been debunked" each
contain a speech word and name nobody. Attribution requires a source, not a verb.
"""
from __future__ import annotations

import pytest

from phansora.products.research_atlas import extraction as E
from phansora.products.research_atlas import neutrality as N
from phansora.products.research_atlas import report as R


class TestSpecExamples:
    """The BAD/GOOD pairs from the product brief, asserted literally."""

    @pytest.mark.parametrize("text", [
        "This conspiracy theory alleges that the group met in secret.",
        "This claim has been debunked.",
        "This proves that Person A was involved.",
    ])
    def test_bad_examples_are_flagged(self, text):
        assert N.scan(text), f"must flag the report judging in its own voice: {text!r}"

    @pytest.mark.parametrize("text", [
        'Source 4 describes the account as a "conspiracy theory".',
        "Source 7 disputes the account presented in Source 3.",
        "Sources 2 and 6 both connect Person A to Organization B.",
    ])
    def test_good_examples_pass(self, text):
        assert not N.scan(text), f"attributed reporting must be allowed: {text!r}"


class TestOwnVoiceJudgments:
    @pytest.mark.parametrize("text", [
        "The material is largely propaganda.",
        "These are fringe, extremist positions.",
        "The publication is unreliable and not credible.",
        "The account is false.",
        "The report is pseudoscience.",
        "The claim was disproven years ago.",
        # Anonymous voice is the report's own voice with the subject removed, which is
        # the loophole a verb-based check leaves open.
        "It is alleged that the material is disinformation.",
        "It has been widely debunked.",
    ])
    def test_flagged(self, text):
        assert N.scan(text), f"unattributed characterization must be flagged: {text!r}"


class TestOrdinaryEnglishIsNotFlagged:
    """true/false are ordinary words. Flagging them everywhere trains people to ignore
    the checker, so they are only caught in evaluative constructions."""

    @pytest.mark.parametrize("text", [
        "The notarized document is a true copy of the original.",
        "The scan produced a false positive on page 12.",
        "The timeline lists events in chronological order.",
        "Person A held a legitimate business license per Source 3.",
        "Source 5 states the account is false.",
        "According to Source 2, the report is misinformation.",
        "Source 9 calls the group extremist.",
    ])
    def test_not_flagged(self, text):
        assert not N.scan(text), f"must not flag ordinary or attributed usage: {text!r}"


class TestStructureWalk:
    def test_booleans_and_keys_are_never_scanned(self):
        """`"resolved": false` is a data value. Scanning raw JSON would flag it forever."""
        obj = {"conflicts": [{"topic": "date", "resolved": False}], "count": 3}
        assert N.scan_fields(obj) == []

    def test_finds_violation_and_reports_its_path(self):
        obj = {"synthesis": "The account has been debunked."}
        found = N.scan_fields(obj)
        assert len(found) == 1
        assert found[0].path == "synthesis"

    def test_redact_drops_only_the_judging_sentence(self):
        text = ("Source 3 lists the date. This claim has been debunked. "
                "Source 4 lists a different date.")
        out = N.redact(text)
        assert "debunked" not in out
        assert "Source 3 lists the date." in out
        assert "Source 4 lists a different date." in out


# ---------------------------------------------------------------------------
# The deterministic half of the pipeline: merge and render, no model involved.
# ---------------------------------------------------------------------------

@pytest.fixture
def per_chunk():
    """Three sources that deliberately disagree about a date and form a link chain."""
    return [
        ("brief.pdf", {
            "people": [{"name": "Alice Reyes", "aliases": ["A. Reyes"],
                        "described_as": "listed as operations lead"}],
            "organizations": [{"name": "Meridian Group", "aliases": ["Meridian"],
                               "described_as": "a logistics firm"}],
            "places": [{"name": "Harbor Point", "described_as": "site of the depot"}],
            "timeline": [{"date": "March 12, 1998", "time": "",
                          "event": "The depot inspection was carried out"}],
            "events": [], "claims": [], "documents": [],
            "relationships": [
                {"from": "Alice Reyes", "relation": "worked for", "to": "Meridian Group",
                 "basis": "listed as operations lead"},
                {"from": "Meridian Group", "relation": "operated from", "to": "Harbor Point",
                 "basis": "depot address given"}],
            "gaps": [],
        }),
        ("interview.mp3", {
            "people": [{"name": "Alice Reyes", "aliases": [], "described_as": "present that day"},
                       {"name": "Daniel Okoro", "aliases": [], "described_as": "the depot supervisor"}],
            "organizations": [], "places": [],
            # Same occurrence, different date. Must survive as a conflict.
            "timeline": [{"date": "March 14, 1998", "time": "09:00",
                          "event": "The depot inspection was carried out"}],
            "events": [], "claims": [], "documents": [],
            "relationships": [{"from": "Harbor Point", "relation": "was supervised by",
                               "to": "Daniel Okoro", "basis": "stated in interview"}],
            "gaps": [],
        }),
    ]


class TestMerge:
    def test_entities_join_across_sources_and_keep_every_description(self, per_chunk):
        rec = E.merge_extractions(per_chunk)
        alice = next(p for p in rec["people"] if p["name"] == "Alice Reyes")
        assert len(alice["sources"]) == 2
        assert "A. Reyes" in alice["aliases"]
        # Both descriptions are kept, each tagged with its source. Flattening them into
        # one line would quietly choose whose description wins.
        assert {d["source"] for d in alice["descriptions"]} == {"brief.pdf", "interview.mp3"}

    def test_date_disagreement_is_detected_not_resolved(self, per_chunk):
        rec = E.merge_extractions(per_chunk)
        assert len(rec["date_conflicts"]) == 1
        dates = {v["date"] for v in rec["date_conflicts"][0]["versions"]}
        assert dates == {"March 12, 1998", "March 14, 1998"}
        # Both entries survive in the timeline; neither was dropped in favor of the other.
        assert len(rec["timeline"]) == 2

    def test_undated_entries_sort_last_rather_than_being_guessed(self):
        entries = [{"date": "", "event": "undated thing", "source": "a"},
                   {"date": "1998-03-12", "event": "dated thing", "source": "a"}]
        assert [e["event"] for e in E.sort_timeline(entries)] == ["dated thing", "undated thing"]


class TestReport:
    def test_all_sixteen_sections_always_render(self, per_chunk):
        rec = E.merge_extractions(per_chunk)
        md = R.render_report(rec, {}, ["brief.pdf", "interview.mp3"])
        for i, name in enumerate(R.SECTIONS, start=1):
            assert f"## {i}. {name}" in md

    def test_empty_sections_say_so_instead_of_vanishing(self):
        """Absence is a finding. A missing heading looks like nobody looked."""
        empty = E.merge_extractions([])
        md = R.render_report(empty, {}, ["only.pdf"])
        assert "## 10. Documents and Supporting Material" in md
        assert "No entries of this kind were recorded" in md

    def test_chains_span_sources_but_only_use_attributed_links(self, per_chunk):
        rec = E.merge_extractions(per_chunk)
        md = R.render_report(rec, {}, ["brief.pdf", "interview.mp3"])
        assert "Chains formed by the links above" in md
        assert "Alice Reyes -> worked for -> Meridian Group -> operated from -> Harbor Point" in md

    def test_report_is_ascii_only(self, per_chunk):
        """The Node PDF renderer downstream strips non-ASCII glyphs."""
        rec = E.merge_extractions(per_chunk)
        md = R.render_report(rec, {"synthesis": "An em-dash — and an arrow →."},
                             ["brief.pdf", "interview.mp3"])
        assert all(ord(c) < 128 for c in md)

    def test_rendered_report_passes_its_own_neutrality_audit(self, per_chunk):
        rec = E.merge_extractions(per_chunk)
        md = R.render_report(rec, {}, ["brief.pdf", "interview.mp3"])
        assert R.audit_report(md, ["brief.pdf", "interview.mp3"]) == []

    def test_corroboration_is_a_count_never_a_rating(self, per_chunk):
        """The banned vocabulary includes 'credible'. Counts replace confidence."""
        rec = E.merge_extractions(per_chunk)
        md = R.render_report(rec, {}, ["brief.pdf", "interview.mp3"])
        assert "carried by 2 sources" in md
        for word in ("confidence", "credible", "reliable", "Very High"):
            assert word.lower() not in md.lower()


class TestSourceMap:
    def test_citations_read_as_sentences(self):
        sm = R.SourceMap(["a.pdf", "b.mp3", "c.txt"])
        assert sm.num("b.mp3") == "Source 2"
        assert sm.cite(["a.pdf"]) == "Source 1"
        assert sm.cite(["a.pdf", "c.txt"]) == "Sources 1 and 3"

    def test_unknown_label_still_attributes(self):
        """Falling back to the raw label keeps the sentence attributed. Emitting an
        unattributed sentence is the one outcome the product cannot have."""
        sm = R.SourceMap(["a.pdf"])
        assert sm.num("mystery.txt") == "mystery.txt"
