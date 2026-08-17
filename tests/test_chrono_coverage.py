"""Whether a trace researches a whole subject, and whether the research survives.

A trace for "Jesus Christ" came back with twelve entries and passed every
mechanical check the rubric had. It also had no text in it — no gospel, no
letter, nothing composed by anyone — no calendar, and a name that crossed three
languages in a single sentence. The rounds had researched all three. Nothing was
wrong with the research; the report was assembled by a stage that had never been
told any of it existed.

Two failures, and both are about a signal being dropped rather than a judgement
being wrong:

  The extract stage types every mention it finds, and the loop measures its own
  coverage against those types. The block that carried mentions into synthesis
  sent the date, the title and the tier — and not the type. So the loop could
  record "text_composition: covered" while the stage that writes the report had
  no idea a text had ever been found.

  Tiers were assertions by a model, and nothing stopped one promoting a lead. A
  Wikipedia file-description page and an AI-generated wiki were both labelled
  primary and both duly READ, spending the page budget on exactly the two
  sources the policy exists to skip.

These tests pin the signals, not the wording: what synthesis is told, what the
loop searches for when a strand is still open, and which tier survives contact
with a model that would rather it were higher.
"""
from __future__ import annotations

from phansora.products.chrono_origin.pipeline import evidence as ev
from phansora.products.chrono_origin.pipeline import orchestrator as orch
from phansora.products.chrono_origin.pipeline import source_policy as sp


class TestResearchReachesTheReport:
    """The mentions block is the only channel between the rounds and the report."""

    def test_carries_the_node_type(self):
        block = orch._format_mentions_block([
            {"year": 55, "node_type": "text_composition", "source_title": "Pauline letters",
             "claim": "Composed in the 50s CE.", "citations": ["https://x"], "precision": "decade"},
        ])
        assert "type=text_composition" in block

    def test_carries_a_span_rather_than_pinning_a_process_to_one_year(self):
        block = orch._format_mentions_block([
            {"year": -200, "year_end": 70, "node_type": "context",
             "source_title": "Second Temple Judaism", "claim": "…"},
        ])
        assert "when=-200..70" in block

    def test_carries_the_signals_that_say_a_claim_is_still_on_a_lead(self):
        block = orch._format_mentions_block([
            {"year": 30, "source_title": "X", "claim": "…",
             "discovery_only": True, "cites": "Ehrman 2012", "published": "2019"},
        ])
        assert "LEAD_ONLY" in block
        assert "that_page_cites=Ehrman 2012" in block
        assert "source_published=2019" in block

    def test_an_untyped_mention_still_reads_as_an_event(self):
        block = orch._format_mentions_block([{"year": 1, "source_title": "X", "claim": "…"}])
        assert "type=event" in block


class TestStrandsSurviveADedupe:
    """Rounds overlap; the merge must keep the better reading, not the first one."""

    def test_a_later_round_supplies_the_type_the_first_one_missed(self):
        merged = ev.dedupe_mentions([
            {"source_title": "Gospel of Mark", "year": 70, "citations": ["https://a"]},
            {"source_title": "Gospel of Mark", "year": 70, "citations": ["https://b"],
             "node_type": "text_composition", "year_end": 80},
        ])
        assert len(merged) == 1
        assert merged[0]["node_type"] == "text_composition"
        assert merged[0]["year_end"] == 80
        assert merged[0]["citations"] == ["https://a", "https://b"]

    def test_a_type_already_established_is_not_overwritten(self):
        merged = ev.dedupe_mentions([
            {"source_title": "P52", "year": 125, "node_type": "manuscript_witness"},
            {"source_title": "P52", "year": 125, "node_type": "event"},
        ])
        assert merged[0]["node_type"] == "manuscript_witness"


class TestOpenStrandsBuyTheirOwnSearches:
    """A strand nobody searched for is a strand that stays open forever."""

    def test_each_open_strand_gets_a_query(self):
        qs = orch._strand_queries("Jesus Christ", ["dating_framework", "text_composition"], [], limit=5)
        assert len(qs) == 2
        assert any("calendar" in q for q in qs)
        assert any("composition dates" in q for q in qs)
        assert all("Jesus Christ" in q for q in qs)

    def test_respects_the_reserved_slot_count(self):
        qs = orch._strand_queries(
            "X", ["dating_framework", "text_composition", "manuscript_witness"], [], limit=2
        )
        assert len(qs) == 2

    def test_never_reruns_a_search_already_paid_for(self):
        first = orch._strand_queries("X", ["dating_framework"], [], limit=1)
        again = orch._strand_queries("X", ["dating_framework"], first, limit=1)
        assert first and again == []

    def test_a_covered_strand_asks_nothing(self):
        assert orch._strand_queries("X", [], [], limit=5) == []


class TestSynthesisIsToldWhatWasResearched:
    def test_separates_a_researched_strand_from_an_uncovered_one(self):
        block = orch._format_strands_block(
            ["text_composition", "dating_framework"], {"text_composition"}
        )
        assert "text_composition: RESEARCHED" in block
        assert "dating_framework: NOT COVERED" in block

    def test_a_strand_found_without_being_planned_still_has_to_appear(self):
        block = orch._format_strands_block(["text_composition"], {"text_composition", "term_history"})
        assert "term_history: RESEARCHED (unplanned)" in block

    def test_says_so_plainly_when_there_was_no_plan(self):
        assert "no strand plan" in orch._format_strands_block([], set())


class TestATierIsNotWhateverTheModelSaysItIs:
    """The rule that makes the policy binding rather than decorative."""

    def test_a_wiki_claimed_as_primary_is_still_a_wiki(self):
        assert sp.resolve_tier("primary", "https://en.wikipedia.org/wiki/File:Codex.jpg") == "low_authority"

    def test_a_generated_encyclopaedia_is_a_lead(self):
        assert sp.resolve_tier("repository", "https://grokipedia.com/page/Annals") == "low_authority"

    def test_an_author_upload_platform_establishes_that_a_paper_exists_not_that_it_is_right(self):
        # Academia.edu is a container. The paper it carries may be excellent; the
        # URL is evidence of neither review nor publication, which is tier 3.
        assert sp.default_tier("https://www.academia.edu/37829437/Messianic_Expectations") == "reference_index"
        assert sp.resolve_tier("academic", "https://www.academia.edu/37829437/X") == "reference_index"

    def test_a_promotion_the_host_cannot_make_is_still_allowed(self):
        # An unrecognised domain that really is a national archive: the model
        # knows something the host list does not, and demoting it would make
        # traces worse while looking stricter.
        assert sp.resolve_tier("repository", "https://digi.example-archiv.de/ms/1") == "repository"

    def test_an_unlabelled_source_falls_back_to_its_host(self):
        assert sp.resolve_tier(None, "https://www.jstor.org/stable/1") == "academic"
        assert sp.resolve_tier("nonsense", "https://reddit.com/r/x") == "low_authority"


class TestTheReadBudgetIsSpentOnEvidence:
    def test_a_wiki_a_model_called_primary_is_not_opened(self):
        citations = [
            {"url": "https://en.wikipedia.org/wiki/File:Codex.jpg"},
            {"url": "https://grokipedia.com/page/Annals"},
            {"url": "https://www.bl.uk/collection-items/codex-sinaiticus"},
        ]
        mentions = [{
            "citations": [c["url"] for c in citations],
            "source_tier": "primary",
            "confidence": 0.9,
        }]

        def tier_for(url):
            return sp.resolve_tier("primary", url)

        chosen = ev.select_for_reading(citations, mentions, tier_for, limit=5)
        assert chosen == ["https://www.bl.uk/collection-items/codex-sinaiticus"]
