"""Whether a trace researches a whole subject, and whether the research survives.

A trace for "Jesus Christ" came back with twelve entries and passed every
mechanical check the rubric had. It also had no text in it — no gospel, no
letter, nothing composed by anyone — and a name that crossed three languages in a
single sentence. The rounds had researched all of it. Nothing was wrong with the
research; the report was assembled by a stage that had never been told any of it
existed.

Two failures, and both are about a signal being dropped rather than a judgement
being wrong:

  The extract stage types every mention it finds, and the loop measures its own
  coverage against those types. The block that carried mentions into synthesis
  sent the date, the title and the tier — and not the type. So the loop could
  record "earliest_texts: covered" while the stage that writes the report had no
  idea a text had ever been found.

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



class TestStrandsSurviveADedupe:
    """Rounds overlap; the merge must keep the better reading, not the first one."""

    def test_a_later_round_supplies_the_type_the_first_one_missed(self):
        merged = ev.dedupe_mentions([
            {"source_title": "Gospel of Mark", "year": 70, "citations": ["https://a"]},
            {"source_title": "Gospel of Mark", "year": 70, "citations": ["https://b"],
             "node_type": "text", "year_end": 80},
        ])
        assert len(merged) == 1
        assert merged[0]["node_type"] == "text"
        assert merged[0]["year_end"] == 80
        assert merged[0]["citations"] == ["https://a", "https://b"]

    def test_a_type_already_established_is_not_overwritten(self):
        merged = ev.dedupe_mentions([
            {"source_title": "P52", "year": 125, "node_type": "manuscript"},
            {"source_title": "P52", "year": 125, "node_type": "event"},
        ])
        assert merged[0]["node_type"] == "manuscript"





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
