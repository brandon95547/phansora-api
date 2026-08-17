"""The source policy, pinned.

These tests exist because of one trace. Asked for the origin of Jesus Christ,
Chrono Origin came back citing Wikipedia as the evidentiary basis of its claims
and treating the Gospel of Mark as evidence for the events it narrates — while
the prompt sitting in front of the model said, in as many words, that wikis are
leads rather than foundations and that a text's composition date is not the date
of what it describes. The instruction was right and unenforced.

So the rules here are the ones that no longer depend on the model agreeing:
what tier a source is under a given claim, whether it may be the basis of that
claim at all, what an evidence type is allowed to be given its citations, and
what happens when nothing arguable can be found. Each one is a rule the pipeline
applies rather than requests.
"""
from __future__ import annotations

import pytest

from phansora.products.chrono_origin.pipeline import evidence as ev
from phansora.products.chrono_origin.pipeline import source_policy as sp
from phansora.products.chrono_origin.pipeline.orchestrator import (
    _coerce_dossier,
    _as_strands,
    _open_strands,
    _strands_covered,
)

WIKI = "https://en.wikipedia.org/wiki/Gospel_of_Mark"
JSTOR = "https://www.jstor.org/stable/3268034"
DOI = "https://doi.org/10.1017/S0028688500001234"
BL = "https://www.bl.uk/collection-items/codex-sinaiticus"
NYT_1963 = "https://www.nytimes.com/1963/11/23/archives/kennedy-shot.html"
NASA = "https://www.nasa.gov/history/apollo-11"


class TestHostClassification:
    def test_places_each_host_family(self):
        assert sp.default_tier(WIKI) == "low_authority"
        assert sp.default_tier(JSTOR) == "academic"
        assert sp.default_tier(BL) == "repository"
        assert sp.default_tier(NYT_1963) == "press"
        assert sp.default_tier(NASA) == "institutional"

    def test_a_doi_is_its_own_tier(self):
        # A DOI resolver establishes that a work exists and who wrote it. It is
        # not an argument that the work is right, which is why the mandate gives
        # it a tier below scholarship rather than folding it in.
        assert sp.default_tier(DOI) == "reference_index"
        assert sp.EVIDENCE_RANK["reference_index"] == 3

    def test_an_archive_on_a_gov_domain_is_still_an_archive(self):
        # loc.gov and archives.gov both contain ".gov"; ordering decides whether
        # the Library of Congress is read as a holding institution or a website.
        assert sp.default_tier("https://www.loc.gov/item/2021667925/") == "repository"
        assert sp.default_tier("https://www.archives.gov/founding-docs") == "repository"


class TestTwoOrderings:
    """Reading order and evidentiary weight are different questions."""

    def test_an_unclassified_source_cannot_be_evidence(self):
        assert sp.EVIDENCE_RANK["unknown"] == 5

    def test_but_is_still_worth_opening_before_a_wiki(self):
        # Opening an unrecognised page is how the pipeline discovers it is a
        # national archive on an odd domain. Ranking `unknown` as tier 5 for
        # reading too would quietly stop it reading good sources — a change that
        # would make traces worse while looking stricter.
        assert sp.READ_RANK["unknown"] < sp.READ_RANK["low_authority"]


class TestTierIsRelational:
    """A tier is a relation between a source and a claim, not a property of a domain."""

    def test_a_newspaper_is_primary_evidence_for_its_own_week(self):
        assert sp.rank_for("press", published_year=1963, claim_year=1963, claim_precision="year") == 1

    def test_and_commentary_on_everything_else(self):
        assert sp.rank_for("press", published_year=2024, claim_year=1963, claim_precision="year") == 5

    def test_never_promotes_on_an_unknown_date(self):
        # A wrong promotion hands tier 1 to exactly the class of source the
        # policy exists to demote, so the guard is the absence of a date, not a
        # guess at one.
        assert sp.rank_for("press", published_year=None, claim_year=1963, claim_precision="year") == 5

    def test_never_promotes_against_a_vaguely_dated_claim(self):
        assert sp.rank_for(
            "press", published_year=1963, claim_year=1963, claim_precision="century"
        ) == 5

    def test_scholarship_is_never_aged_out(self):
        # Analysis does not decay. Demoting a 2024 paper for being recent would
        # inverse the hierarchy and reward whoever wrote first.
        assert sp.rank_for("academic", published_year=2024, claim_year=1963, claim_precision="year") == 2

    def test_a_self_declared_primary_source_on_a_wiki_is_still_a_wiki(self):
        assert sp.rank_for("primary", url=WIKI) == 5

    def test_reads_a_year_out_of_a_url_for_free(self):
        assert sp.url_year(NYT_1963) == 1963
        assert sp.url_year("https://example.org/about") is None


class TestRoles:
    def test_tiers_one_to_three_may_be_evidence(self):
        assert [sp.role_for(r) for r in (1, 2, 3)] == ["evidence"] * 3

    def test_tiers_four_and_five_are_leads(self):
        # Tier 4 is a lead on purpose: an institution explaining something it did
        # not itself record is the case the mandate says to follow backward.
        assert sp.role_for(4) == "discovery"
        assert sp.role_for(5) == "discovery"


class TestEvidenceCap:
    """No page that says a thing is thereby evidence for it."""

    def test_a_wiki_cannot_buy_a_primary_document(self):
        assert sp.cap_evidence_type("primary_document", 5) == "later_historical_account"

    def test_metadata_cannot_buy_one_either(self):
        assert sp.cap_evidence_type("primary_document", 3) == "scholarly_inference"

    def test_a_real_primary_source_keeps_its_type(self):
        assert sp.cap_evidence_type("primary_document", 1) == "primary_document"

    def test_the_cap_never_promotes(self):
        # It is a ceiling, not a grade. A tradition backed by an archive is
        # still a tradition.
        assert sp.cap_evidence_type("tradition", 1) == "tradition"


class TestTheJesusCase:
    """The exact shape that produced the trace this work exists to fix."""

    def test_a_wikipedia_only_claim_cannot_render_as_direct_evidence(self):
        d = _coerce_dossier(
            {
                "claim": "Jesus was crucified c. 30 CE",
                "evidence_type": "primary_document",
                "claim_class": "direct_evidence",
                "earliest_supporting_source": "Gospel of Mark",
            },
            fallback_claim="",
            confidence=0.8,
            read_urls=set(),
            citations=[WIKI],
            citation_ranks={WIKI: 5},
        )
        assert d.evidence_type == "later_historical_account"
        assert d.claim_class == "historical_inference"  # the board's green dot is now unreachable
        assert d.verification == "unverified"

    def test_an_archive_page_that_was_actually_read_survives_intact(self):
        d = _coerce_dossier(
            {"claim": "Codex Sinaiticus is a 4th-century manuscript",
             "evidence_type": "primary_document",
             "earliest_supporting_source": "Codex Sinaiticus, British Library"},
            fallback_claim="",
            confidence=0.9,
            read_urls={BL},
            citations=[BL],
            citation_ranks={BL: 1},
        )
        assert d.evidence_type == "primary_document"
        assert d.claim_class == "direct_evidence"
        assert d.verification == "verified"

    def test_a_claim_with_nothing_behind_it_says_so(self):
        d = _coerce_dossier(
            {"claim": "x", "evidence_type": "primary_document"},
            fallback_claim="",
            confidence=0.8,
            read_urls=set(),
            citations=[],
            citation_ranks={},
        )
        assert d.verification == "unknown"
        assert d.evidence_type == "absent"
        assert d.claim_class == "unknown"

    def test_a_doi_alone_is_unverified_however_confident(self):
        d = _coerce_dossier(
            {"claim": "x", "evidence_type": "primary_document",
             "earliest_supporting_source": "A journal article"},
            fallback_claim="",
            confidence=0.95,
            read_urls=set(),
            citations=[DOI],
            citation_ranks={DOI: 3},
        )
        assert d.verification == "unverified"
        assert d.evidence_type == "scholarly_inference"

    def test_a_scholarly_dispute_marks_the_claim_disputed(self):
        # Scholars disagreeing is a different fact from evidence against, and a
        # dossier that records only the second under-reports the first.
        d = _coerce_dossier(
            {"claim": "x", "scholarly_dispute": "Markan priority is contested by the Griesbach school."},
            fallback_claim="",
            confidence=0.6,
            read_urls={JSTOR},
            citations=[JSTOR],
            citation_ranks={JSTOR: 2},
        )
        assert d.disputed is True
        assert d.scholarly_dispute.startswith("Markan priority")


class TestCapAndRelabelNeverDrop:
    """The invariant that keeps enforcement from making traces worse."""

    @pytest.mark.parametrize("rank", [1, 2, 3, 4, 5])
    def test_every_claim_survives_at_every_rank(self, rank):
        # Nothing in this policy removes a node or a citation. A trace whose
        # every claim ends up UNVERIFIED is still a trace; an over-filtered one
        # trips the empty-timeline guard and costs the user a refund and an error
        # where an honest weak answer would have done.
        d = _coerce_dossier(
            {"claim": "a claim", "evidence_type": "primary_document",
             "earliest_supporting_source": "something"},
            fallback_claim="",
            confidence=0.7,
            read_urls=set(),
            citations=["https://example.org/a"],
            citation_ranks={"https://example.org/a": rank},
        )
        assert d is not None
        assert d.claim == "a claim"
        assert d.verification in {"verified", "unverified", "unknown"}


class TestChasingLeads:
    @staticmethod
    def _tier(url: str) -> str:
        return sp.default_tier(url)

    def test_only_claims_resting_on_leads_are_chased(self):
        weak = {"claim": "weak", "citations": [WIKI], "confidence": 0.6}
        strong = {"claim": "strong", "citations": [JSTOR], "confidence": 0.9}
        targets = ev.chase_targets([weak, strong], self._tier, limit=5)
        assert [t["claim"] for t in targets] == ["weak"]

    def test_a_well_sourced_trace_pays_nothing(self):
        strong = {"claim": "strong", "citations": [BL, JSTOR], "confidence": 0.9}
        assert ev.chase_targets([strong], self._tier, limit=5) == []

    def test_respects_its_budget(self):
        weak = [{"claim": f"w{i}", "citations": [WIKI], "confidence": 0.5} for i in range(9)]
        assert len(ev.chase_targets(weak, self._tier, limit=3)) == 3

    def test_mining_selects_only_lead_pages(self):
        cites = [{"url": WIKI}, {"url": JSTOR}, {"url": BL}, {"url": "https://www.reddit.com/r/x"}]
        picked = ev.select_for_reference_mining(cites, self._tier, limit=5)
        assert picked == [WIKI, "https://www.reddit.com/r/x"]

    def test_reading_still_refuses_to_read_leads(self):
        # Mining is a separate budget for a separate purpose, not a widening of
        # the reading set: a wiki's prose must still never reach the model.
        cites = [{"url": WIKI}, {"url": BL}]
        assert ev.select_for_reading(cites, [], self._tier, limit=5) == [BL]


class TestStrandCoverage:
    def test_reads_both_planner_shapes(self):
        assert _as_strands(
            [{"strand": "manuscript_witness", "why": "x"}, "term_history", "not_a_strand"]
        ) == ["manuscript_witness", "term_history"]

    def test_node_types_report_their_strand_covered(self):
        covered = _strands_covered(
            [{"node_type": "manuscript_witness"}, {"node_type": "context"}, {"node_type": "event"}]
        )
        # "context" satisfies the precursor_context strand under a different name;
        # a plain event satisfies no strand at all.
        assert covered == {"manuscript_witness", "precursor_context"}

    def test_open_strands_drive_the_next_round(self):
        planned = ["text_composition", "manuscript_witness", "external_attestation"]
        assert _open_strands(planned, {"text_composition"}) == [
            "manuscript_witness",
            "external_attestation",
        ]


class TestContextIsNotInfluence:
    def test_a_causal_edge_out_of_a_context_node_is_downgraded(self):
        # "Older religions existed, therefore they shaped this one" is the oldest
        # error in comparative history, and the one the user's own outline opens
        # by warning against.
        edges = ev.validate_connections(
            [{"from_id": "c1", "to_id": "t1", "relation": "derives_from",
              "evidence": {"mechanism": "shared motifs"}}],
            {"c1", "t1"},
            max_connections=5,
            node_types={"c1": "context"},
        )
        assert edges[0]["relation"] == "provides_context"

    def test_a_real_descent_claim_is_left_alone(self):
        edges = ev.validate_connections(
            [{"from_id": "t1", "to_id": "t2", "relation": "derives_from",
              "evidence": {"mechanism": "direct textual dependency, shared verbatim passages"}}],
            {"t1", "t2"},
            max_connections=5,
            node_types={"t1": "text_composition"},
        )
        assert edges[0]["relation"] == "derives_from"
