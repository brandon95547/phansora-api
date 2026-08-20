"""Following a lead's references backward, and the prompt discipline around it.

"Follow their references backward to stronger material and cite that instead"
was in the doctrine from the beginning and had no implementation anywhere in the
repo — an instruction naming a research action the pipeline could not perform.
The page whose footnotes were most worth having was also the one page the reader
was forbidden to open, so a wiki was simultaneously the thing a trace rested on
and the thing it could never learn anything from.

The mechanism is deliberately cheap: open the lead, take the reference block,
never take a word of the prose. That last property is what keeps this from being
a loophole in the policy rather than an expression of it, so it is asserted here
rather than left to the reviewer's memory.
"""
from __future__ import annotations

from phansora.products.chrono_origin.pipeline import prompts as P
from phansora.products.chrono_origin.pipeline.reader import parse_references

# A wiki-shaped page: prose, then a reference list pointing at real scholarship,
# mixed with the wiki's own internal furniture.
WIKI_PAGE = """
<html><body>
  <h1>Gospel of Mark</h1>
  <p>Most scholars date the composition of Mark to around 70 CE, shortly after
     the destruction of the Second Temple.</p>
  <p>See also <a href="https://en.wikipedia.org/wiki/Q_source">Q source</a>.</p>
  <h2><span class="mw-headline" id="References">References</span></h2>
  <ol class="references">
    <li>Ehrman, Bart D. <a href="https://www.jstor.org/stable/3268034">Textual criticism</a>.</li>
    <li><a href="https://doi.org/10.1017/S0028688500001234">Dating the Gospel of Mark</a></li>
    <li><a href="https://www.bl.uk/collection-items/codex-sinaiticus">Codex Sinaiticus</a>,
        British Library, Add MS 43725.</li>
    <li><a href="https://en.wikipedia.org/wiki/Markan_priority">Markan priority</a></li>
    <li><a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA</a></li>
  </ol>
</body></html>
"""

PROSE = "Most scholars date the composition of Mark to around 70 CE"


class TestReferenceHarvesting:
    def test_returns_the_scholarship_a_lead_page_cites(self):
        urls = [r["url"] for r in parse_references(WIKI_PAGE)]
        assert "https://www.jstor.org/stable/3268034" in urls
        assert "https://doi.org/10.1017/S0028688500001234" in urls
        assert "https://www.bl.uk/collection-items/codex-sinaiticus" in urls

    def test_never_returns_page_prose(self):
        # THE policy-compliance property. Mining a lead page must not become a
        # way of reading one: the text of a wiki may not reach the model by any
        # route, including this one.
        refs = parse_references(WIKI_PAGE)
        blob = " ".join(r.get("text", "") for r in refs)
        assert PROSE not in blob
        assert "destruction of the Second Temple" not in blob

    def test_ignores_the_wikis_own_furniture(self):
        urls = [r["url"] for r in parse_references(WIKI_PAGE)]
        # Harvesting internal links and licence boilerplate would just refill the
        # candidate pool with more of the tier we are trying to climb out of.
        assert not any("wikipedia.org" in u for u in urls)
        assert not any("creativecommons.org" in u for u in urls)

    def test_ignores_links_above_the_reference_block(self):
        # The "See also" link sits in the body. Only what the page cites counts.
        urls = [r["url"] for r in parse_references(WIKI_PAGE)]
        assert not any("Q_source" in u for u in urls)

    def test_a_page_with_no_references_yields_nothing(self):
        assert parse_references("<html><body><p>Nothing here.</p></body></html>") == []
        assert parse_references("") == []

    def test_finds_an_unheaded_reference_list_by_its_class(self):
        html = '<div class="reflist"><a href="https://www.jstor.org/stable/1">A</a></div>'
        assert [r["url"] for r in parse_references(html)] == ["https://www.jstor.org/stable/1"]


class TestPromptDiscipline:
    """Where the doctrine reaches, and what it costs to put it there."""

    def test_the_full_hierarchy_reaches_the_stages_that_can_act_on_it(self):
        decompose = P.DECOMPOSE_PROMPT.format(
            title="t", context="", max_queries=5, source_hierarchy=P.SOURCE_HIERARCHY
        )
        synthesize = P.SYNTHESIZE_PROMPT.format(
            title="t", mentions_block="", citations_block="", pages_block="",
            strands_block="", source_hierarchy=P.SOURCE_HIERARCHY, max_connections=24,
        )
        for prompt in (decompose, synthesize):
            assert "TIER 1 — PRIMARY EVIDENCE" in prompt
            assert "LEAD GENERATORS, NOT EVIDENCE" in prompt

    def test_the_short_form_reaches_every_searching_stage(self):
        search = P.SEARCH_PROMPT.format(
            title="t", context_clause="", query="q", search_doctrine=P.SEARCH_DOCTRINE
        )
        chase = P.CHASE_SEARCH_PROMPT.format(
            title="t", claim="c", weak_source="w", cites="", references="",
            search_doctrine=P.SEARCH_DOCTRINE,
        )
        expand = P.EXPAND_SEARCH_PROMPT.format(
            story_title="t", context_clause="", when="", parent_source_title="",
            parent_claim="", search_doctrine=P.SEARCH_DOCTRINE,
            # An expansion is aimed at one axis and told what the board already shows.
            mode_search=P.expand_mode("related")["search"],
            existing_block=P.format_existing_block(["Already shown"]),
        )
        for prompt in (search, chase, expand):
            assert "LEADS" in prompt

    def test_the_stage_that_assigns_tiers_is_told_what_they_are(self):
        # EXTRACT assigns every source its tier and used to receive none of the
        # doctrine at all — the one stage doing the classifying was the one stage
        # never given the vocabulary.
        extract = P.EXTRACT_PROMPT.format(
            title="t", notes="", pages_block="", citations_block="", earliest_known="",
            max_queries=5, extract_doctrine=P.EXTRACT_DOCTRINE, open_strands="", prior_queries="",
        )
        assert "Tiers 4-5 are leads" in extract

    def test_search_doctrine_stays_small_enough_to_send_twenty_times(self):
        # This string is multiplied by every search in a trace, which is why the
        # doctrine was split in the first place. A ceiling here is cheaper than
        # rediscovering the cost in a bill.
        assert len(P.SEARCH_DOCTRINE) < 1200

    def test_every_relation_and_node_type_appears_in_the_prompt(self):
        # The classic drift: a member added to the enum, and the model never told
        # it exists. Both directions of the vocabulary must agree.
        from phansora.products.chrono_origin.models import EvidenceKind, RelationType
        import typing

        synthesize = P.SYNTHESIZE_PROMPT
        for relation in typing.get_args(RelationType):
            assert relation in synthesize, relation
        for kind in typing.get_args(EvidenceKind):
            assert kind in synthesize, kind

    def test_the_structural_rules_survive_editing(self):
        # Each of these is a rule the code cannot enforce on its own, and the
        # reason the trace read like a summary instead of a piece of research.
        s = P.SYNTHESIZE_PROMPT
        # The chain rule itself, stated as the question every step answers.
        assert "WHAT IS THE NEXT SURVIVING PIECE OF EVIDENCE?" in s
        assert "EVIDENCE ONLY, NO EXCEPTIONS" in s
        assert "A STEP MUST BE LOCATABLE" in s
        assert "COMPOSITION AND SURVIVING COPY ARE NOT THE SAME FACT" in s
        assert "A TEXT IS NOT EVIDENCE FOR WHAT IT NARRATES" in s
        assert "SAY WHAT IS MISSING" in s
        # The failure mode the whole structure exists to prevent.
        assert "DO NOT LAUNDER A CONCLUSION INTO A STEP" in s

    def test_the_prompt_shows_the_worked_chain_and_what_it_excludes(self):
        # The brief's own example. Without it the model reliably reaches for the
        # connective tissue between documents, which is the one thing banned.
        s = P.SYNTHESIZE_PROMPT
        assert "Dead Sea Scrolls" in s and "Septuagint" in s and "Tacitus" in s
        assert "messianic expectation" in s.lower()

    def test_the_planner_offers_the_strands_the_loop_measures(self):
        # The planner names strands and the loop counts them covered; if the two
        # lists drift the loop silently never finishes covering anything.
        from phansora.products.chrono_origin.pipeline.orchestrator import _STRANDS

        decompose = P.DECOMPOSE_PROMPT.format(
            title="t", context="", max_queries=5, source_hierarchy=P.SOURCE_HIERARCHY
        )
        for strand in _STRANDS:
            assert strand in decompose, strand
