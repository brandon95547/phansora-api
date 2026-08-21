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


