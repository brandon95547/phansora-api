"""When the dossier's LLM cleanup pass is worth paying for.

Cleanup exists to repair PDF/OCR extraction damage, and it is the most expensive stage in
the pipeline that can be entirely unnecessary — article text arrives via Readability with
none of that damage. Skipping it there is a large speedup, but the failure modes point in
opposite directions and only one of them is visible:

  false positive  -> a clean source is cleaned anyway. Slow; harmless.
  false negative  -> a damaged source is NOT cleaned. Fast; silently worse dossier.

So the tests below care most about the damaged cases still being caught.
"""
import pytest

from phansora.products.research_atlas.text_cleaner import needs_cleanup

# Deliberately seeded with what naive artifact detection trips over: a bare domain, an
# initialism, "e.g.", a Ph.D, and camelCase product names.
CLEAN_ARTICLE = """# Reactor Report
Source: https://example.com/a

The committee met on 4 July 2024. Investigators from the JavaScript Foundation and the
iPhone division said the cause was not yet established. The budget rose by 12% in the
same period, though the report notes this figure excludes contractor costs. See
example.com/report and the U.S. filing (e.g. Ph.D theses) for more detail on macOS.
""" * 6

# Missing spaces, jammed punctuation, OCR letter errors, a hyphenated line break.
DAMAGED_PDF = """TheChurch hasalways maintained thatthe index ofprohibited books,Index
libr-
orum prohibitorum,wasnot merely aList.TheCatho Hc position onthis wasclear.
Manyauthors weresilenced,andtheir worksdestroyed.Thisshort summary cannotdevote
""" * 6


def test_damaged_pdf_is_still_cleaned():
    damaged, density = needs_cleanup(DAMAGED_PDF, "book.pdf")
    assert damaged
    assert density > 5, f"expected dense damage, measured {density:.2f}/1k"


def test_clean_article_skips_the_llm_pass():
    damaged, density = needs_cleanup(CLEAN_ARTICLE, "https://example.com/a")
    assert not damaged
    assert density == 0.0


def test_clean_prose_skips_even_without_a_url_label():
    # Provenance is the first signal, but it must not be the only one — a .txt or .docx
    # that happens to be clean should not pay for a repair pass either.
    damaged, density = needs_cleanup(CLEAN_ARTICLE, "notes.txt")
    assert not damaged, f"clean prose measured {density:.2f}/1k"


@pytest.mark.parametrize("noise", [
    "Read more at example.com/story and nytimes.com/section today.",
    "The U.S. and U.K. agreed. See e.g. the Ph.D thesis, i.e. chapter four.",
    "Built with JavaScript, TypeScript, macOS, iPhone and eBay integrations.",
])
def test_ordinary_prose_is_not_mistaken_for_damage(noise):
    # Each of these breaks a naive detector: bare domains look like jammed punctuation,
    # initialisms look like missing spaces, camelCase looks like merged words.
    # Joined with a space — concatenating them bare would glue "today.Read", which is a
    # real missing-space artifact and would be measuring the test, not the detector.
    damaged, density = needs_cleanup(" ".join([noise] * 20), "notes.txt")
    assert not damaged, f"measured {density:.2f}/1k"


def test_url_provenance_beats_the_heuristic():
    # Readability output is clean by construction, so the label settles it without
    # measuring — and this is the path every article-sourced dossier takes.
    damaged, density = needs_cleanup(DAMAGED_PDF, "https://example.com/a")
    assert not damaged and density == 0.0


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_empty_text_is_never_sent_for_cleaning(text):
    assert needs_cleanup(text, "x.pdf") == (False, 0.0)
