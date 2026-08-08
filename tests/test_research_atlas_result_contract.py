"""The keys run_pipeline owes its callers.

This is a boundary test, not a behavior test. The pipeline's return value is a contract
with two consumers that live outside this repo's import graph, so nothing here fails
when a key disappears -- the run succeeds, bills the model, and is then thrown away:

  ok             the Node worker (phansora/src/research-atlas/index.js) treats a result
                 without a truthy `ok` as "pipeline returned an invalid response" and
                 fails the job AFTER the full report has been generated and paid for.
  toc_full_path  api.py resolves the finished Markdown through it to read the report
                 back off disk.

Both were dropped when Dossier Nova's engine was replaced, which is exactly the failure
this file exists to catch on the next rewrite.
"""
import pytest

from phansora.products.research_atlas import extraction, pipeline

# The keys the Node worker reads off the pipeline's own result. `toc_markdown` is not
# here because api.py adds it, and `research` is not here because the worker guards it
# (a job without a research dataset is still a job) -- that one is a feature the neutral
# engine has not reimplemented rather than a broken contract.
NODE_WORKER_KEYS = ("ok", "chunk_count")

CHUNK_RECORD = {
    "people": [{"name": "Ada Lovelace", "described_as": "mathematician"}],
    "organizations": [{"name": "Analytical Engine Committee"}],
    "places": [{"name": "London"}],
    "timeline": [{"date": "1843", "event": "Notes on the Analytical Engine published"}],
    "events": [{"title": "Publication", "date": "1843", "summary": "The notes appear."}],
    "claims": [{"statement": "The engine could weave algebraic patterns.",
                "attributed_to": "Lovelace"}],
    "documents": [{"title": "Notes", "described_as": "translator's notes"}],
    "relationships": [{"from": "Ada Lovelace", "relation": "wrote", "to": "Notes"}],
    "gaps": [{"gap": "The engine was never built."}],
}

SOURCES = [
    {"label": "memoir.txt", "text": "The committee met in London.\n\nThe notes were published in 1843."},
    {"label": "article.txt", "text": "A second account of the same meeting.\n\nIt agrees on the date."},
]


@pytest.fixture
def stubbed_pipeline(monkeypatch):
    """Run the real pipeline with every network call replaced."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-not-used")
    # The stages that would otherwise reach DeepSeek on clean, undamaged test prose.
    monkeypatch.setenv("CLEAN_EXTRACTED_TEXT", "false")
    monkeypatch.setenv("ENABLE_SOURCE_PROFILING", "false")
    monkeypatch.setattr(
        extraction, "extract_from_chunk",
        lambda client, chunk_text, source_label: dict(CHUNK_RECORD),
    )
    monkeypatch.setattr(
        extraction, "build_connective_sections",
        lambda client, record, source_labels: ({"overview": "Two sources describe one meeting."}, []),
    )


def test_result_carries_the_keys_the_node_worker_requires(stubbed_pipeline, tmp_path):
    out = tmp_path / "report.md"
    result = pipeline.run_pipeline(sources=SOURCES, toc_full_path=str(out))

    assert result["ok"] is True
    for key in NODE_WORKER_KEYS:
        assert key in result, f"the Node worker reads {key} off the result"
    assert result["chunk_count"] >= len(SOURCES)


def test_toc_full_path_points_at_the_written_report(stubbed_pipeline, tmp_path):
    # api.py does `result.get("toc_full_path") or run_toc_path`, so a missing key falls
    # back and hides itself. Assert the key is present AND correct.
    out = tmp_path / "nested" / "report.md"
    result = pipeline.run_pipeline(sources=SOURCES, toc_full_path=str(out))

    assert result["toc_full_path"] == str(out)
    assert out.exists() and out.read_text(encoding="utf-8").strip()


def test_no_usable_sources_raises_rather_than_returning_not_ok(stubbed_pipeline, tmp_path):
    # The API layer turns exceptions into status="failed" with a message; a falsy `ok`
    # would instead surface as the generic "invalid response". Failures belong in the
    # first channel.
    with pytest.raises(ValueError):
        pipeline.run_pipeline(
            sources=[{"label": "empty.txt", "text": "   "}],
            toc_full_path=str(tmp_path / "report.md"),
        )
