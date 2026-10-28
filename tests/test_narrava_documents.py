"""Importing a document for the narration form's Import tab (narrava_studio.services.documents,
POST /ebook/analyze, GET /ebook/jobs/{id}, POST /script/from-chapters).

The parser and DeepSeek are stubbed. What is checked is ours: how blocks become chapters,
that the reading runs as a job to its result and cleans up after itself, and what a
narration written from the chosen topics is asked to say.
"""
from __future__ import annotations

import glob
import os
import tempfile
import time
from types import SimpleNamespace

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed on this host")

from fastapi.testclient import TestClient  # noqa: E402

from phansora.products.book_alchemy import parsers  # noqa: E402
from phansora.products.narrava_studio import server  # noqa: E402
from phansora.products.narrava_studio.services import documents, llm, script  # noqa: E402


def block(text, chapter=None):
    return SimpleNamespace(text=text, chapter=chapter)


def words(n, word="word"):
    return " ".join([word] * n)


def test_blocks_under_one_heading_are_one_chapter_in_reading_order():
    doc = SimpleNamespace(blocks=[
        block("Opening.", None),
        block("The storm came.", "The storm"),
        block("It lasted all night.", "The storm"),
        block("   ", "The storm"),
        block("Morning.", "The rescue"),
    ])
    out = documents.chapters_from_doc(doc)
    assert [c["title"] for c in out] == ["Section 1", "The storm", "The rescue"]
    assert out[1]["text"] == "The storm came.\n\nIt lasted all night."
    assert [c["index"] for c in out] == [0, 1, 2]
    assert out[1]["word_count"] == 7


def _leftovers():
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "narrava_document_*")))


@pytest.fixture()
def reading(monkeypatch):
    """A document with a short front matter and two real chapters; DeepSeek stubbed."""
    doc = SimpleNamespace(blocks=[
        block("Copyright page.", "Front matter"),
        block(words(200, "storm"), "The storm"),
        block(words(200, "rescue"), "The rescue"),
    ])
    seen = {"parsed": [], "condensed": []}

    def parse(*, source_format, path, title_hint, **_):
        seen["parsed"].append((source_format, title_hint, os.path.exists(path)))
        return doc

    def condense(system, user, *, max_output_tokens=2000):
        seen["condensed"].append(user.split("\n", 1)[0])
        return f"About {user.split(': ', 1)[1].splitlines()[0]}."

    monkeypatch.setattr(parsers, "parse_source", parse)
    monkeypatch.setattr(llm, "deepseek_configured", lambda: True)
    monkeypatch.setattr(llm, "deepseek_text", condense)
    monkeypatch.setattr(llm, "deepseek_json", lambda *a, **k: {
        "suggested_title": "Keepers of the Light",
        "chapters": [{"index": 2, "recommended": True, "why": "The rescue is the turn."}],
    })
    return seen


def test_a_document_is_read_as_a_job_to_its_topics(reading):
    before = _leftovers()
    with TestClient(server.app) as client:
        res = client.post("/ebook/analyze", files={"file": ("keepers.txt", b"text", "text/plain")}, data={"title": "Keepers"})
        assert res.status_code == 200, res.text
        job_id = res.json()["job_id"]
        for _ in range(100):
            status = client.get(f"/ebook/jobs/{job_id}").json()
            if status["status"] in ("done", "failed"):
                break
            time.sleep(0.05)
    assert status["status"] == "done", status
    result = status["result"]
    assert result["book_title"] == "Keepers of the Light"
    # The front matter is too short to be a topic; the two chapters were condensed and ranked.
    assert [c["title"] for c in result["chapters"]] == ["The storm", "The rescue"]
    assert [c["summary"] for c in result["chapters"]] == ["About The storm.", "About The rescue."]
    assert [c["recommended"] for c in result["chapters"]] == [False, True]
    assert result["chapters"][1]["why"] == "The rescue is the turn."
    assert reading["parsed"] == [("txt", "Keepers", True)]
    # The upload is the job's to remove, and it did.
    assert _leftovers() == before


def test_a_document_with_nothing_readable_fails_with_a_reason(reading, monkeypatch):
    monkeypatch.setattr(parsers, "parse_source", lambda **_: SimpleNamespace(blocks=[block("Too short.", "One")]))
    with TestClient(server.app) as client:
        job_id = client.post("/ebook/analyze", files={"file": ("scan.pdf", b"%PDF", "application/pdf")}).json()["job_id"]
        for _ in range(100):
            status = client.get(f"/ebook/jobs/{job_id}").json()
            if status["status"] in ("done", "failed"):
                break
            time.sleep(0.05)
    assert status["status"] == "failed"
    assert "No readable text was found" in status["error"]


def test_an_unsupported_format_is_refused_before_anything_is_kept(reading):
    before = _leftovers()
    client = TestClient(server.app, raise_server_exceptions=False)
    res = client.post("/ebook/analyze", files={"file": ("notes.xyz", b"x", "application/octet-stream")})
    assert res.status_code == 400
    assert "Unsupported document format" in res.json()["detail"]
    assert _leftovers() == before


def test_an_unknown_job_is_a_404(reading):
    client = TestClient(server.app, raise_server_exceptions=False)
    assert client.get("/ebook/jobs/nope").status_code == 404


def test_a_narration_from_chosen_topics_leads_with_the_conversation(monkeypatch):
    seen = {}

    def fake(prompt, **kw):
        seen["prompt"], seen["kw"] = prompt, kw
        return "script"

    monkeypatch.setattr(script, "generate_script", fake)
    out = script.from_chapters(
        "Keepers", [{"title": "The rescue", "summary": "The keeper rows out."}],
        direction="For children, warm.", style="story", target_duration_sec=60,
    )
    assert out == "script"
    lines = seen["prompt"].splitlines()
    assert lines[0] == "For children, warm."
    assert 'Adapt these parts of "Keepers" into one narration.' in seen["prompt"]
    assert "- The rescue: The keeper rows out." in lines
    assert seen["kw"]["style"] == "story" and seen["kw"]["target_duration_sec"] == 60
