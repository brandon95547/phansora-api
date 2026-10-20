"""Research Atlas's scanned PDFs (/ocr), read by the shared OCR (phansora.shared.ocr).

The Node app used to read scans itself, with its own Tesseract calls, the first 60 pages
only and no cleanup. These pin what replaced it: every page goes to the shared converter
with Research Atlas's model, the reading is a job a caller can watch page by page, a
reading cut short resumes from its pages, and a failure says what to do without handing
the caller our internals.
"""
from __future__ import annotations

import time

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed on this host")

from fastapi.testclient import TestClient  # noqa: E402

from phansora.products.research_atlas import api  # noqa: E402
from phansora.shared.ocr.pdf import PdfConverter  # noqa: E402

PDF = b"%PDF-1.4\n" + b"scan " * 64


@pytest.fixture(autouse=True)
def _fresh_jobs():
    api._JOBS.clear()
    yield
    api._JOBS.clear()


def _wait_for(client, job_id, statuses, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in statuses:
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached {statuses}")


def _converter(monkeypatch, *, text="", pages=3, error=None):
    """A stand-in PdfConverter: writes `text` (or raises `error`) and reports `pages` pages."""
    seen = {}

    class Fake(PdfConverter):
        def __init__(self, cfg=None):
            seen["cfg"] = cfg

        async def convert_pdf_to_txt_async(self, pdf_path, out_txt_path, *, cache_dir=None, on_progress=None):
            seen["pdf"] = pdf_path.read_bytes()
            seen["cache_dir"] = cache_dir
            cache_dir.mkdir(parents=True, exist_ok=True)
            for done in range(1, pages + 1):
                await on_progress("ocr", done, pages)
            if error is not None:
                raise error
            out_txt_path.write_text(text, encoding="utf-8")
            return out_txt_path

    monkeypatch.setattr("phansora.shared.ocr.pdf.PdfConverter", Fake)
    return seen


def _upload(client, data=PDF):
    return client.post("/ocr", files={"file": ("scan.pdf", data, "application/pdf")})


def test_a_scan_is_read_whole_with_research_atlas_model(monkeypatch):
    seen = _converter(monkeypatch, pages=240, text="PAGE 1\nThe first page.\n\n\n\nPAGE 2\nThe second page.")
    with TestClient(api.app) as client:
        res = _upload(client)
        assert res.status_code == 200 and res.json()["status"] == "queued"
        job_id = res.json()["job_id"]
        body = _wait_for(client, job_id, {"completed", "failed"})
        assert body["status"] == "completed", body
        assert body["progress"] == {"step": "ocr", "done": 240, "total": 240}
        text = client.get(f"/jobs/{job_id}/result").json()["result"]["text"]

    assert text == "The first page.\n\nThe second page."
    cfg = seen["cfg"]
    assert cfg.model_var == "RESEARCH_ATLAS_MODEL"
    assert cfg.keep_page_breaks is False and cfg.to_chapters is False
    assert seen["pdf"] == PDF
    # The pages kept for a resume go once the text is out, and so does the upload.
    assert not seen["cache_dir"].exists()
    assert not list(api.OCR_DIR.glob("*.pdf"))


def test_a_reading_cut_short_keeps_its_pages_for_the_next_try(monkeypatch):
    seen = _converter(monkeypatch, error=RuntimeError("worker restarted"))
    with TestClient(api.app) as client:
        _wait_for(client, _upload(client).json()["job_id"], {"failed"})
        first = seen["cache_dir"]
        assert first.exists()
        # The same file asked again reads into the same folder, so its pages are reused.
        _wait_for(client, _upload(client).json()["job_id"], {"failed"})
        assert seen["cache_dir"] == first
        # A different file does not.
        _wait_for(client, _upload(client, PDF + b"more").json()["job_id"], {"failed"})
        assert seen["cache_dir"] != first


def test_a_failure_does_not_return_the_exception_text(monkeypatch):
    _converter(monkeypatch, error=RuntimeError("DeepSeek HTTP 401: {'error': 'sk-live-secret'} at /var/www/phansora-api"))
    with TestClient(api.app) as client:
        job_id = _upload(client).json()["job_id"]
        body = _wait_for(client, job_id, {"failed"})
        assert "secret" not in body["error"] and "/var/www" not in body["error"]
        assert body["error"].startswith("This scanned PDF could not be read.") and "reference" in body["error"]
        result = client.get(f"/jobs/{job_id}/result")
        assert result.status_code == 409 and "secret" not in result.text


def test_a_missing_tesseract_is_said_in_words(monkeypatch):
    _converter(monkeypatch, error=RuntimeError("tesseract is not installed or it's not in your PATH. See README file."))
    with TestClient(api.app) as client:
        body = _wait_for(client, _upload(client).json()["job_id"], {"failed"})
    assert "Tesseract OCR engine needs to be installed" in body["error"]


def test_a_scan_with_no_text_fails_and_says_so(monkeypatch):
    _converter(monkeypatch, text="PAGE 1\n\nPAGE 2\n")
    with TestClient(api.app) as client:
        body = _wait_for(client, _upload(client).json()["job_id"], {"completed", "failed"})
    assert body["status"] == "failed"
    assert body["error"] == "No text could be read from this scanned PDF's pages."


def test_a_file_that_is_not_a_pdf_is_refused(monkeypatch):
    seen = _converter(monkeypatch, text="never")
    with TestClient(api.app) as client:
        res = _upload(client, b"PK\x03\x04 a zip, not a PDF")
    assert res.status_code == 400
    assert "cfg" not in seen and not api._JOBS
    assert not list(api.OCR_DIR.glob("*.pdf"))


def test_too_many_waiting_jobs_is_a_busy_answer(monkeypatch):
    _converter(monkeypatch, text="never")
    monkeypatch.setattr(api, "_MAX_OUTSTANDING", 0)
    with TestClient(api.app) as client:
        res = _upload(client)
    assert res.status_code == 503 and res.headers.get("retry-after") == "60"
    assert not list(api.OCR_DIR.glob("*.pdf"))


def test_a_pipeline_job_reports_no_progress(monkeypatch):
    monkeypatch.setattr(api, "run_pipeline", lambda **kw: {"ok": True})
    with TestClient(api.app) as client:
        body = _wait_for(client, client.post("/run", json={"text": "x"}).json()["job_id"], {"completed"})
    assert body["progress"] is None
