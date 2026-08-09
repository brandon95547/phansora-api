"""Tests for the scanned-PDF path: does it survive a restart, and does it say so.

Two failures motivated these. A thousand-page scan reported a single flat 6% for
the two-plus hours it spent recognizing and cleaning, which is indistinguishable
from a hung job. And restarting the worker mid-read killed Tesseract with it,
which surfaced as "OCR failed" and permanently failed a book with nothing wrong
with it — throwing away every page already recognized.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from phansora.products.book_alchemy import pipeline
from phansora.products.spokenverse.txt_to_voice import pdf_pipeline
from phansora.products.spokenverse.txt_to_voice.pdf_pipeline import PdfConverter, PdfToTxtConfig


def make_converter(**overrides) -> PdfConverter:
    """A converter without __init__: building one reads .env and the DeepSeek key,
    and none of the code under test talks to either."""
    conv = object.__new__(PdfConverter)
    conv.cfg = PdfToTxtConfig(**overrides)
    conv.ocr_cfg = None
    conv.chat_cfg = None
    conv.render_cfg = None
    return conv


def pages(n: int):
    return [(i, f"image-{i}".encode()) for i in range(1, n + 1)]


class Ticks:
    """Collects the (step, done, total) the pipeline reports."""

    def __init__(self):
        self.seen = []

    async def __call__(self, step, done, total):
        self.seen.append((step, done, total))

    def counts(self, step):
        return [done for s, done, _ in self.seen if s == step]


# --------------------------------------------------------------------- OCR
def test_ocr_reports_every_page_and_caches_it(tmp_path, monkeypatch):
    conv = make_converter(ocr_concurrency=4)
    calls = []

    def fake_ocr(image_bytes, *, cfg=None):
        calls.append(image_bytes)
        return f"text of {image_bytes.decode()}"

    monkeypatch.setattr(pdf_pipeline, "ocr_image_bytes", fake_ocr)
    ticks = Ticks()

    out = asyncio.run(
        conv._ocr_pages(pages(10), "book.pdf", cache_dir=tmp_path, on_progress=ticks)
    )

    assert len(calls) == 10
    assert out[0] == (1, "text of image-1")
    assert [p for p, _ in out] == list(range(1, 11))
    # One tick per page, ending at the total — that is what moves the bar.
    assert sorted(ticks.counts("ocr")) == list(range(1, 11))
    assert ticks.seen[-1] == ("ocr", 10, 10)
    assert (tmp_path / "page_00007.txt").read_text() == "text of image-7"


def test_ocr_resumes_from_cache_without_re_recognizing(tmp_path, monkeypatch):
    conv = make_converter()
    monkeypatch.setattr(
        pdf_pipeline, "ocr_image_bytes", lambda b, *, cfg=None: f"text of {b.decode()}"
    )
    asyncio.run(conv._ocr_pages(pages(6), "book.pdf", cache_dir=tmp_path))

    # Second run: recognizing anything again is the bug this guards.
    def explode(image_bytes, *, cfg=None):
        raise AssertionError("re-recognized a page that was already cached")

    monkeypatch.setattr(pdf_pipeline, "ocr_image_bytes", explode)
    ticks = Ticks()
    out = asyncio.run(
        conv._ocr_pages(pages(6), "book.pdf", cache_dir=tmp_path, on_progress=ticks)
    )

    assert out == [(i, f"text of image-{i}") for i in range(1, 7)]
    assert ticks.seen[-1] == ("ocr", 6, 6)


def test_ocr_resumes_a_partial_run(tmp_path, monkeypatch):
    """The real shape of an interrupted book: some pages done, the rest not."""
    conv = make_converter(ocr_concurrency=1)
    done = []

    def fake_ocr(image_bytes, *, cfg=None):
        page = int(image_bytes.decode().split("-")[1])
        if page > 4:
            raise RuntimeError("killed mid-book")
        done.append(page)
        return f"text of {image_bytes.decode()}"

    monkeypatch.setattr(pdf_pipeline, "ocr_image_bytes", fake_ocr)
    with pytest.raises(RuntimeError):
        asyncio.run(conv._ocr_pages(pages(8), "book.pdf", cache_dir=tmp_path))
    assert done == [1, 2, 3, 4]

    monkeypatch.setattr(
        pdf_pipeline, "ocr_image_bytes", lambda b, *, cfg=None: f"text of {b.decode()}"
    )
    second = []

    def counting_ocr(image_bytes, *, cfg=None):
        second.append(int(image_bytes.decode().split("-")[1]))
        return f"text of {image_bytes.decode()}"

    monkeypatch.setattr(pdf_pipeline, "ocr_image_bytes", counting_ocr)
    out = asyncio.run(conv._ocr_pages(pages(8), "book.pdf", cache_dir=tmp_path))

    assert second == [5, 6, 7, 8]          # only the unfinished tail
    assert len(out) == 8


def test_ocr_without_cache_dir_behaves_as_before(tmp_path, monkeypatch):
    conv = make_converter()
    monkeypatch.setattr(
        pdf_pipeline, "ocr_image_bytes", lambda b, *, cfg=None: f"text of {b.decode()}"
    )
    out = asyncio.run(conv._ocr_pages(pages(3), "book.pdf"))
    assert out == [(i, f"text of image-{i}") for i in range(1, 4)]
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------------------------ cleaning
def test_clean_batches_cache_by_content(tmp_path, monkeypatch):
    conv = make_converter(clean_concurrency=2)
    seen = []

    async def fake_clean(batch, *, cfg=None, max_output_tokens=None):
        seen.append(batch)
        return f"cleaned:{batch}"

    monkeypatch.setattr(pdf_pipeline, "clean_ocr_text", fake_clean)
    ticks = Ticks()
    out = asyncio.run(
        conv._clean_batches(["a", "b", "c"], "book.pdf", cache_dir=tmp_path, on_progress=ticks)
    )
    assert out == ["cleaned:a", "cleaned:b", "cleaned:c"]
    assert ticks.seen[-1] == ("clean", 3, 3)

    # Same batches again: answered from disk, and still in order.
    async def explode(batch, *, cfg=None, max_output_tokens=None):
        raise AssertionError("re-cleaned a batch that was already cached")

    monkeypatch.setattr(pdf_pipeline, "clean_ocr_text", explode)
    again = asyncio.run(conv._clean_batches(["a", "b", "c"], "book.pdf", cache_dir=tmp_path))
    assert again == ["cleaned:a", "cleaned:b", "cleaned:c"]

    # A batch it has never seen is still asked for.
    monkeypatch.setattr(pdf_pipeline, "clean_ocr_text", fake_clean)
    seen.clear()
    asyncio.run(conv._clean_batches(["a", "d"], "book.pdf", cache_dir=tmp_path))
    assert seen == ["d"]


def test_empty_clean_result_is_not_cached(tmp_path, monkeypatch):
    """An empty answer is a failed call. Caching it would bake the hole in."""
    conv = make_converter()

    async def empty(batch, *, cfg=None, max_output_tokens=None):
        return "   "

    monkeypatch.setattr(pdf_pipeline, "clean_ocr_text", empty)
    asyncio.run(conv._clean_batches(["a"], "book.pdf", cache_dir=tmp_path))
    assert list(tmp_path.glob("clean_*.txt")) == []


# ------------------------------------------------- interrupted, not failed
@pytest.mark.parametrize(
    "message",
    [
        # The exact shape seen in production when a restart killed Tesseract.
        "(-15, 'Tesseract Open Source OCR Engine v4.1.1 with Leptonica')",
        "(-9, 'out of memory')",
    ],
)
def test_signal_kill_is_recognized(message):
    assert pipeline._killed_by_signal(RuntimeError(message))


def test_tesseract_status_attribute_is_recognized():
    exc = RuntimeError("child died")
    exc.status = -15
    assert pipeline._killed_by_signal(exc)


def test_ordinary_failures_are_not_mistaken_for_a_signal():
    assert not pipeline._killed_by_signal(RuntimeError("tesseract is not installed"))
    assert not pipeline._killed_by_signal(RuntimeError("(1, 'bad page')"))


def test_shutdown_turns_an_ocr_failure_into_a_resume(tmp_path, monkeypatch):
    """A deploy mid-book must not fail (and refund) the book."""
    writes = []

    async def fake_set_project(pid, **fields):
        writes.append(fields)

    monkeypatch.setattr(pipeline.db, "set_project", fake_set_project)
    monkeypatch.setattr(pipeline, "_shutting_down", True)

    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    class Boom(PdfConverter):
        def __init__(self, cfg=None):
            pass

        async def convert_pdf_to_txt_async(self, *a, **kw):
            raise RuntimeError("Tesseract died")

    monkeypatch.setattr(
        "phansora.products.spokenverse.txt_to_voice.pdf_pipeline.PdfConverter", Boom
    )

    with pytest.raises(pipeline.RetryableError):
        asyncio.run(pipeline._ocr_pdf_to_doc({"id": 1, "name": "b"}, str(pdf)))


def test_a_genuine_ocr_failure_still_fails_the_book(tmp_path, monkeypatch):
    async def fake_set_project(pid, **fields):
        return None

    monkeypatch.setattr(pipeline.db, "set_project", fake_set_project)
    monkeypatch.setattr(pipeline, "_shutting_down", False)

    pdf = tmp_path / "source.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    class Boom(PdfConverter):
        def __init__(self, cfg=None):
            pass

        async def convert_pdf_to_txt_async(self, *a, **kw):
            raise RuntimeError("tesseract is not installed or it's not in your PATH")

    monkeypatch.setattr(
        "phansora.products.spokenverse.txt_to_voice.pdf_pipeline.PdfConverter", Boom
    )

    with pytest.raises(pipeline.TerminalError) as err:
        asyncio.run(pipeline._ocr_pdf_to_doc({"id": 1, "name": "b"}, str(pdf)))
    assert "Tesseract OCR engine needs to be installed" in str(err.value)


# ------------------------------------------------------------- the bar itself
def test_progress_reporter_moves_the_bar_and_names_the_page(monkeypatch):
    writes = []

    async def fake_set_project(pid, **fields):
        writes.append(fields)

    monkeypatch.setattr(pipeline.db, "set_project", fake_set_project)
    report = pipeline._ocr_progress_reporter(7)

    async def drive():
        await report("ocr", 1, 1000)        # first tick always writes
        await report("ocr", 1000, 1000)     # a finished stage always writes
        await report("clean", 200, 200)

    asyncio.run(drive())

    assert writes[0]["progress"] == pipeline.OCR_PROGRESS_FLOOR
    assert "1/1,000" in writes[0]["stage"]
    assert writes[1]["progress"] == pipeline.OCR_PROGRESS_CEILING
    assert writes[2]["progress"] == pipeline.CLEAN_PROGRESS_CEILING
    assert "Cleaning recognized text" in writes[2]["stage"]
    # Never past its budget: chunking picks up from there.
    assert all(w["progress"] <= pipeline.CLEAN_PROGRESS_CEILING for w in writes)


def test_progress_writes_are_throttled(monkeypatch):
    """A page a second across four workers must not be a write a second."""
    writes = []

    async def fake_set_project(pid, **fields):
        writes.append(fields)

    monkeypatch.setattr(pipeline.db, "set_project", fake_set_project)
    report = pipeline._ocr_progress_reporter(7)

    async def drive():
        for i in range(1, 51):
            await report("ocr", i, 1000)

    asyncio.run(drive())
    assert len(writes) == 1  # the first; the rest fall inside the interval


def test_a_progress_write_failure_never_ends_the_book(monkeypatch):
    async def broken(pid, **fields):
        raise RuntimeError("db is down")

    monkeypatch.setattr(pipeline.db, "set_project", broken)
    report = pipeline._ocr_progress_reporter(7)
    asyncio.run(report("ocr", 1, 10))  # must not raise


def test_cache_lives_beside_the_source(tmp_path):
    assert pipeline.ocr_cache_dir(tmp_path / "source.pdf") == tmp_path / "ocr_cache"
