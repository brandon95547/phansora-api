"""A Book Alchemy URL source is fetched only through safe_fetch.

safe_fetch is what refuses file:// and private addresses. trafilatura.fetch_url used to
run before it, checking neither, so a link to a service on this host was read and handed
back to the caller as their book.
"""
from __future__ import annotations

import pytest

pytest.importorskip("trafilatura", reason="trafilatura not installed on this host")

from phansora.products.book_alchemy import parsers  # noqa: E402
from phansora.shared.net import safe_fetch  # noqa: E402


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8188/history",
    "http://169.254.169.254/latest/meta-data/",
    "file:///var/www/phansora-api/.env",
])
def test_a_link_to_this_host_is_refused_without_being_fetched(url, monkeypatch):
    import trafilatura

    monkeypatch.setattr(trafilatura, "fetch_url", lambda *a, **k: pytest.fail("trafilatura fetched it"))
    with pytest.raises(parsers.UnsupportedSourceError):
        parsers._parse_url(url, "")


def test_a_public_page_is_read_through_safe_fetch(monkeypatch):
    import trafilatura

    page = "<html><head><title>T</title></head><body><article><p>" + "Real words here. " * 40 + "</p></article></body></html>"
    seen = []
    monkeypatch.setattr(safe_fetch, "fetch_text", lambda url, **kw: seen.append(url) or page)
    monkeypatch.setattr(trafilatura, "fetch_url", lambda *a, **k: pytest.fail("trafilatura fetched it"))
    doc = parsers._parse_url("https://example.org/article", "")
    assert seen == ["https://example.org/article"]
    assert "Real words here." in " ".join(b.text for b in doc.blocks)
