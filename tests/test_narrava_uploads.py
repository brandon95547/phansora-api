"""Narrava Studio's audio uploads (phansora.products.narrava_studio.server).

These arrive from the Node app directly rather than through nginx, so nginx's body limit
never applies: the cap has to be here, and a refused upload must leave nothing behind.
"""
from __future__ import annotations

import glob
import os
import tempfile

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed on this host")

from fastapi.testclient import TestClient  # noqa: E402

from phansora.products.narrava_studio import server  # noqa: E402
from phansora.products.narrava_studio.services import align  # noqa: E402


def _leftovers():
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "narrava_*")))


def test_an_upload_past_the_cap_is_refused_and_removed(monkeypatch):
    monkeypatch.setattr(server, "MAX_MEDIA_BYTES", 10)
    monkeypatch.setattr(align, "aligned_words", lambda *a, **k: pytest.fail("aligned an oversized upload"))
    before = _leftovers()
    client = TestClient(server.app, raise_server_exceptions=False)
    res = client.post(
        "/narration/align",
        files={"file": ("n.mp3", b"x" * 11, "audio/mpeg")},
        data={"full_text": "hello there"},
    )
    assert res.status_code == 413
    assert _leftovers() == before


def test_an_upload_under_the_cap_reaches_the_aligner(monkeypatch):
    seen = {}

    def fake(path, full_text, **kw):
        with open(path, "rb") as fh:
            seen["bytes"] = fh.read()
        return {"words": [[0.0, 0.5]], "guessed": [False], "heard_ratio": 1.0}

    monkeypatch.setattr(align, "aligned_words", fake)
    before = _leftovers()
    client = TestClient(server.app, raise_server_exceptions=False)
    res = client.post(
        "/narration/align",
        files={"file": ("n.mp3", b"audio-bytes", "audio/mpeg")},
        data={"full_text": "hello"},
    )
    assert res.status_code == 200, res.text
    assert seen["bytes"] == b"audio-bytes"
    assert _leftovers() == before
