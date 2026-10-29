"""Where /txt-to-audio leaves its audio.

SpokenVerse keeps every generation here, for its history to play back. Narrava and Book
Alchemy keep their own copy, and until they could say so every narration and every lesson
also stayed in output_audio/ with nothing naming it: 112 files in one account by October.
The voice engine is faked: nothing here needs, or may use, the real model.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from phansora.products.spokenverse import server


class _Converter:
    """Writes one small "mp3" per .txt, as the real converter names them."""

    fail = False

    def __init__(self, cfg):
        self.cfg = cfg

    async def convert_folder(self, in_dir, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / ".tmp_chunks").mkdir(exist_ok=True)
        if _Converter.fail:
            return 1
        for txt in in_dir.glob("*.txt"):
            (out_dir / f"{txt.stem}.{self.cfg.output_format}").write_bytes(b"ID3 fake audio")
        return 0


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "BatchConverter", _Converter)
    monkeypatch.setattr(server, "unavailable_reason", lambda: None)
    _Converter.fail = False
    return TestClient(server.app)


def _render(client, user, **fields):
    return client.post(
        "/txt-to-audio",
        data={"user_id": user, **fields},
        files={"file": ("session.txt", b"A lesson about the stomach.", "text/plain")},
    )


def _left_in_tmp():
    return sorted(p.name for p in server.TMP_UPLOADS_DIR.iterdir() if p.name.startswith("out_"))


def test_audio_the_caller_keeps_is_sent_and_not_left_behind(client):
    res = _render(client, "501", keep="false")
    assert res.status_code == 200
    assert res.content == b"ID3 fake audio"
    assert not list((server.OUTPUT_AUDIO_DIR / "501").glob("*.mp3"))
    assert _left_in_tmp() == []


def test_spokenverse_audio_is_kept_for_its_history(client):
    res = _render(client, "502")
    assert res.status_code == 200
    assert (server.OUTPUT_AUDIO_DIR / "502" / "session.mp3").read_bytes() == b"ID3 fake audio"


def test_a_failed_render_leaves_nothing_behind(client):
    _Converter.fail = True
    res = _render(client, "503", keep="false")
    assert res.status_code == 500
    assert _left_in_tmp() == []
