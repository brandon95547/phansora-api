"""A piece the engine's number reader (wetext) rejects is read as written, not failed.

wetext asserts ``len(input) > 0`` on strings its tagger has no rule for ("IHC 3+"), and
that happens inside ``inference_*`` before any audio exists. The text is identical on
every retry, so on 2026-09-29 one pathology score failed a whole Book Alchemy book three
times over. ``_synthesize_sync`` now retries just that piece with ``text_frontend=False``;
these pin the two helpers it leans on.

Pure-function tests: no model, no CUDA, no CosyVoice checkout.
"""
from __future__ import annotations

import pytest

cv = pytest.importorskip(
    "phansora.products.spokenverse.txt_to_voice.adapters.cosyvoice3_client",
    reason="spokenverse adapter not importable on this host",
)


def _assert_from(filename: str) -> AssertionError:
    ns: dict = {}
    exec(compile("def load(s):\n    assert len(s) > 0\n", filename, "exec"), ns)
    try:
        ns["load"]("")
    except AssertionError as exc:
        return exc
    raise AssertionError("expected the stub to assert")


def test_an_assert_raised_inside_wetext_is_recognised():
    exc = _assert_from("/venv/site-packages/wetext/token_parser.py")
    assert cv._raised_in_text_normalizer(exc)


def test_any_other_assert_is_not():
    """CosyVoice3's missing-<|endofprompt|> assert must still fail loudly."""
    exc = _assert_from("/var/www/CosyVoice/cosyvoice/llm/llm.py")
    assert not cv._raised_in_text_normalizer(exc)


class _FakeCosy:
    def __init__(self):
        self.calls = []

    def inference_zero_shot(self, text, prompt_text, prompt_wav, **kw):
        self.calls.append(("zero_shot", kw["text_frontend"]))
        yield {"tts_speech": "audio"}

    def inference_instruct2(self, text, instruct_text, prompt_wav, **kw):
        self.calls.append(("instruct2", kw["text_frontend"]))
        yield {"tts_speech": "audio"}


@pytest.mark.parametrize("instruct, method", [(None, "zero_shot"), ("Speak calmly.", "instruct2")])
def test_render_piece_passes_text_frontend_through(instruct, method):
    cosy = _FakeCosy()
    assert cv._render_piece(cosy, "IHC 3+", instruct, "cond", "spk", 1.0) == ["audio"]
    assert cv._render_piece(cosy, "IHC 3+", instruct, "cond", "spk", 1.0, text_frontend=False) == ["audio"]
    assert cosy.calls == [(method, True), (method, False)]
