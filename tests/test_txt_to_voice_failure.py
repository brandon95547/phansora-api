"""What a failed chunk does to the rest of its file.

On 2026-09-27 one chunk of a Book Alchemy lesson failed. The file was lost, but the
lesson's other chunks kept rendering on the shared GPU for minutes afterwards, and
the only record of the cause was "Failed converting session.txt:" — the exception
had no message and no traceback was logged. The engine is faked throughout: nothing
here needs, or may use, the real voice model.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

from phansora.products.spokenverse.txt_to_voice.adapters.cosyvoice3_client import SynthesisStopped
from phansora.products.spokenverse.txt_to_voice.pipeline import BatchConverter, TTSConfig, _describe

PARAGRAPHS = [f"Paragraph {i} about the stomach and its neighbours." for i in range(8)]


def make_converter(synth) -> BatchConverter:
    """A converter without __init__, which would resolve the real engine module."""
    conv = object.__new__(BatchConverter)
    conv.cfg = TTSConfig(
        voice="default", use_gpu=False, rate="+0%", volume="+0%", output_format="mp3",
        chunk_chars=60, max_concurrency=2, engine="cosyvoice3",
    )
    conv._synthesize = synth
    return conv


def test_one_failed_chunk_stops_the_rest_and_names_the_failure(tmp_path, caplog):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / "session.txt").write_text("\n\n".join(PARAGRAPHS), encoding="utf-8")

    started, stopped, finished = [], [], []

    async def synth(*, text, out_path, should_stop, **_kw):
        started.append(text)
        if "Paragraph 0" in text:
            await asyncio.sleep(0.02)
            raise RuntimeError()           # no message, like the real one

        def render():                      # the real engine runs in a worker thread,
            for _piece in range(100):      # which a task cancel cannot reach
                if should_stop():
                    stopped.append(text)
                    raise SynthesisStopped("stopped")
                time.sleep(0.005)
            finished.append(text)

        await asyncio.to_thread(render)

    conv = make_converter(synth)
    with caplog.at_level(logging.ERROR):
        rc = asyncio.run(conv.convert_folder(in_dir, tmp_path / "out"))

    assert rc == 1
    # The chunk running beside the failure stopped at its next piece...
    assert stopped and not finished
    # ...and the ones still waiting for a slot never started.
    assert len(started) < len(PARAGRAPHS)
    # The log names the failure and carries its traceback.
    record = next(r for r in caplog.records if "Failed converting" in r.getMessage())
    assert record.getMessage().endswith("session.txt: RuntimeError")
    assert record.exc_info is not None


def test_describe_never_returns_blank():
    assert _describe(RuntimeError()) == "RuntimeError"
    assert _describe(AssertionError()) == "AssertionError"
    assert _describe(ValueError("bad voice")) == "ValueError: bad voice"
