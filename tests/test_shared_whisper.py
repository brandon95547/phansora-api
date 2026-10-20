"""One faster-whisper for the whole process, run as a guest of the shared GPU.

SpokenVerse and Narrava Studio used to load whisper separately, so a process serving both
could hold the same weights twice, and SpokenVerse decoded on CUDA without the gate that
keeps work out of the voice engine's graph capture. What is pinned here: both products
decode with one copy of a model, each still with its own options; the cache is keyed by
model, device and compute type and stays bounded; on CUDA the decode itself happens inside
the gate; and each product still words a missing or broken whisper its own way.

faster-whisper is replaced by a stand-in, so these run anywhere.
"""
import sys
from collections import OrderedDict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import HTTPException

from phansora.shared import gpu
from phansora.shared.audio import whisper


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    """The cache and the gate are process-wide; each test gets its own, and a CPU host."""
    monkeypatch.setattr(whisper, "_MODELS", OrderedDict())
    monkeypatch.setattr(gpu, "_guests", 0, raising=False)
    monkeypatch.setattr(gpu, "_exclusive", False, raising=False)
    monkeypatch.setattr(gpu, "_waiting", 0, raising=False)
    for name in ("WHISPER_DEVICE", "WHISPER_COMPUTE_TYPE", "WHISPER_CPU_THREADS",
                 "WHISPER_NUM_WORKERS", "WHISPER_MODEL", "WHISPER_BEAM_SIZE",
                 "WHISPER_VAD_FILTER", "NARRAVA_ALIGN_MODEL"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def fake(monkeypatch):
    """faster_whisper as a stand-in that records what was built and what each decode saw."""
    record = SimpleNamespace(built=[], calls=[], guests_during_decode=[])

    class WhisperModel:
        def __init__(self, name, **kwargs):
            if name == "nope":
                raise OSError("no such model on the hub")
            self.name, self.kwargs = name, kwargs
            record.built.append(self)

        def transcribe(self, path, **options):
            record.calls.append((path, options))

            def decode():
                # A generator, as faster-whisper's is: the work happens as it is consumed.
                record.guests_during_decode.append(gpu._guests)
                yield SimpleNamespace(
                    text=" Hello there.",
                    words=[
                        SimpleNamespace(word=" Hello", start=0.0, end=0.4, probability=0.9),
                        SimpleNamespace(word=" there.", start=0.5, end=0.9, probability=0.9),
                    ],
                )

            return decode(), SimpleNamespace(language="en")

    module = ModuleType("faster_whisper")
    module.WhisperModel = WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return record


def _spokenverse():
    from phansora.products.spokenverse import server

    return server


def _align():
    from phansora.products.narrava_studio.services import align

    return align


def test_both_products_decode_with_one_copy_of_the_model(fake, monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "small")

    text = _spokenverse()._transcribe_audio_to_text_sync(Path("talk.wav"), "small", None)
    heard = _align()._heard("narration.wav", "en")
    lyrics = _align().lyric_words("song.m4a")

    assert text == "Hello there."
    assert heard == [("hello", 0.0, 0.4), ("there", 0.5, 0.9)]
    assert lyrics["text"] == "Hello there."
    assert [m.name for m in fake.built] == ["small"]


def test_each_product_keeps_its_own_decode_options(fake):
    _spokenverse()._transcribe_audio_to_text_sync(Path("talk.wav"), "small", None)
    _align()._heard("narration.wav", "en")

    (sv_path, sv), (na_path, na) = fake.calls
    assert sv_path == "talk.wav"
    assert sv == {"beam_size": 5, "vad_filter": True, "language": None}
    assert na_path == "narration.wav"
    assert na == {"word_timestamps": True, "vad_filter": False, "language": "en",
                  "beam_size": 1, "condition_on_previous_text": False}


def test_narrava_picks_its_model_by_its_own_name_first(fake, monkeypatch):
    _align()._heard("a.wav", None)
    monkeypatch.setenv("WHISPER_MODEL", "small")
    _align()._heard("a.wav", None)
    monkeypatch.setenv("NARRAVA_ALIGN_MODEL", "tiny")
    _align()._heard("a.wav", None)

    assert [m.name for m in fake.built] == ["base", "small", "tiny"]


def test_a_model_is_kept_per_name_device_and_compute_type(fake, monkeypatch):
    monkeypatch.setattr(whisper, "_MAX_MODELS", 10)
    small = whisper.load("small")
    assert whisper.load("small") is small
    assert whisper.load("base") is not small
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")
    on_cuda = whisper.load("small")
    monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "float16")
    in_fp16 = whisper.load("small")

    assert len({id(m) for m in (small, on_cuda, in_fp16)}) == 3
    assert (on_cuda.kwargs["device"], in_fp16.kwargs["compute_type"]) == ("cuda", "float16")


def test_the_cache_is_bounded_and_drops_the_least_recently_used(fake):
    """/audio-to-txt takes the model name from the request, so an unbounded cache would keep
    every model a user ever named."""
    small = whisper.load("small")
    whisper.load("base")
    assert whisper.load("small") is small  # touched, so "base" is now the oldest
    whisper.load("tiny")

    assert whisper.load("small") is small
    assert [key[0] for key in whisper._MODELS] == ["tiny", "small"]


def test_the_hosts_settings_reach_the_model(fake, monkeypatch):
    monkeypatch.setenv("WHISPER_CPU_THREADS", "4")
    monkeypatch.setenv("WHISPER_NUM_WORKERS", "2")

    assert whisper.load("small").kwargs == {
        "device": "cpu", "compute_type": "int8", "cpu_threads": 4, "num_workers": 2,
    }


def test_on_cuda_the_decode_itself_runs_inside_the_gate(fake, monkeypatch):
    """The segments come back already decoded: iterated after the gate was released, the
    generator would do its work outside it."""
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")
    _spokenverse()._transcribe_audio_to_text_sync(Path("talk.wav"), "small", None)
    _align()._heard("narration.wav", None)

    assert fake.guests_during_decode == [1, 1]
    assert gpu._guests == 0


def test_on_cpu_whisper_is_not_a_gpu_guest(fake):
    _spokenverse()._transcribe_audio_to_text_sync(Path("talk.wav"), "small", None)

    assert fake.guests_during_decode == [0]


def test_on_cuda_neither_product_loads_or_decodes_during_the_voice_engine_load(fake, monkeypatch):
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")
    monkeypatch.setenv("PHANSORA_GPU_GUEST_WAIT_SEC", "0.05")

    with gpu.exclusive("the CosyVoice load"):
        with pytest.raises(HTTPException) as refused:
            _spokenverse()._transcribe_audio_to_text_sync(Path("talk.wav"), "small", None)
        with pytest.raises(_align().AlignmentUnavailable, match="still loading"):
            _align()._heard("narration.wav", None)

    assert refused.value.status_code == 503
    assert "still loading the voice engine" in refused.value.detail
    assert refused.value.headers == {"Retry-After": "60"}
    assert fake.built == [] and fake.calls == []


def test_without_faster_whisper_each_product_says_so_in_its_own_words(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)

    with pytest.raises(HTTPException) as refused:
        _spokenverse()._transcribe_audio_to_text_sync(Path("talk.wav"), "small", None)
    assert refused.value.status_code == 500
    assert refused.value.detail.startswith("faster-whisper is not installed. Install dependencies")
    with pytest.raises(_align().AlignmentUnavailable, match="not installed on the API host"):
        _align()._heard("narration.wav", None)


def test_a_model_that_will_not_load_is_named_and_not_kept(fake, monkeypatch):
    with pytest.raises(whisper.WhisperUnavailable, match="'nope' could not be loaded"):
        whisper.transcribe("talk.wav", "nope")
    monkeypatch.setenv("NARRAVA_ALIGN_MODEL", "nope")
    with pytest.raises(_align().AlignmentUnavailable, match="'nope' could not be loaded"):
        _align()._heard("narration.wav", None)

    assert not whisper._MODELS
