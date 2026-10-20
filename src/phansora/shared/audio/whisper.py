"""faster-whisper, loaded once per process for every product that turns speech into words.

SpokenVerse and Narrava Studio each used to keep a model of their own, so a process serving
both could hold the same weights twice, and SpokenVerse decoded on the GPU without the gate
in shared/gpu.py — the one that keeps CUDA work out of the voice engine's graph capture.

``load`` keeps the models, one per (model name, device, compute type): two products asking
for "small" on the same device get the same copy. ``transcribe`` loads and decodes as a GPU
guest when whisper is on CUDA, and returns the segments already decoded. faster-whisper
hands back a generator and does the work as it is consumed, so a caller iterating it after
the gate was released would be decoding outside the gate.

Which model, and how to decode (beam, VAD, word timestamps), stay the caller's: transcribing
wants the best wording and alignment only wants positions. How the model is RUN is the
host's, from the environment, read at each call as the products did:

  WHISPER_DEVICE        cpu, cuda, cuda:N or auto (default cpu)
  WHISPER_COMPUTE_TYPE  default int8
  WHISPER_CPU_THREADS   blank = the library's default
  WHISPER_NUM_WORKERS   blank = the library's default
"""
from __future__ import annotations

import logging
import os
from collections import OrderedDict
from threading import Lock
from typing import Any, List, Tuple

from phansora.shared import gpu

logger = logging.getLogger("phansora.whisper")

# Before this module each product held one model, so the process held at most two, and two
# is all the defaults ever ask for: with WHISPER_MODEL set every caller shares one, and
# without it SpokenVerse asks for "small" and "base" and Narrava for "base". /audio-to-txt
# takes the model name from the request, so without a bound every model a user ever named
# would stay in memory. The least recently used one is dropped first.
_MAX_MODELS = 2
_MODELS: "OrderedDict[Tuple[str, str, str], Any]" = OrderedDict()
_LOCK = Lock()


class WhisperUnavailable(RuntimeError):
    """The machinery isn't there: no faster-whisper, or a model that would not load.

    The operator's to fix, and the same for every request. The message can be shown as
    written; the cause goes to the log.
    """


class WhisperNotInstalled(WhisperUnavailable):
    """faster-whisper itself is not installed on this host."""


def _device() -> str:
    return os.getenv("WHISPER_DEVICE", "cpu")


def on_gpu() -> bool:
    """Whether whisper runs on CUDA on this host, and so must be a guest of the GPU."""
    device = _device().strip().lower()
    return device.startswith("cuda") or (device == "auto" and gpu.cuda_available())


def load(model_name: str):
    """The faster-whisper model ``model_name`` on this host's device, loaded once.

    Loading allocates on the device, so on CUDA it belongs inside ``gpu.guest`` — which is
    where ``transcribe`` calls it.
    """
    device = _device()
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    key = (model_name, device, compute_type)
    with _LOCK:
        model = _MODELS.get(key)
        if model is not None:
            _MODELS.move_to_end(key)
            return model
        try:
            from faster_whisper import WhisperModel  # lazy — the API still boots without it
        except Exception as exc:  # noqa: BLE001
            raise WhisperNotInstalled(
                "faster-whisper is not installed on the API host, so audio cannot be "
                "transcribed. Install the API requirements and restart it."
            ) from exc

        kwargs = {"device": device, "compute_type": compute_type}
        threads = os.getenv("WHISPER_CPU_THREADS", "").strip()
        if threads.isdigit():
            kwargs["cpu_threads"] = max(1, int(threads))
        workers = os.getenv("WHISPER_NUM_WORKERS", "").strip()
        if workers.isdigit():
            kwargs["num_workers"] = max(1, int(workers))
        try:
            model = WhisperModel(model_name, **kwargs)
        except Exception as exc:  # noqa: BLE001 — a bad model name or no download
            # The message can reach a user as written, so the cause (cache paths, a
            # library's own error) goes to the log rather than into it.
            logger.error("Speech model %r failed to load", model_name, exc_info=True)
            raise WhisperUnavailable(
                f"The speech model '{model_name}' could not be loaded on the API host. "
                "The API log has the cause."
            ) from exc
        _MODELS[key] = model
        while len(_MODELS) > _MAX_MODELS:
            _MODELS.popitem(last=False)
        return model


def transcribe(audio_path, model_name: str, **options) -> Tuple[List[Any], Any]:
    """``(segments, info)`` for ``audio_path``, decoded with ``model_name`` and ``options``.

    ``options`` go to ``WhisperModel.transcribe`` as given. Raises ``gpu.GpuBusy`` if the
    voice engine is still loading, ``WhisperUnavailable`` if there is no model to decode
    with; anything else came from the audio.
    """
    with gpu.guest("transcription", active=on_gpu()):
        model = load(model_name)
        segments, info = model.transcribe(str(audio_path), **options)
        segments = list(segments)
        # Let go of it inside the gate: if the cache dropped this model meanwhile, this is
        # the last reference, and its device memory is freed here rather than after.
        del model
    return segments, info
