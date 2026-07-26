"""Real word timings for a narration — what makes storyboard scenes land on the voice.

Estimating a scene boundary from character counts and punctuation lands within a few
tenths of a second. That is fine when a scene is a chapter and plainly wrong when a scene
is five seconds long: a tenth here is a word, and the placeholder changes while the voice
is still mid-sentence. So rather than model how long the text *should* take, transcribe
what the voice *did* — with word-level timestamps — and align that transcript back onto
the script the user wrote, giving every word of the script a real second.

The transcript never matches the script exactly (mis-hearings, numbers written as digits,
dropped filler), so the two are matched as sequences and the words in between are
interpolated.

Nothing here falls back. A storyboard whose placeholders are *nearly* on the voice is not
a cheaper version of a correct one — it is the bug, and one that only shows up after the
user has dropped media into forty placeholders. So every failure raises: no model, an
unreadable clip, a transcript that does not resemble the script. The caller reports it and
the user fixes the cause.
"""
from __future__ import annotations

import logging
import os
import re
from difflib import SequenceMatcher
from threading import Lock
from typing import List, Optional, Tuple

logger = logging.getLogger("narrava-studio.align")

_WORD_RE = re.compile(r"[A-Za-z0-9']+")

_MODEL = None
_MODEL_NAME: Optional[str] = None
_MODEL_LOCK = Lock()

# Below this share of the script found in the transcript, the two are not the same
# recording — a stale upload, the wrong clip, a synthesis that dropped half its input.
# Inventing timings from that would be worse than the estimate it would replace.
_MIN_MATCH_RATIO = 0.6


class AlignmentUnavailable(RuntimeError):
    """The machinery isn't there — no faster-whisper, no model, a broken backend.

    Separate from AlignmentFailed because the fixes are different: this one is the
    operator's (install it, point at a model), and it is the same for every request.
    """


class AlignmentFailed(ValueError):
    """The audio and the script don't correspond, so no honest timing can come out of it."""


def _load_model():
    """faster-whisper, loaded once per process.

    Reads the same environment as SpokenVerse's transcriber but through its own name first
    (``NARRAVA_ALIGN_MODEL``), because the two want different things: transcription wants
    the best wording it can get, alignment only wants to know where each word sits and can
    take a smaller, faster model to get it.
    """
    global _MODEL, _MODEL_NAME
    name = (os.getenv("NARRAVA_ALIGN_MODEL") or os.getenv("WHISPER_MODEL") or "base").strip()
    with _MODEL_LOCK:
        if _MODEL is not None and _MODEL_NAME == name:
            return _MODEL
        try:
            from faster_whisper import WhisperModel  # lazy — the API still boots without it
        except Exception as exc:  # noqa: BLE001
            raise AlignmentUnavailable(
                "faster-whisper is not installed on the API host, so narration cannot be "
                "timed. Install the API requirements and restart it."
            ) from exc

        kwargs = {
            "device": os.getenv("WHISPER_DEVICE", "cpu"),
            "compute_type": os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        }
        threads = os.getenv("WHISPER_CPU_THREADS", "").strip()
        if threads.isdigit():
            kwargs["cpu_threads"] = max(1, int(threads))
        try:
            _MODEL = WhisperModel(name, **kwargs)
        except Exception as exc:  # noqa: BLE001 — a bad model name or no download
            raise AlignmentUnavailable(
                f"The speech model '{name}' could not be loaded on the API host: {exc}"
            ) from exc
        _MODEL_NAME = name
        return _MODEL


def _heard(audio_path: str, language: Optional[str]) -> List[Tuple[str, float, float]]:
    """``(word, start, end)`` for everything the model hears, in order."""
    model = _load_model()
    segments, _ = model.transcribe(
        audio_path,
        word_timestamps=True,
        # VAD cuts silence out and maps the timestamps back afterwards. We want the
        # timeline of the file exactly as it will play under the placeholders, including
        # its lead-in and the gaps between chunks, so it stays off.
        vad_filter=False,
        language=language or None,
        # Alignment needs positions, not the best possible wording — a wider beam costs
        # time and changes almost nothing about where the words fall.
        beam_size=1,
    )
    out: List[Tuple[str, float, float]] = []
    for segment in segments:
        for word in (getattr(segment, "words", None) or []):
            tokens = _WORD_RE.findall(str(getattr(word, "word", "")).lower())
            if not tokens:
                continue
            start, end = float(word.start), float(word.end)
            for token in tokens:  # a token that splits shares its parent's span
                out.append((token, start, end))
    return out


def word_times(
    audio_path: str,
    full_text: str,
    *,
    language: Optional[str] = None,
    total_duration_sec: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """One ``(start, end)`` per word of ``full_text``. Raises rather than approximating.

    The result is positional: element i belongs to the i-th match of the word pattern over
    ``full_text``, which is the same walk the storyboard uses to locate its scenes.
    """
    said = [m.group(0).lower() for m in _WORD_RE.finditer(full_text or "")]
    if not said:
        raise AlignmentFailed("There are no words in this narration script to time.")

    try:
        heard = _heard(audio_path, language)
    except AlignmentUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 — an unreadable or undecodable clip
        raise AlignmentFailed(
            f"The narration audio could not be read for timing: {exc}"
        ) from exc
    if not heard:
        raise AlignmentFailed(
            "No speech was found in the narration audio, so the storyboard has nothing to "
            "time itself against."
        )

    matcher = SequenceMatcher(None, said, [w for w, _, _ in heard], autojunk=False)
    times: List[Optional[Tuple[float, float]]] = [None] * len(said)
    matched = 0
    for i, j, size in matcher.get_matching_blocks():
        for k in range(size):
            times[i + k] = (heard[j + k][1], heard[j + k][2])
        matched += size

    ratio = matched / len(said)
    if ratio < _MIN_MATCH_RATIO:
        raise AlignmentFailed(
            f"Only {ratio * 100:.0f}% of the script was heard in the narration audio, so "
            "they are not the same recording. Re-voice the narration so the audio on the "
            "timeline matches the script, then build the storyboard again."
        )

    filled = _fill_gaps(times, total_duration_sec)
    if filled is None:
        raise AlignmentFailed("The narration audio produced no usable word timings.")
    logger.info("Narration aligned: %d words, %.0f%% heard directly", len(said), ratio * 100)
    return filled


def _fill_gaps(
    times: List[Optional[Tuple[float, float]]],
    total: Optional[float],
) -> Optional[List[Tuple[float, float]]]:
    """Interpolate the words the transcript missed, then force the result monotonic.

    A mis-heard word still has to carry a time, or the scene that starts on it has nothing
    to anchor to. Spreading the gap evenly across the run is crude but bounded — the words
    on either side are real, so the error can never exceed the gap itself.
    """
    known = [t for t in times if t]
    if not known:
        return None
    tail = float(total) if total else known[-1][1]

    out: List[Tuple[float, float]] = []
    i = 0
    while i < len(times):
        current = times[i]
        if current:
            out.append(current)
            i += 1
            continue
        run_end = i
        while run_end < len(times) and not times[run_end]:
            run_end += 1
        left = out[-1][1] if out else 0.0
        right = times[run_end][0] if run_end < len(times) else tail
        width = max(0.0, right - left)
        steps = run_end - i
        for k in range(steps):
            out.append((left + width * (k / steps), left + width * ((k + 1) / steps)))
        i = run_end

    # Whisper can emit a word that starts before the previous one ended; a clock that goes
    # backwards would put a scene before the one it follows.
    cleaned: List[Tuple[float, float]] = []
    floor = 0.0
    for start, end in out:
        start = max(floor, start)
        end = max(start, end)
        if total:
            start, end = min(start, float(total)), min(end, float(total))
        cleaned.append((start, end))
        floor = start
    return cleaned
