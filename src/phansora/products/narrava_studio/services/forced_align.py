"""Where each word of a known text is actually sung — read off the audio, not guessed.

The rest of the alignment machinery infers timing from a TRANSCRIPT: whisper says what it
heard and when, and the user's words are matched onto that. Everything whisper mis-hears
therefore has no time of its own and is interpolated, and whisper's own word boundaries are
a by-product of decoding rather than a measurement. On speech that is good enough. On
singing it is not, and the two failures compound:

  · A held note looks exactly like a rest. "stars" is sung for 1.1 seconds; the syllable
    rule allows a one-syllable word 0.25s, concludes the other 0.85s must be silence in
    front of it, and starts the word 0.85s late. Measured on the prod song, the karaoke
    highlight lit that word 1.5s after it was sung.
  · A line whisper never wrote down has nothing to anchor to at all.

This module answers the question directly instead. A CTC acoustic model scores every
20ms frame against every character; given the characters the text is made of, the best
monotonic path through that grid is where each one is sung. It is a measurement, and it
needs no transcript to be right.

MEASURED on the prod song (4:52, a 116-word sheet) against known onsets: "holding" 31.82
against a true 31.83, "stars" 44.48 against ~44.40, the last word 87.88 against ~88.0 —
where the shipped path had them at 31.00, 46.07 and adrift. It runs in about a second on
the box's GPU.

WHAT IT STILL NEEDS THE TRANSCRIPT FOR is a window. A sheet almost never covers a whole
song — this one covers 90 seconds of 293 — and the model is free to skip audio no word
accounts for, which over three and a half unaccounted minutes means it will happily jump
a minute to find a slightly better match for a word whose wording is off. Unwindowed, the
sheet's last line landed at 151-225s instead of 87-93s. Bounded by where the transcript
found the sheet's words, every one of them lands. So this refines an alignment; it does not
replace one, and it is never the only thing standing between a user and their captions.
"""
from __future__ import annotations

import logging
import os
import subprocess
from threading import Lock
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger("narrava-studio.forced-align")

SAMPLE_RATE = 16000

# Emissions are computed in chunks so a long song cannot decide how much memory this takes.
# 60s was measured on the prod song; the model's receptive field is far shorter than that,
# so the seam costs nothing a word would notice.
_CHUNK_SEC = 60.0

# How far outside the transcript's own idea of where the sheet sits the model may look.
# Measured: anywhere from 1s to 3s gives the same answer to the centisecond, and the result
# only moves once the window runs far enough past the last word for the trailing skip to
# start competing with it.
_WINDOW_PAD_SEC = 1.5

# Below this, on the mean over a LINE, the model is saying it cannot find those words in
# this audio — a line the singer does not sing, or a sheet out of step with the song.
#
# A line, not a word, because a per-word threshold measures the wrong thing. Sung scores run
# far below spoken ones (a held vowel, vibrato, a band underneath), and the lowest are
# always the short function words — on the prod song the bottom eight were "of", "world",
# "of", "by", "I", "the"... which are hard to pin acoustically, not wrong. Any per-word
# threshold loose enough to catch a genuinely wrong line also marked 40% of a correct one.
#
# MEASURED on that song's 22 lines: the eighteen that match the singing score 0.22 to 0.82,
# and four consecutive lines the singer plainly does not sing score 0.038 to 0.081. There is
# an order of magnitude between them, and 0.15 sits in the gap.
_UNSURE_LINE_SCORE = 0.15

_MODEL = None
_BUNDLE = None
_LOCK = Lock()


class ForcedAlignUnavailable(RuntimeError):
    """No torch, no torchaudio, no model, or a device that will not take it."""


def enabled() -> bool:
    """Whether to try at all. Off by name, so a host can fall back to the transcript."""
    return (os.getenv("NARRAVA_FORCED_ALIGN", "1").strip().lower()
            not in {"0", "false", "no", "off"})


def _device() -> str:
    import torch

    name = os.getenv("NARRAVA_FORCED_ALIGN_DEVICE", "").strip()
    if name:
        return name
    # The GPU on this box is shared with the TTS engine, which preallocates. CUDA is used
    # when it is there and left alone when it is not; a CPU run is slower but still exact.
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load():
    """The MMS forced-alignment bundle, loaded once per process."""
    global _MODEL, _BUNDLE
    with _LOCK:
        if _MODEL is not None:
            return _MODEL, _BUNDLE
        try:
            import torch  # noqa: F401
            from torchaudio.pipelines import MMS_FA as bundle
        except Exception as exc:  # noqa: BLE001
            raise ForcedAlignUnavailable(
                "torchaudio's forced-alignment pipeline is not available on this host."
            ) from exc
        try:
            model = bundle.get_model().to(_device()).eval()
        except Exception as exc:  # noqa: BLE001 — no weights, no GPU memory, a bad device
            raise ForcedAlignUnavailable(
                f"The forced-alignment model could not be loaded: {exc}"
            ) from exc
        _MODEL, _BUNDLE = model, bundle
        logger.info("Forced aligner ready on %s", _device())
        return _MODEL, _BUNDLE


def _decode(path: str, start: float, end: float):
    """Mono 16k float32 for ``[start, end)`` of ``path``, as a 1xN tensor."""
    import numpy as np
    import torch

    argv = ["ffmpeg", "-v", "error"]
    if start > 0:
        argv += ["-ss", f"{start:.3f}"]
    argv += ["-i", path]
    if end > start:
        argv += ["-t", f"{end - start:.3f}"]
    argv += ["-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    raw = subprocess.run(argv, capture_output=True, check=True).stdout
    if not raw:
        raise ForcedAlignUnavailable("The audio could not be decoded for alignment.")
    return torch.from_numpy(np.frombuffer(raw, dtype="float32").copy()).unsqueeze(0)


def spellable(word: str) -> str:
    """``word`` as the model's alphabet can hold it, or '' if nothing is left.

    The MMS dictionary is 26 letters, an apostrophe and a hyphen. A word that survives as
    nothing — a bare number, a symbol — is not passed in at all; the caller gives it a time
    between its neighbours instead, which is what happened to every word before this.
    """
    kept = [c for c in word.lower().replace("’", "'") if c.isascii() and (c.isalpha() or c in "'-")]
    return "".join(kept).strip("'-")


def align(
    audio_path: str,
    words: Sequence[str],
    *,
    window: Optional[Tuple[float, float]] = None,
    total_duration_sec: Optional[float] = None,
) -> List[Optional[Tuple[float, float, float]]]:
    """``(start, end, score)`` per word of ``words``, in seconds of the FILE — or None.

    One entry per word given, positional. None where the word held nothing the model's
    alphabet can spell, so the caller keeps whatever it had for that one.

    ``window`` bounds the search. Pass one whenever the text may not cover the whole
    recording, which for a lyric sheet is almost always — see the note at the top of this
    module for what happens without it.
    """
    import torch

    model, bundle = _load()
    spelled = [spellable(w) for w in words]
    keep = [i for i, w in enumerate(spelled) if w]
    if not keep:
        raise ForcedAlignUnavailable("None of these words can be spelled for alignment.")

    lo = 0.0
    hi = float(total_duration_sec) if total_duration_sec else 0.0
    if window:
        lo = max(0.0, window[0] - _WINDOW_PAD_SEC)
        hi = window[1] + _WINDOW_PAD_SEC
        if total_duration_sec:
            hi = min(hi, float(total_duration_sec))
    wav = _decode(audio_path, lo, hi)

    device = _device()
    with torch.inference_mode():
        chunk = int(_CHUNK_SEC * SAMPLE_RATE)
        parts = []
        for i in range(0, wav.size(1), chunk):
            piece = wav[:, i:i + chunk]
            if piece.size(1) < SAMPLE_RATE // 10:   # a sliver decodes to nothing useful
                continue
            emission, _ = model(piece.to(device))
            parts.append(emission)
        if not parts:
            raise ForcedAlignUnavailable("There is not enough audio here to align against.")
        emission = torch.cat(parts, dim=1)

        # A star token, free to emit, absorbing every stretch of audio no word accounts for
        # — the instrumental bars between verses, and whatever the sheet leaves out. Without
        # it the model has to put a word on every second of the recording.
        star = torch.zeros((1, emission.size(1), 1), device=emission.device, dtype=emission.dtype)
        emission = torch.cat((emission, star), 2)
        star_id = emission.size(2) - 1

        tokens = bundle.get_tokenizer()([spelled[i] for i in keep])
        padded: List[List[int]] = [[star_id]]
        where: List[int] = []
        for token in tokens:
            where.append(len(padded))
            padded.append(list(token))
            padded.append([star_id])
        spans = bundle.get_aligner()(emission[0], padded)

    ratio = wav.size(1) / emission.size(1) / SAMPLE_RATE
    out: List[Optional[Tuple[float, float, float]]] = [None] * len(words)
    for i, k in zip(keep, where):
        span = spans[k]
        # Averaged over the frames the tokens actually occupy, NOT over the word's whole
        # span: the blank frames between two letters belong to neither, and dividing by
        # them deflated every score far enough to mark a clean word unsure.
        held = sum(t.end - t.start for t in span) or 1
        score = sum(t.score * (t.end - t.start) for t in span) / held
        out[i] = (lo + span[0].start * ratio, lo + span[-1].end * ratio, float(score))
    return out


def unsure(line_score: float) -> bool:
    """Whether the model failed to find a LINE in the audio, so the cue is worth reading."""
    return line_score < _UNSURE_LINE_SCORE
