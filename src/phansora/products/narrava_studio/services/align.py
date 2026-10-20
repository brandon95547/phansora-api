"""Real word timings for a narration — what makes storyboard scenes land on the voice.

Estimating a scene boundary from character counts and punctuation lands within a few
tenths of a second. That is fine when a scene is a chapter and plainly wrong when a scene
is five seconds long: a tenth here is a word, and the placeholder changes while the voice
is still mid-sentence. So rather than model how long the text *should* take, transcribe
what the voice *did* — with word-level timestamps — and align that transcript back onto
the script the user wrote, giving every word of the script a real second.

The transcript never matches the script exactly (mis-hearings, numbers written as digits,
dropped filler), so the two are matched as sequences and the words in between are
interpolated. The matching happens in a SPOKEN-FORM space: "1969" and "nineteen sixty-nine"
are the same narration wearing two spellings, and before both sides are expanded to the
spoken form they simply never matched — which turned every date, count and sum of money
into an interpolated stretch, and interpolated stretches are exactly where captions and
scene cuts drift off the voice.

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
from typing import Any, Dict, List, Optional, Tuple

from phansora.shared import gpu
from phansora.shared.audio import whisper

logger = logging.getLogger("narrava-studio.align")

# THE tokenizer for narration text — storyboard.py imports this rather than keeping its own,
# because the two must agree on the word COUNT exactly or the timings are rejected as
# mismatched. Curly apostrophes are in the class and normalized away below: a script written
# by an LLM says "didn’t" while whisper transcribes "didn't", and splitting the first into
# "didn" + "t" both changed the count and stopped every contraction from matching.
WORD_RE = re.compile(r"[A-Za-z0-9'’]+")
_WORD_RE = WORD_RE



def normalize(word: str) -> str:
    """Lowercase, and one apostrophe to rule them all."""
    return word.lower().replace("’", "'")


# ── Spoken forms ─────────────────────────────────────────────────────────────
# A narration script says "1969" and the voice says "nineteen sixty-nine"; whisper writes
# back whichever form it feels like. Token-for-token matching treats those as total
# strangers, so every number in the script used to open an interpolation gap — bounded, but
# wide enough to put a caption a second off the voice through any dated, counted or priced
# sentence. Both sequences are therefore expanded into their spoken form before matching,
# with each expanded piece remembering which original word it came from.
#
# The spelling is deliberately plain American with no "and" and no hyphens — every piece
# must itself be a single WORD_RE token, because the heard side is tokenized the same way
# ("sixty-nine" arrives as "sixty", "nine").

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10 ** 9, "billion"), (10 ** 6, "million"), (10 ** 3, "thousand"), (100, "hundred")]

# Ordinal irregulars; anything else is cardinal + "th" (with y -> ieth).
_ORDINAL = {
    "one": "first", "two": "second", "three": "third", "five": "fifth",
    "eight": "eighth", "nine": "ninth", "twelve": "twelfth",
}
_ORD_RE = re.compile(r"^(\d+)(st|nd|rd|th)$")

# Past this a "number" is a serial or an id, which nobody reads aloud as a cardinal —
# expanding it would just manufacture tokens that cannot match anything.
_MAX_NUMBER_DIGITS = 12


def _under_hundred(n: int) -> List[str]:
    if n < 20:
        return [_ONES[n]]
    tens, ones = divmod(n, 10)
    return [_TENS[tens]] + ([_ONES[ones]] if ones else [])


def _spell_cardinal(n: int) -> List[str]:
    if n == 0:
        return ["zero"]
    out: List[str] = []
    for value, name in _SCALES:
        if n >= value:
            count, n = divmod(n, value)
            out += _spell_cardinal(count) + [name]
    if n:
        out += _under_hundred(n)
    return out


def _spell_year(n: int) -> List[str]:
    """How a year is actually said, which is not how a cardinal is said.

    1969 is "nineteen sixty nine", never "one thousand nine hundred sixty nine" — and the
    whole point of expansion is to land on the form the VOICE used, so the year form wins
    for anything that reads as a year. Round hundreds keep their own idiom (1900 is
    "nineteen hundred"), the 2000s go through cardinal ("two thousand five"), and an
    early-decade year says "oh" ("nineteen oh five").
    """
    century, yy = divmod(n, 100)
    if n % 1000 == 0:
        return _spell_cardinal(n)
    if yy == 0:
        return _under_hundred(century) + ["hundred"]
    if century == 20 and yy < 10:
        return _spell_cardinal(n)
    return _under_hundred(century) + (["oh", _ONES[yy]] if yy < 10 else _under_hundred(yy))


def _ordinalize(words: List[str]) -> List[str]:
    last = words[-1]
    if last in _ORDINAL:
        words[-1] = _ORDINAL[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return words


def _spoken_forms(token: str) -> List[str]:
    """One normalized token -> the words a voice makes of it. Most tokens pass through."""
    if token.isdigit():
        if len(token) > _MAX_NUMBER_DIGITS:
            return [token]
        n = int(token)
        # 1000–2999 reads as a year far more often than as a cardinal in narration, and
        # matching is forgiving of the guess being wrong: a mismatch here just leaves the
        # word interpolated, which is exactly where it stood before expansion existed.
        return _spell_year(n) if 1000 <= n <= 2999 else _spell_cardinal(n)
    m = _ORD_RE.match(token)
    if m and len(m.group(1)) <= _MAX_NUMBER_DIGITS:
        return _ordinalize(_spell_cardinal(int(m.group(1))))
    return [token]


def _expand(tokens: List[str]) -> Tuple[List[str], List[int]]:
    """Tokens -> (spoken-form tokens, index of the original word each came from)."""
    exp: List[str] = []
    src: List[int] = []
    for i, token in enumerate(tokens):
        for piece in _spoken_forms(token):
            exp.append(piece)
            src.append(i)
    return exp, src


# ── Matching a text that repeats itself ───────────────────────────────────────
# SequenceMatcher takes the LONGEST run the two sides share first, then matches what is left
# on either side of it separately. On a text that says the same thing more than once that
# first choice can be the wrong copy, and nothing afterwards can undo it. A song's chorus is
# exactly that: on the prod song, whisper heard the LAST chorus cleanly enough for it to be
# the longest run, SequenceMatcher paired it with the sheet's FIRST chorus — 80 seconds
# against 248 — and every verse between them was left with nothing on the other side to
# match. 81% of the sheet found became 27%, the sheet was refused as the wrong song, and
# whether that happened depended only on how cleanly whisper heard one chorus that run.
#
# So the anchors are chosen by the alignment that matches the most words IN ORDER, which
# can never pair the first chorus with the last because that would cost it every verse in
# between. Runs of _ANCHOR_RUN words or more from it are kept as anchors; SequenceMatcher
# then matches the stretches between them exactly as it always did, but bounded, so it can
# only ever choose among the copies that sit between the same two anchors.
_ANCHOR_RUN = 3


def _ordered_runs(a: List[str], b: List[str]) -> List[Tuple[int, int, int]]:
    """``(i, j, size)`` runs of the longest in-order matching of ``a`` against ``b``.

    The textbook longest-common-subsequence table, a row at a time in numpy: a song sheet
    against its transcript is ~600 x 600, a long narration ~4,000 x 4,000, and both are
    milliseconds this way where a Python double loop over the second is seconds.
    """
    import numpy as np

    if not a or not b:
        return []
    vocab: Dict[str, int] = {}
    a_ids = np.array([vocab.setdefault(w, len(vocab)) for w in a])
    b_ids = np.array([vocab.setdefault(w, len(vocab)) for w in b])
    n, m = len(a), len(b)
    table = np.zeros((n + 1, m + 1), dtype=np.int16 if min(n, m) < 32000 else np.int32)
    for i in range(1, n + 1):
        # best[i][j] = max(best[i-1][j], best[i-1][j-1] + same, best[i][j-1]); the last
        # term is a running maximum along the row.
        step = np.maximum(table[i - 1, 1:], table[i - 1, :-1] + (b_ids == a_ids[i - 1]))
        table[i, 1:] = np.maximum.accumulate(step)

    pairs: List[Tuple[int, int]] = []
    i, j = n, m
    while i and j:
        if a[i - 1] == b[j - 1] and table[i, j] == table[i - 1, j - 1] + 1:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif table[i - 1, j] >= table[i, j - 1]:
            i -= 1
        else:
            j -= 1
    pairs.reverse()

    runs: List[Tuple[int, int, int]] = []
    for i, j in pairs:
        if runs and runs[-1][0] + runs[-1][2] == i and runs[-1][1] + runs[-1][2] == j:
            runs[-1] = (runs[-1][0], runs[-1][1], runs[-1][2] + 1)
        else:
            runs.append((i, j, 1))
    return runs


def _matching_blocks(a: List[str], b: List[str]) -> List[Tuple[int, int, int]]:
    """SequenceMatcher's matching blocks, anchored so a repeated passage cannot cross-pair."""
    blocks: List[Tuple[int, int, int]] = []
    at_a = at_b = 0
    anchors = [run for run in _ordered_runs(a, b) if run[2] >= _ANCHOR_RUN]
    for i, j, size in anchors + [(len(a), len(b), 0)]:
        if i > at_a and j > at_b:
            between = SequenceMatcher(None, a[at_a:i], b[at_b:j], autojunk=False)
            blocks.extend(
                (at_a + x, at_b + y, n) for x, y, n in between.get_matching_blocks() if n
            )
        if size:
            blocks.append((i, j, size))
        at_a, at_b = i + size, j + size
    return blocks


def _anchored(
    said: List[str],
    heard: List[Tuple[str, float, float]],
) -> Tuple[List[Optional[Tuple[float, float]]], float]:
    """Match script words to heard words in spoken-form space.

    Returns one ``(start, end)`` or ``None`` per SCRIPT word, plus the share of script
    words that found at least one anchor. A script word whose expansion matched several
    heard pieces ("1969" against "nineteen sixty nine") spans from the first piece's start
    to the last piece's end — which is when the year was actually being said.
    """
    s_exp, s_src = _expand(said)
    h_exp, h_src = _expand([w for w, _, _ in heard])
    times: List[Optional[Tuple[float, float]]] = [None] * len(said)
    for i, j, size in _matching_blocks(s_exp, h_exp):
        for k in range(size):
            original = s_src[i + k]
            _, start, end = heard[h_src[j + k]]
            current = times[original]
            times[original] = (
                (start, end) if current is None
                else (min(current[0], start), max(current[1], end))
            )
    ratio = (sum(1 for t in times if t) / len(said)) if said else 0.0
    return times, ratio


# A stretch SequenceMatcher calls a "replace" is the sheet's words opposite the wrong words —
# and on a song the wrong words are usually the right ones misheard, sung at the same seconds.
# On the prod song whisper wrote "Played on the, yeah, we been" for "They don't know where
# we've been", one word for one, at exactly the seconds it was sung. Treating the sheet's side
# as unheard threw those seconds away and spread the line across the rest in front of it.
#
# Only where the two sides are close in length. Five sheet words opposite two heard ones is
# not a stand-in, it is a line the transcript missed with something else heard nearby, and
# pairing them would squeeze the whole line into two words' time.
_STAND_IN_RATIO = 0.5


def _stand_ins(
    said: List[str],
    heard: List[Tuple[str, float, float]],
    times: List[Optional[Tuple[float, float]]],
) -> List[Optional[Tuple[float, float]]]:
    """``times`` with each word whisper MISHEARD given the time of the word heard in its place.

    Positional within the stretch, so a line keeps its rests: the third sheet word takes the
    third heard word's time rather than a share of the whole stretch. Words ``_anchored``
    matched keep their time, and a word with no stand-in stays None for the gap filler.
    Never counted as heard — ``_anchored``'s ratio is what says whether this is the song.
    """
    s_exp, s_src = _expand(said)
    h_exp, h_src = _expand([w for w, _, _ in heard])
    found: Dict[int, Tuple[float, float]] = {}
    # The stretches between two matched blocks that are unmatched on BOTH sides — what
    # SequenceMatcher's opcodes call a "replace" — from the same blocks _anchored used.
    gaps = []
    at_s = at_h = 0
    for i, j, size in _matching_blocks(s_exp, h_exp) + [(len(s_exp), len(h_exp), 0)]:
        if i > at_s and j > at_h:
            gaps.append((at_s, i, at_h, j))
        at_s, at_h = i + size, j + size
    for i1, i2, j1, j2 in gaps:
        n, m = i2 - i1, j2 - j1
        if min(n, m) / max(n, m) < _STAND_IN_RATIO:
            continue
        for k in range(n):
            word = s_src[i1 + k]
            if times[word] is not None:
                continue
            _, start, end = heard[h_src[j1 + k * m // n]]
            current = found.get(word)
            found[word] = (start, end) if current is None else (
                min(current[0], start), max(current[1], end))
    out = list(times)
    for word, span in found.items():
        out[word] = span
    return out


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


def _model_name() -> str:
    """Which whisper model Narrava decodes with — for narration and for songs alike.

    Reads the same environment as SpokenVerse's transcriber but through its own name first
    (``NARRAVA_ALIGN_MODEL``), because the two want different things: transcription wants
    the best wording it can get, alignment only wants to know where each word sits and can
    take a smaller, faster model to get it.
    """
    return (os.getenv("NARRAVA_ALIGN_MODEL") or os.getenv("WHISPER_MODEL") or "base").strip()


def _transcribe(audio_path: str, **options) -> list:
    """Whisper's segments for ``audio_path``, decoded with ``options``.

    The model is the process's shared one (shared/audio/whisper.py), and the decode runs as
    a guest of the GPU when whisper is on one: the voice engine captures CUDA graphs when it
    loads, and a decode landing inside that capture breaks the whole process's CUDA context
    — see shared/gpu.py. Transcription waits for the load rather than walking into it.
    """
    try:
        segments, _ = whisper.transcribe(audio_path, _model_name(), **options)
    except gpu.GpuBusy as busy:
        raise AlignmentUnavailable(str(busy)) from busy
    except whisper.WhisperUnavailable as exc:
        raise AlignmentUnavailable(str(exc)) from exc
    return segments


def _heard(audio_path: str, language: Optional[str]) -> List[Tuple[str, float, float]]:
    """``(word, start, end)`` for everything the model hears, in order."""
    segments = _transcribe(
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
        # Whisper's classic long-audio failure: one hallucinated segment becomes the prompt
        # for the next, the wording walks off the script for a stretch, and every word in
        # that stretch fails to match — so its timings are interpolated, and the captions
        # visibly lag the voice there while fitting perfectly elsewhere. Our narrations are
        # chunked TTS with real silences between chunks, which is precisely the audio that
        # sets the cascade off. Each window is decoded cold instead; alignment does not
        # need cross-window context, only positions.
        condition_on_previous_text=False,
    )
    out: List[Tuple[str, float, float]] = []
    for segment in segments:
        for word in (getattr(segment, "words", None) or []):
            tokens = [normalize(t) for t in _WORD_RE.findall(str(getattr(word, "word", "")))]
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
    return aligned_words(
        audio_path, full_text, language=language, total_duration_sec=total_duration_sec,
    )["words"]


def aligned_words(
    audio_path: str,
    full_text: str,
    *,
    language: Optional[str] = None,
    total_duration_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """``word_times`` plus which of those times were heard and which were placed.

    Between the 60% floor and a perfect match, the words the transcript missed are
    interpolated between the ones it caught — bounded, and usually close, but a guess. The
    caller used to get the two kinds mixed together with no way to tell them apart, so a
    caption sitting on a stretch of guessed timing looked exactly as certain as one on a
    measured word. ``guessed`` is one boolean per word, positional like ``words``, and the
    UI marks a cue built on any of them. The pair travels as two parallel lists rather than
    a wider tuple because ``words`` is round-tripped into the storyboard request, whose
    model pins each entry to exactly two floats.
    """
    said = [normalize(m.group(0)) for m in _WORD_RE.finditer(full_text or "")]
    if not said:
        raise AlignmentFailed("There are no words in this narration script to time.")

    try:
        heard = _heard(audio_path, language)
    except AlignmentUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 — an unreadable or undecodable clip
        logger.warning("Narration audio could not be decoded", exc_info=True)
        raise AlignmentFailed("The narration audio could not be read for timing.") from exc
    if not heard:
        raise AlignmentFailed(
            "No speech was found in the narration audio, so the storyboard has nothing to "
            "time itself against."
        )

    times, ratio = _anchored(said, heard)
    if ratio < _MIN_MATCH_RATIO:
        raise AlignmentFailed(
            f"Only {ratio * 100:.0f}% of the script was heard in the narration audio, so "
            "they are not the same recording. Re-voice the narration so the audio on the "
            "timeline matches the script, then build the storyboard again."
        )

    filled = _fill_gaps(times, total_duration_sec, weights=[len(w) for w in said])
    if filled is None:
        raise AlignmentFailed("The narration audio produced no usable word timings.")
    logger.info("Narration aligned: %d words, %.0f%% heard directly", len(said), ratio * 100)
    return {
        "words": filled,
        "guessed": [t is None for t in times],
        "heard_ratio": ratio,
    }


def _around_the_break(
    first: int,
    last: int,
    left: float,
    right: float,
    weights: Optional[List[int]],
    paces: List[float],
    breaks: List[bool],
) -> Optional[List[Tuple[float, float]]]:
    """Words ``first..last-1`` placed between ``left`` and ``right`` with the rest at the line
    break — or None when no line breaks among them, or they will not fit at their pace.

    A singer rests between lines, not inside one. Spread evenly, a run that ends one line and
    starts the next puts the rest INSIDE both, so the next line's card goes up while the
    singer is still silent: on the prod song "They don't know" came out 1.4s before it was
    sung. So the words finishing the left word's line follow it at their pace, the words
    starting the right word's line lead into it at theirs, and any whole lines nobody heard
    share what is left between them.
    """
    cuts = [b for b in range(first - 1, last) if b >= 0 and breaks[b]]
    if not cuts:
        return None
    head = range(first, cuts[0] + 1)
    middle = range(cuts[0] + 1, cuts[-1] + 1)
    tail = range(cuts[-1] + 1, last)
    lead_in = sum(paces[k] for k in tail)
    if sum(paces[k] for k in head) + lead_in > right - left:
        return None
    out: List[Tuple[float, float]] = []
    at = left
    for k in head:
        out.append((at, at + paces[k]))
        at += paces[k]
    room = max(0.0, right - lead_in - at)
    share = [max(1, weights[k]) if weights else 1 for k in middle]
    whole = sum(share) or 1
    acc = 0
    for piece in share:
        start = at + room * (acc / whole)
        acc += piece
        out.append((start, at + room * (acc / whole)))
    at = right - lead_in
    for k in tail:
        out.append((at, at + paces[k]))
        at += paces[k]
    return out


def _fill_gaps(
    times: List[Optional[Tuple[float, float]]],
    total: Optional[float],
    weights: Optional[List[int]] = None,
    paces: Optional[List[float]] = None,
    breaks: Optional[List[bool]] = None,
) -> Optional[List[Tuple[float, float]]]:
    """Interpolate the words the transcript missed, then force the result monotonic.

    A mis-heard word still has to carry a time, or the scene that starts on it has nothing
    to anchor to. The gap is split in proportion to each missing word's LENGTH rather than
    evenly — "a" and "extraordinarily" do not take the same breath, and an even split put
    the second half of every interpolated run measurably late. Still crude, still bounded:
    the words on either side are real, so the error can never exceed the gap itself.

    ``paces`` — how long each word plausibly takes to say — closes the one case where that
    bound does not hold. A run with nothing anchored BEFORE it has only one real edge, and
    stretching it from second zero of the file is not interpolation, it is invention: on the
    prod song whisper missed the whole first sung line, and its six words were spread from
    0.00s across a 28-second instrumental intro, so the first caption card sat on screen
    through the entire introduction with the karaoke highlight crawling through silence.
    Given paces, such a run is placed by how long its words take instead — backwards from
    the first word that WAS heard, and forwards from the last one at the other end. Measured
    on that song: the run moves from 0.00-32.30 to 29.80-32.30, against a true 28.16-31.68.

    Without ``paces`` the old behaviour stands, which is right for a narration: the voice
    starts when the recording does, so second zero is a real edge there.

    ``breaks`` — one per word, whether a line break follows it — is for a sung text, and
    moves the rest in a run between two anchors to the line break it spans (see
    _around_the_break). Without it, or with no break in the run, the run is spread as above.
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
        if paces:
            # An OPEN edge — the start of the file, or its end — is not evidence of anything,
            # so it is replaced by where these words can actually have been said. Never past
            # the anchor at the other end, and never outside the file.
            span = sum(paces[i:run_end])
            if not out and run_end < len(times):
                left = max(0.0, right - span)
            elif out and run_end >= len(times):
                right = min(tail, left + span)
        placed = (
            _around_the_break(i, run_end, left, right, weights, paces, breaks)
            if breaks and paces and out and run_end < len(times) else None
        )
        if placed is None:
            width = max(0.0, right - left)
            share = [max(1, weights[k]) if weights else 1 for k in range(i, run_end)]
            whole = sum(share)
            acc = 0
            placed = []
            for piece in share:
                start = left + width * (acc / whole)
                acc += piece
                placed.append((start, left + width * (acc / whole)))
        out.extend(placed)
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


# ── Lyrics ────────────────────────────────────────────────────────────────────
# A music video has no script. The words are the song's, and the only record of them is
# the recording — so this TRANSCRIBES, the one thing the aligner above will not do, and
# returns the transcript with a time per word in the same positional shape the narration
# path returns for its script. Everything downstream (caption cards, the storyboard's
# clock) takes it unchanged.
#
# What it cannot know is whether the words were heard right. Singing over a band is the
# hardest audio whisper meets, so every word it was unsure of comes back marked, through
# the same `guessed` list the aligner uses for interpolated words. The editor already flags
# a caption built on one, and here that flag lands on exactly the cards worth reading
# against the song.

# Below this, whisper's own confidence in a word marks it for checking. Words it heard
# clearly sit well above it; the misheard ones cluster low.
_UNSURE_PROBABILITY = 0.5

# Whisper's captions for sound it could not write down as words: "[Music]", "(upbeat
# music)", "(laughs)", a run of ♪. Right in a subtitle file, wrong here, where they would
# become caption cards reading "Music" and lyrics for the storyboard to illustrate. A
# parenthesis is only a note when it names a sound; "(yeah)" is a backing vocal and stays.

# The sounds whisper names when it cannot write words down. Named, because the pattern below
# looks for them twice: inside brackets, and bare on a line of their own.
_SOUNDS = (
    r"music|applause|laugh\w*|instrumental|silence|inaudible|singing|humming|cheer\w*"
    r"|noise|beat"
)
# The words whisper reaches for when it is describing a sound rather than writing one down.
# Deliberately tight, and deliberately without articles or common verbs: they are only ever
# used to decide that a whole line is a stage direction, so anything a song might sing in
# the same breath as "music" has to stay out of this list.
_STAGE = (
    r"outro|intro|interlude|continues|playing|fades|fading|background|distant|muffled"
    r"|softly|throughout|offstage|reprise"
)
_NOTE_RE = re.compile(
    r"\[[^\]]*\]"
    r"|\((?=[^)]*\b(?:" + _SOUNDS + r")\b)[^)]*\)"
    # The SAME note with its brackets missing, which whisper writes just as often: a line
    # reading "Music", or "Outro music playing". Left in, those become caption cards — the
    # prod song produced both, and the second landed at 27.9s, right on top of the first
    # line actually sung.
    #
    # A whole line only, and only when EVERY word in it is a sound or a stage direction.
    # That is what keeps "The music plays" — a line a song could really sing — while taking
    # "Outro music playing", which no song sings: "the" and "plays" are in neither list, so
    # a lyric that merely mentions music survives.
    r"|(?m:^[ \t]*(?:(?:" + _SOUNDS + r"|" + _STAGE + r")\b[ \t,]*)+\.?[ \t]*$)"
    r"|[♪♫]+",
    re.IGNORECASE,
)
_BRACKET_RE = re.compile(r"[\[\]()]")

# ── Vocal onsets ──────────────────────────────────────────────────────────────
# Whisper never leaves a gap between two words of the same segment: word N's end IS word
# N+1's start. On speech that is harmless, because the words really do run together. On a
# song it is not. The rests between sung phrases are real, and whisper folds each one into
# the FRONT of the word that follows it — that word comes back with a span several times
# longer than it is sung for, starting where the previous word ended. The karaoke highlight
# lights the last word to have STARTED, so it lights at the beginning of the rest and sits
# there through the silence, naming a word nobody is singing yet.
#
# So a word that whisper ran straight on from the one before it, and gave more time than it
# could plausibly take to sing, is assumed to have a rest at its front and is started a
# plausible duration before its end instead.
#
# Only where there was NO gap. A rest whisper DID report is one it actually heard, and that
# start is honest — on the song below just 34 of 428 words had a real gap in front of them,
# so this costs almost nothing and keeps the model's own answer wherever it gave one.
#
# MEASURED on a 4:52 song (428 words) against torchaudio forced alignment: "Holding" came
# back as 30.20-32.18 when it is sung at 31.83; this puts it at 31.68, which is 0.15s out
# instead of 1.63s. Over the whole song it moves the share of words landing within 0.3s of
# their true onset from 77% to 83%, and the median error from 0.13s to 0.10s.
#
# What it cannot do is tell a rest from a genuinely sustained note, where the onset really
# is at the start and the long span is honest. That is the case it makes worse — the worst
# single word went from 1.90s out to 2.85s. Reading the audio is the only way to know the
# difference, which is what a forced aligner does; raising the threshold so this fires only
# on the longest spans was measured and does NOT help (it costs more than it saves).
_SEC_PER_SYLLABLE = 0.25
# Whisper's timestamps are quantised to 20ms, so "ran straight on" needs that much room.
_CONTIGUOUS_EPS = 0.02
_VOWEL_RUN_RE = re.compile(r"[aeiouy]+")
_NOT_LETTER_RE = re.compile(r"[^a-z]")


def _syllables(word: str) -> int:
    """Roughly how many syllables ``word`` has — enough to say how long it takes to sing."""
    letters = _NOT_LETTER_RE.sub("", word.lower())
    if not letters:
        return 1
    runs = len(_VOWEL_RUN_RE.findall(letters))
    # A trailing silent "e" is not a syllable ("time"), unless it is the only vowel run
    # ("the") or part of one that is always sounded ("little", "free", "bye").
    if letters.endswith("e") and not letters.endswith(("le", "ee", "ye")) and runs > 1:
        runs -= 1
    return max(1, runs)


def _onsets(
    times: List[Tuple[float, float]], words: List[str]
) -> List[Tuple[float, float]]:
    """Start each word at its likely onset rather than at the end of the one before it."""
    out: List[Tuple[float, float]] = []
    for i, ((start, end), word) in enumerate(zip(times, words)):
        # The previous END as whisper gave it. Only `start` is ever moved below, so reading
        # the original list rather than `out` cannot matter — but it says which it means.
        ran_on = i > 0 and start - times[i - 1][1] <= _CONTIGUOUS_EPS
        plausible = _syllables(word) * _SEC_PER_SYLLABLE
        if ran_on and end - start > plausible:
            # Never past its own end, and never earlier than whisper already had it.
            start = max(start, end - plausible)
        out.append((start, end))
    return out


def lyric_words(
    audio_path: str,
    *,
    language: Optional[str] = None,
    total_duration_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """The words sung on a music track, and when each one is sung.

    ``text`` is the transcript as whisper wrote it, cased and punctuated, one sung line per
    line. ``words`` is one ``(start, end)`` per WORD_RE match over ``text`` — the positional
    contract the narration path keeps with its script. ``guessed`` is one boolean per word,
    true where the model was unsure of it. Raises rather than returning nothing: a track
    with no voice on it is the user's to fix, and saying so beats an empty caption track.
    """
    try:
        pieces = _sung_pieces(_sung_decode(audio_path, language))
    except AlignmentUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 — an unreadable or undecodable file
        logger.warning("Music track could not be decoded", exc_info=True)
        raise AlignmentFailed("The music track could not be read for transcription.") from exc
    result = _transcript(pieces, total_duration_sec)
    logger.info(
        "Lyrics transcribed: %d words, %d unsure", len(result["words"]), sum(result["guessed"])
    )
    return result


def _sung_decode(audio_path: str, language: Optional[str]) -> list:
    """Whisper's segments for a music track.

    The same model narration is aligned with. A larger one would hear singing better, but
    this is the one the box already runs; a word it is unsure of is flagged instead.
    """
    return _transcribe(
        audio_path,
        word_timestamps=True,
        # Off, as in the aligner: captions and scenes are laid on the file's own timeline,
        # gaps and all. Silero's VAD is a speech detector too, and a voice under a band is
        # exactly where it misfires.
        vad_filter=False,
        language=language or None,
        # Unlike alignment this wants the best WORDING — nothing corrects it afterwards —
        # so the wider beam earns its time here.
        beam_size=5,
        # A song is the worst case for whisper's prompt cascade: choruses repeat, one
        # hallucinated line becomes the prompt for the next window, and the lyrics walk off
        # the recording for a verse. Each window is decoded cold.
        condition_on_previous_text=False,
    )


def _sung_pieces(segments) -> List[Tuple[str, float, float, float, bool]]:
    """``(text, start, end, probability, opens_line)`` per word whisper wrote down.

    ``text`` is the word as whisper spelled it, leading space and punctuation included, so
    the pieces join back into the transcript. A segment is roughly one sung line, so its
    first word opens a line. A piece with no word in it is dropped unless it could be part
    of a note: whisper splits "[Music]" into " [", "Music", "]" as readily as it keeps it
    whole, and a bracket thrown away here would leave "Music" standing as a lyric.
    """
    out: List[Tuple[str, float, float, float, bool]] = []
    for segment in segments:
        first = True
        for word in (getattr(segment, "words", None) or []):
            text = str(getattr(word, "word", "") or "")
            if not (_WORD_RE.search(text) or _NOTE_RE.search(text) or _BRACKET_RE.search(text)):
                continue
            probability = getattr(word, "probability", None)
            out.append((
                text,
                float(word.start),
                float(word.end),
                1.0 if probability is None else float(probability),
                first,
            ))
            first = False
    return out


def _join(pieces) -> Tuple[str, List[Tuple[int, int]]]:
    """The pieces as one transcript, and the character span each piece occupies in it."""
    text = ""
    spans: List[Tuple[int, int]] = []
    for word, _start, _end, _probability, opens_line in pieces:
        if not text:
            word = word.lstrip()
        elif opens_line:
            word = "\n" + word.lstrip()
        spans.append((len(text), len(text) + len(word)))
        text += word
    return text, spans


def _transcript(pieces, total: Optional[float]) -> Dict[str, Any]:
    """Pieces -> ``{text, words, guessed}``, with words positional over ``text``.

    The time of a word is read off the pieces its characters came from, rather than by
    tokenizing each piece on its own: whisper splits "I'm" into " I" + "'m", which
    tokenizes as two words piecewise and as ONE across the joined text. The joined text is
    what everything downstream tokenizes, so it is what the timings have to follow.
    """
    text, spans = _join(pieces)
    notes = [m.span() for m in _NOTE_RE.finditer(text)]
    if notes:
        pieces = [
            piece for piece, (a, b) in zip(pieces, spans)
            if not any(a < note_end and note_start < b for note_start, note_end in notes)
        ]
        text, spans = _join(pieces)

    words: List[Tuple[float, float]] = []
    guessed: List[bool] = []
    spelled: List[str] = []
    k = 0
    for match in _WORD_RE.finditer(text):
        a, b = match.span()
        while k < len(spans) and spans[k][1] <= a:
            k += 1
        start = end = None
        unsure = False
        j = k
        while j < len(spans) and spans[j][0] < b:
            _, piece_start, piece_end, probability, _ = pieces[j]
            start = piece_start if start is None else min(start, piece_start)
            end = piece_end if end is None else max(end, piece_end)
            unsure = unsure or probability < _UNSURE_PROBABILITY
            j += 1
        words.append((start, end))
        guessed.append(unsure)
        spelled.append(match.group(0))

    if not words:
        raise AlignmentFailed(
            "No singing was found on the music track. Captions and the storyboard are made "
            "from the words that are sung, so they need a song with vocals — an "
            "instrumental has no words to work from."
        )
    # Every word is known here, so _fill_gaps is only the monotonic clean-up and the clamp
    # to the file's length — whisper's timestamps can still run backwards between windows.
    # _onsets runs first and only ever moves a start LATER, never past its own end, so it
    # cannot disturb the ordering that clean-up then enforces.
    return {
        "text": text,
        "words": _fill_gaps(_onsets(words, spelled), total),
        "guessed": guessed,
    }


# ── Known lyrics ──────────────────────────────────────────────────────────────
# Transcribing a song is the fallback, not the goal. A band is the hardest audio whisper
# meets — on the 4:52 song this was built against it marked 79 of 430 words unsure and
# wrote "Maskin' on my feet" for a line nobody sang — and a caption card with the wrong
# words in it reads as a caption in the wrong PLACE, because the viewer is matching what
# they hear to what they see. So when the user has the lyrics, they are the words, and the
# recording is only asked WHEN each one is sung.
#
# That is the narration path exactly: a known text, matched onto what was heard, with the
# words in between interpolated. The one difference is that a narration script is read once
# and a lyric sheet is sung — so this goes through the song transcriber (wider beam, notes
# stripped, vocal onsets recovered) rather than the aligner's, and it forgives a lower
# match, because a quarter of a song coming back mis-heard is normal and a quarter of a
# narration coming back mis-heard means the wrong file was uploaded.

# A lyric sheet is written for a person, so it carries furniture a caption must not show:
# section markers on their own line, and the blank lines between verses.
_SECTION_RE = re.compile(r"(?m)^[ \t]*[\[(][^\])]{0,60}[\])][ \t]*$")
_BLANKS_RE = re.compile(r"\n[ \t]*\n+")

# How much of the sheet has to be found in the singing before the two are the same song.
# Lower than narration's 0.6 on purpose: measured on the prod song, whisper heard about
# three quarters of the words it was given well enough to match, and the floor has only one
# job — catching a sheet pasted for a different track, which lands near zero because
# SequenceMatcher pays for RUNS of words and stray "the"s do not make runs.
_MIN_LYRIC_MATCH_RATIO = 0.35


def clean_lyrics(text: str) -> str:
    """A pasted lyric sheet as the words that are actually sung, one line per sung line.

    Line breaks survive, because a lyric line IS a caption card — the caption builder
    breaks on them (see captions.js). Everything else a sheet carries for the reader goes.
    """
    out = _SECTION_RE.sub("", str(text or ""))
    out = "\n".join(line.strip() for line in out.split("\n"))
    out = _BLANKS_RE.sub("\n", out)
    return out.strip()


def _line_breaks(text: str) -> List[bool]:
    """One per WORD_RE match over ``text``: whether a line break follows that word."""
    matches = list(_WORD_RE.finditer(text))
    return [
        i + 1 < len(matches) and "\n" in text[m.end():matches[i + 1].start()]
        for i, m in enumerate(matches)
    ]


def _measured(
    audio_path: str,
    text: str,
    inferred: List[Tuple[float, float]],
    *,
    evidence: Optional[List[Optional[Tuple[float, float]]]] = None,
    total_duration_sec: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """The same words, timed by reading the audio — or None to keep what came from the
    transcript.

    ``evidence`` is what the transcript actually yielded per word — heard, or heard as a
    stand-in — with None where it yielded nothing. It is what a line the model cannot find
    falls back to; absent, ``inferred`` stands in for it.

    Never raises. This is an improvement on an answer that already exists, so every way it
    can fail (no torchaudio, no weights, no GPU memory, an audio file ffmpeg will not open)
    has the same correct outcome: the caller keeps the inferred timings and the user gets
    captions. It is logged rather than surfaced, because there is nothing for the user to do
    about it.
    """
    from . import forced_align

    if not forced_align.enabled() or not inferred:
        return None
    words = [m.group(0) for m in _WORD_RE.finditer(text)]
    try:
        spans = forced_align.align(
            audio_path,
            words,
            # Where the transcript found this sheet. The aligner is otherwise free to place
            # a word anywhere in the song, and on a sheet covering a third of a track it
            # took that freedom.
            window=(min(t[0] for t in inferred), max(t[1] for t in inferred)),
            total_duration_sec=total_duration_sec,
        )
    except Exception as exc:  # noqa: BLE001 — every failure keeps the inferred answer
        logger.info("Forced alignment unavailable, keeping the inferred timings: %s", exc)
        return None

    # Whether to tell the user to read a cue is decided a LINE at a time. The model's
    # confidence in a single sung word is too noisy to act on — see forced_align — but over
    # a line it separates cleanly: on the prod song the lines that match score 0.22 to 0.82
    # and the four the singer does not sing score 0.038 to 0.081.
    line_of = [text.count("\n", 0, m.start()) for m in _WORD_RE.finditer(text)]
    scores: Dict[int, List[float]] = {}
    for line, span in zip(line_of, spans):
        scores.setdefault(line, []).append(0.0 if span is None else span[2])
    weak = {
        line: forced_align.unsure(sum(v) / len(v)) for line, v in scores.items() if v
    }
    guessed = [span is None or weak.get(line, False) for line, span in zip(line_of, spans)]

    # A word the model's alphabet cannot spell is left for the gap-filler to place between
    # its MEASURED neighbors. Handing it the time the transcript gave it would look like
    # the obvious thing and is wrong: the two are different readings of the recording, and
    # dropping one of whisper's numbers into a run of measured ones puts it out of order as
    # often as not — which the monotonic pass then resolves by flattening the word to zero
    # width. Interpolating between two measurements keeps one frame of reference.
    #
    # A LINE the model could not find is the opposite case, and takes the transcript's
    # times. Its spans are not a measurement — a score that low says the model found
    # nothing there, and the path through those frames is wherever the other lines left room
    # for it. On the prod song every sung chorus scored 0.02-0.11 (a held, layered vocal over
    # the band, which a speech model has no letters for) and its lines landed 2-7 seconds off
    # the singing, while whisper had heard all of them, if not always the words. There is no
    # measurement in that line to keep consistent with, so the transcript's answer is the
    # best one there is.
    fallback = evidence if evidence is not None else inferred
    out: List[Optional[Tuple[float, float]]] = []
    for k, (line, span) in enumerate(zip(line_of, spans)):
        if weak.get(line, False):
            out.append(fallback[k])
        else:
            out.append(None if span is None else (span[0], span[1]))

    # Two readings of the recording still disagree by a few tenths, so a transcript time is
    # only kept where it falls between the measured words either side of it. Past them it
    # would push a measured line out of place on the monotonic pass; dropped, the word is
    # placed between its neighbors like any other.
    measured = [not weak.get(line, False) and span is not None for line, span in zip(line_of, spans)]
    after = 0.0
    for k in range(len(out)):
        if measured[k]:
            after = out[k][1]
        elif out[k] is not None and out[k][0] < after:
            out[k] = None
    before = float("inf")
    for k in range(len(out) - 1, -1, -1):
        if measured[k]:
            before = out[k][0]
        elif out[k] is not None and out[k][0] > before:
            out[k] = None

    cleaned = _fill_gaps(
        out,
        total_duration_sec,
        weights=[len(w) for w in words],
        paces=[_syllables(w) * _SEC_PER_SYLLABLE for w in words],
        breaks=_line_breaks(text),
    )
    if cleaned is None:
        return None
    logger.info(
        "Lyrics measured against the audio: %d words, %d the model could not find",
        len(cleaned), sum(guessed),
    )
    return {"words": cleaned, "guessed": guessed}


def lyric_alignment(
    audio_path: str,
    lyrics: str,
    *,
    language: Optional[str] = None,
    total_duration_sec: Optional[float] = None,
) -> Dict[str, Any]:
    """When each word of a KNOWN lyric sheet is sung on ``audio_path``.

    Returns the same shape ``lyric_words`` does — ``text``, one ``(start, end)`` per
    WORD_RE match over it, and a positional ``guessed`` — so the caption builder and the
    storyboard take it without knowing which of the two produced it. ``text`` is the
    CLEANED sheet rather than the one that was passed in, since that is what the timings
    are positional over; the caller must caption that text, not its own copy.

    ``guessed`` here means a word the singing did not yield and the gap placed, which is
    the same thing it means for a narration — not, as in the transcribed path, a word the
    model was unsure it heard right. Both are "check this one by ear", which is all the
    editor does with it.
    """
    said_text = clean_lyrics(lyrics)
    said = [normalize(m.group(0)) for m in _WORD_RE.finditer(said_text)]
    if not said:
        raise AlignmentFailed(
            "There are no words in the lyrics saved for this song, so there is nothing to "
            "time against it."
        )

    # The transcript, exactly as the no-lyrics path would have used it — notes stripped,
    # vocal onsets recovered. Those onsets are the timings this hands back for every word
    # it matches, so the correction has to happen before the matching, not after.
    heard_transcript = lyric_words(
        audio_path, language=language, total_duration_sec=total_duration_sec
    )
    heard = [
        (normalize(m.group(0)), float(start), float(end))
        for m, (start, end) in zip(
            _WORD_RE.finditer(heard_transcript["text"]), heard_transcript["words"]
        )
    ]

    times, ratio = _anchored(said, heard)
    if ratio < _MIN_LYRIC_MATCH_RATIO:
        raise AlignmentFailed(
            f"Only {ratio * 100:.0f}% of the saved lyrics could be found in this song, so "
            "they do not look like the same track. Check the lyrics on the song, or clear "
            "them to caption from the singing instead."
        )

    # The words whisper misheard still say WHEN they were sung — see _stand_ins. They are
    # not counted as heard (the ratio above is what decides this is the song), and they stay
    # marked for checking if they reach a caption.
    evidence = _stand_ins(said, heard, times)

    # A sheet's words are SUNG, so how long each takes is knowable — the same syllable pace
    # the onset recovery above uses. It is what keeps a run the transcript missed at either
    # end of the sheet from being stretched across an instrumental intro or outro, and,
    # with the sheet's line breaks, what puts a rest between lines rather than inside one.
    filled = _fill_gaps(
        evidence,
        total_duration_sec,
        weights=[len(w) for w in said],
        paces=[_syllables(w) * _SEC_PER_SYLLABLE for w in said],
        breaks=_line_breaks(said_text),
    )
    if filled is None:
        raise AlignmentFailed("The song produced no usable word timings for these lyrics.")

    # Everything above is inference from a TRANSCRIPT: whisper's word boundaries where it
    # heard the word, and a guess between them where it did not. Now read the answer off
    # the audio instead — which is a measurement, and on singing the difference is not
    # small (see services/forced_align.py). The work above is not wasted: it is what says
    # WHERE in the recording this sheet sits, and the aligner needs that window or it will
    # wander off into a later chorus — and it is the answer for any line the aligner then
    # cannot find.
    measured = _measured(
        audio_path, said_text, filled, evidence=evidence, total_duration_sec=total_duration_sec,
    )
    if measured is not None:
        return {
            "text": said_text,
            "words": measured["words"],
            "guessed": measured["guessed"],
            "heard_ratio": ratio,
        }
    logger.info(
        "Lyrics aligned: %d words, %.0f%% heard directly", len(said), ratio * 100
    )
    return {
        "text": said_text,
        "words": filled,
        "guessed": [t is None for t in times],
        "heard_ratio": ratio,
    }
