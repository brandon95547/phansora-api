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
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

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
    matcher = SequenceMatcher(None, s_exp, h_exp, autojunk=False)
    times: List[Optional[Tuple[float, float]]] = [None] * len(said)
    for i, j, size in matcher.get_matching_blocks():
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


# Loaded models, by name. Two can be resident at once — the aligner's small one and the
# larger one lyrics need (see _load_lyrics_model) — and a narration caption straight after a
# music-video caption must not unload one to load the other.
_MODELS: Dict[str, Any] = {}
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


def _model_named(name: str):
    """faster-whisper model ``name``, loaded once per process."""
    with _MODEL_LOCK:
        if name in _MODELS:
            return _MODELS[name]
        try:
            from faster_whisper import WhisperModel  # lazy — the API still boots without it
        except Exception as exc:  # noqa: BLE001
            raise AlignmentUnavailable(
                "faster-whisper is not installed on the API host, so audio cannot be "
                "transcribed. Install the API requirements and restart it."
            ) from exc

        kwargs = {
            "device": os.getenv("WHISPER_DEVICE", "cpu"),
            "compute_type": os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        }
        threads = os.getenv("WHISPER_CPU_THREADS", "").strip()
        if threads.isdigit():
            kwargs["cpu_threads"] = max(1, int(threads))
        try:
            model = WhisperModel(name, **kwargs)
        except Exception as exc:  # noqa: BLE001 — a bad model name or no download
            raise AlignmentUnavailable(
                f"The speech model '{name}' could not be loaded on the API host: {exc}"
            ) from exc
        _MODELS[name] = model
        return model


def _load_model():
    """The aligner's model.

    Reads the same environment as SpokenVerse's transcriber but through its own name first
    (``NARRAVA_ALIGN_MODEL``), because the two want different things: transcription wants
    the best wording it can get, alignment only wants to know where each word sits and can
    take a smaller, faster model to get it.
    """
    return _model_named(
        (os.getenv("NARRAVA_ALIGN_MODEL") or os.getenv("WHISPER_MODEL") or "base").strip()
    )


def _load_lyrics_model():
    """The model that transcribes a song — the one job here that needs the best wording.

    Alignment knows the words already and only asks where they fall, so a small model does.
    Lyrics have no script behind them: the transcript IS the captions and the storyboard's
    text, and singing over a band is the hardest audio whisper meets. ``base`` mishears it
    badly, so this has its own setting (``NARRAVA_LYRICS_MODEL``) and a far larger default.
    large-v3-turbo is large-v3's encoder with a four-layer decoder: close to its accuracy on
    sung English, a fraction of its decode time, about 1.6 GB at float16.
    """
    return _model_named((os.getenv("NARRAVA_LYRICS_MODEL") or "large-v3-turbo").strip())


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
        raise AlignmentFailed(
            f"The narration audio could not be read for timing: {exc}"
        ) from exc
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


def _fill_gaps(
    times: List[Optional[Tuple[float, float]]],
    total: Optional[float],
    weights: Optional[List[int]] = None,
) -> Optional[List[Tuple[float, float]]]:
    """Interpolate the words the transcript missed, then force the result monotonic.

    A mis-heard word still has to carry a time, or the scene that starts on it has nothing
    to anchor to. The gap is split in proportion to each missing word's LENGTH rather than
    evenly — "a" and "extraordinarily" do not take the same breath, and an even split put
    the second half of every interpolated run measurably late. Still crude, still bounded:
    the words on either side are real, so the error can never exceed the gap itself.
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
        share = [max(1, weights[k]) if weights else 1 for k in range(i, run_end)]
        whole = sum(share)
        acc = 0
        for piece in share:
            start = left + width * (acc / whole)
            acc += piece
            out.append((start, left + width * (acc / whole)))
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
_NOTE_RE = re.compile(
    r"\[[^\]]*\]"
    r"|\((?=[^)]*\b(?:music|applause|laugh\w*|instrumental|silence|inaudible|singing|"
    r"humming|cheer\w*|noise|beat)\b)[^)]*\)"
    r"|[♪♫]+",
    re.IGNORECASE,
)
_BRACKET_RE = re.compile(r"[\[\]()]")


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
    model = _load_lyrics_model()
    try:
        segments, _ = model.transcribe(
            audio_path,
            word_timestamps=True,
            # Off, as in the aligner: captions and scenes are laid on the file's own
            # timeline, gaps and all. Silero's VAD is a speech detector too, and a voice
            # under a band is exactly where it misfires.
            vad_filter=False,
            language=language or None,
            # Unlike alignment this wants the best WORDING — nothing corrects it afterwards
            # — so the wider beam earns its time here.
            beam_size=5,
            # A song is the worst case for whisper's prompt cascade: choruses repeat, one
            # hallucinated line becomes the prompt for the next window, and the lyrics walk
            # off the recording for a verse. Each window is decoded cold.
            condition_on_previous_text=False,
        )
        pieces = _sung_pieces(segments)  # consuming the generator IS the decode
    except AlignmentUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 — an unreadable or undecodable file
        raise AlignmentFailed(
            f"The music track could not be read for transcription: {exc}"
        ) from exc
    result = _transcript(pieces, total_duration_sec)
    logger.info(
        "Lyrics transcribed: %d words, %d unsure", len(result["words"]), sum(result["guessed"])
    )
    return result


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

    if not words:
        raise AlignmentFailed(
            "No singing was found on the music track. Captions and the storyboard are made "
            "from the words that are sung, so they need a song with vocals — an "
            "instrumental has no words to work from."
        )
    # Every word is known here, so this is only the monotonic clean-up and the clamp to
    # the file's length — whisper's timestamps can still run backwards between windows.
    return {"text": text, "words": _fill_gaps(words, total), "guessed": guessed}
