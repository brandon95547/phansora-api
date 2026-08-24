"""Spoken-form text normalization, run once before any chunking.

Two problems, one fix:

1. **Pronunciation.** The engine stumbles on dotted abbreviations — "D.C.", "Dr.",
   "a.m.", "Aug." — because it is reading punctuation that is not there in speech.

2. **Chunking (the more damaging one).** Every chunker in the pipeline splits sentences
   on ``(?<=[.!?])\\s+`` — ``shared/utils/chunking.chunk_text`` at the document level and
   ``cosyvoice3_client._chunk_text`` at the 200-char engine level. Neither knows what an
   abbreviation is, so "The march on Washington, D.C. drew thousands." splits after
   "D.C.". Chunks are synthesized independently and concatenated with ffmpeg, so whenever
   such a split lands on a chunk boundary the result is an audible seam mid-sentence.

This is deterministic, offline and idempotent: pure regex, no model call, so it adds no
per-request cost and cannot fail or introduce nondeterminism into an audio render.

**Sentence-terminal periods are preserved.** Dropping every abbreviation period would
destroy real sentence boundaries — "…moved to D.C. The next year…" would run together and
lose its prosody. So abbreviations that *can* end a sentence (initialisms, "etc.", "p.m.")
keep their period when what follows looks like a new sentence, and lose it otherwise:

    "Washington, D.C. drew thousands"  ->  "Washington, D C drew thousands"   (period dropped)
    "moved to D.C. The next year"      ->  "moved to D C. The next year"      (period kept)

The ambiguous case ("the D.C. Metro") keeps its period, which is exactly today's
behavior — this never makes an existing split worse, it only removes false ones.

Tuning: every lexicon below is inserted verbatim. If the engine mispronounces an entry,
respell it phonetically right here ("D C" -> "Dee See"). That is the entire tuning
surface — no code change required.
"""

from __future__ import annotations

import re

__all__ = ["normalize_for_tts"]


# ── Lexicons ─────────────────────────────────────────────────────────────────

# Never sentence-final in practice: a title introduces a name. Their period is always
# safe to drop, and MUST be dropped without a replacement — "Dr. King" is followed by a
# capital, so the terminal heuristic would wrongly read it as a full stop.
_TITLES: dict[str, str] = {
    "mr.": "Mister", "mrs.": "Missus", "ms.": "Miz", "dr.": "Doctor",
    "prof.": "Professor", "rev.": "Reverend", "hon.": "Honorable",
    "pres.": "President", "gov.": "Governor", "sen.": "Senator",
    "rep.": "Representative", "gen.": "General", "col.": "Colonel",
    "capt.": "Captain", "lt.": "Lieutenant", "sgt.": "Sergeant", "maj.": "Major",
    "jr.": "Junior", "sr.": "Senior",
}

# Also never sentence-final: a month abbreviation introduces a day number.
# May/June/July are never abbreviated with a period, so they are absent by design.
_MONTHS: dict[str, str] = {
    "jan.": "January", "feb.": "February", "mar.": "March", "apr.": "April",
    "jun.": "June", "jul.": "July", "aug.": "August",
    "sept.": "September", "sep.": "September",
    "oct.": "October", "nov.": "November", "dec.": "December",
}

# Introducers: like titles, these always precede something, so they are never a sentence
# end. Kept apart from _PHRASES_TERMINAL because "e.g. Fuji" and "vs. Frazier" are
# followed by a capital, and the terminal heuristic would otherwise read that as a full
# stop and split the phrase off from what it introduces.
_PHRASES_INTRO: dict[str, str] = {
    "e.g.": "for example", "i.e.": "that is",
    "vs.": "versus", "approx.": "approximately",
}

# These genuinely CAN end a sentence ("…pears, etc. Then we left.", "…at 3 p.m. We left."),
# so the terminal period is preserved when what follows looks like a new sentence.
_PHRASES_TERMINAL: dict[str, str] = {
    "etc.": "et cetera", "a.m.": "AM", "p.m.": "PM",
}

# Dotted initialisms that read better expanded than spelled out letter by letter.
# Anything not listed falls back to spaced letters (see _sub_initialisms).
_INITIALISMS: dict[str, str] = {
    "u.s.": "United States", "u.s.a.": "United States",
    "u.k.": "United Kingdom", "e.u.": "European Union",
    "u.n.": "United Nations",
}


def _alt(keys) -> str:
    """Longest-first alternation, so "u.s.a." wins over "u.s." and "sept." over "sep."."""
    return "|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True))


_TITLE_RE = re.compile(rf"\b(?:{_alt(_TITLES)})", re.IGNORECASE)
_MONTH_RE = re.compile(rf"\b(?:{_alt(_MONTHS)})", re.IGNORECASE)
_PHRASE_INTRO_RE = re.compile(rf"\b(?:{_alt(_PHRASES_INTRO)})", re.IGNORECASE)
_PHRASE_TERMINAL_RE = re.compile(rf"\b(?:{_alt(_PHRASES_TERMINAL)})", re.IGNORECASE)

# Two or more single letters each followed by a period: D.C., U.S., F.B.I., U.S.A.
_INITIALISM_RE = re.compile(r"\b(?:[A-Za-z]\.){2,}")

# "John F. Kennedy" — a lone capital before a Titlecase word is a middle initial, not a
# full stop. Requiring [A-Z][a-z] after it leaves outline numerals ("I. THE END") and
# genuine sentence ends alone. Multi-letter words cannot match: the \b anchors to a
# single letter, so the "I." inside "FBI." is not a candidate.
#
# The (?<!\.) guard keeps this out of dotted initialisms. Without it "D.C. The next year"
# matches at the trailing "C.", stripping the very sentence period _sub_initialisms is
# about to preserve — which is why this must also run BEFORE that pass, while the
# initialism still has its interior dots to be recognized by.
#
# (?<![A-Z] ) is what makes the whole module idempotent: after one pass "D.C." has become
# "D C.", and a lone capital preceded by another lone capital is the tail of an already
# spelled initialism, not a middle initial. Without this a second pass would strip that
# period — and this runs at two hooks (document level and engine level), so second passes
# are the normal case, not an edge case.
_INITIAL_RE = re.compile(r"(?<!\.)(?<![A-Z] )\b([A-Z])\.(?=\s+[A-Z][a-z])")

# "St." is genuinely ambiguous. A Titlecase word after it means a name ("St. Louis");
# anything else is a street ("Main St."). Only wrong for the rare "…on Main St. Then…".
_SAINT_RE = re.compile(r"\bSt\.(?=\s+[A-Z][a-z])")
_STREET_RE = re.compile(r"\bSt\.")

# "No. 5" is a number; a bare "No." is the word "no" and is left alone.
_NUMBER_RE = re.compile(r"\bNo\.(?=\s*\d)")

# Symbols only where they are unambiguous. "%" and "°" must follow a digit, so a stray
# percent sign in prose is untouched. "$" is deliberately absent: "$1.5 million" needs
# real number parsing to say "one point five million dollars", and a half-fix that says
# "dollars one point five million" is worse than leaving it.
_SYMBOLS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\s*&\s*"), " and "),
    (re.compile(r"(?<=\d)\s*%"), " percent"),
    (re.compile(r"(?<=\d)\s*°"), " degrees"),
)


# Typography that is written to be SEEN, arriving in text that is only ever heard.
#
# An em dash is the clearest case and the one that prompted this: the engine has no
# reading for it, so it either voices something or swallows the break the dash was
# there to make. It becomes a COMMA rather than nothing, because that break is the
# whole point — "the report — which nobody read — was filed" with the dashes deleted
# runs the words together. A comma is also the chunk-safe choice: a period would
# invent a sentence boundary that ``_chunk_text`` then splits on, changing how the
# audio is batched.
#
# Hyphens are deliberately NOT in the class. "twenty-one" and "co-operate" are words.
#
# The markdown rules are here because narration in this app is model-written. Nobody
# types "**Chapter one**" into a script box, but a model hands one back, and the
# stars are read aloud or mangle the word they wrap.
_DASHES = "\u2014\u2013\u2015"          # em, en, horizontal bar

# Line-leading punctuation is a LIST MARKER, not a pause — dropped outright, since a
# line that starts with a comma reads as a stumble.
_LEADING_MARK_RE = re.compile(rf"(?m)^[ \t]*(?:[{_DASHES}]+|[\u2022\u00b7\u2023\u25aa]+|#{{1,6}}[ \t]+)[ \t]*")
# Between numbers a dash is a RANGE, not a pause. "1914-1918" is read "nineteen
# fourteen to nineteen eighteen"; as a comma it becomes a list of two years, which is
# a different fact. Must run before the general rule below, which would eat it.
_RANGE_RE = re.compile(rf"(?<=\d)[ \t]*[{_DASHES}][ \t]*(?=\d)")
# Anywhere else a dash is a break, and becomes one.
_DASH_RE = re.compile(rf"[ \t]*[{_DASHES}]+[ \t]*")
# Emphasis stars, kept only when they actually wrap something. The inner lookarounds mean
# "2 * 3 * 4" is arithmetic and survives; the OUTER ones extend that to unspaced
# arithmetic, which the inner pair alone got wrong — "a*b*c" matched with "b" as the
# emphasised run and came out "abc". A star preceded by a word character is not opening
# anything. These boundaries are deliberately the same ones the orphan rules below use.
_MD_EMPHASIS_RE = re.compile(
    r"""(?<![^\s([{"'“‘])\*{1,3}(?=\S)(.+?)(?<=\S)\*{1,3}(?![^\s)\]}.,;:!?"'”’])""", re.S)
# UNPAIRED stars, which the rule above cannot see. Models mix their delimiters —
# `the *prima materia" the material` opens with a star and closes with a quote — and the
# survivor is worse than a whole pair, because it fuses to the word: CosyVoice read
# `*prima` as one mangled token rather than saying "star".
#
# Pairing is unknowable here, so go by POSITION instead: a star that hugs a word on one
# side and has whitespace or punctuation on the other is a delimiter, whichever partner it
# lost. A star with non-space on BOTH sides is arithmetic ("2*3") and is left alone, which
# is the same distinction the paired rule draws — just applied to one side at a time.
_MD_OPEN_STAR_RE = re.compile(r"""(?<![^\s([{"'“‘])\*{1,3}(?=[^\s*])""")
_MD_CLOSE_STAR_RE = re.compile(r"""(?<=[^\s*])\*{1,3}(?![^\s)\]}.,;:!?"'”’])""")
# A line-leading "* " is a bullet, not emphasis — the star has a space after it, so neither
# rule above touches it. Dropped like the other list markers.
_MD_BULLET_RE = re.compile(r"(?m)^[ \t]*\*[ \t]+")
# Tidy-up after the above: a dash next to punctuation that was already there leaves
# ", ," or " ,". Runs last so it catches whatever the other rules produced.
_COMMA_RUN_RE = re.compile(r"(?:,[ \t]*){2,}")
_SPACED_COMMA_RE = re.compile(r"[ \t]+,")


def _strip_typography(text: str) -> str:
    """Remove marks that exist for the eye, keeping the pauses they stood for."""
    out = _LEADING_MARK_RE.sub("", text)
    out = _MD_BULLET_RE.sub("", out)
    # Paired first, so "**bold**" is consumed as a unit; only genuine orphans reach the
    # two rules below.
    out = _MD_EMPHASIS_RE.sub(r"\1", out)
    out = _MD_OPEN_STAR_RE.sub("", out)
    out = _MD_CLOSE_STAR_RE.sub("", out)
    out = _RANGE_RE.sub(" to ", out)
    out = _DASH_RE.sub(", ", out)
    out = _COMMA_RUN_RE.sub(", ", out)
    out = _SPACED_COMMA_RE.sub(",", out)
    return out


def _looks_terminal(text: str, end: int) -> bool:
    """True when the period at ``end`` plausibly ends a sentence rather than an abbreviation.

    End of text counts. Otherwise the next non-space character must look like the start of
    a new sentence — a capital or an opening quote.
    """
    rest = text[end:]
    if not rest.strip():
        return True
    m = re.match(r"\s+(.)", rest)
    return bool(m and (m.group(1).isupper() or m.group(1) in "\"'“‘"))


def _sub_map(
    text: str,
    pattern: re.Pattern[str],
    table: dict[str, str],
    *,
    may_end_sentence: bool,
) -> str:
    """Replace each lexicon hit with its spoken form.

    ``may_end_sentence`` decides whether a terminal period is restored. It is False for
    titles and months, which always introduce something and so are never sentence ends.
    """

    def repl(m: re.Match[str]) -> str:
        spoken = table[m.group(0).lower()]
        if may_end_sentence and _looks_terminal(text, m.end()):
            return f"{spoken}."
        return spoken

    return pattern.sub(repl, text)


def _sub_initialisms(text: str) -> str:
    def repl(m: re.Match[str]) -> str:
        raw = m.group(0)
        spoken = _INITIALISMS.get(raw.lower())
        if spoken is None:
            # Space the letters so they are read as letters. Collapsing to "US" instead
            # would invite the engine to read the word "us"; "U S" cannot be mistaken.
            spoken = " ".join(raw.replace(".", "").upper())
        if _looks_terminal(text, m.end()):
            return f"{spoken}."
        return spoken

    return _INITIALISM_RE.sub(repl, text)


def normalize_for_tts(text: str) -> str:
    """Rewrite ``text`` into spoken form. Safe to call more than once."""
    if not text or not text.strip():
        return text

    out = text
    # Phrases first: "e.g." and "a.m." also match the generic initialism pattern, and the
    # specific reading is the one we want.
    out = _sub_map(out, _PHRASE_INTRO_RE, _PHRASES_INTRO, may_end_sentence=False)
    out = _sub_map(out, _PHRASE_TERMINAL_RE, _PHRASES_TERMINAL, may_end_sentence=True)
    out = _sub_map(out, _TITLE_RE, _TITLES, may_end_sentence=False)
    out = _sub_map(out, _MONTH_RE, _MONTHS, may_end_sentence=False)
    out = _SAINT_RE.sub("Saint", out)
    out = _sub_map(out, _STREET_RE, {"st.": "Street"}, may_end_sentence=True)
    out = _NUMBER_RE.sub("number", out)
    # Middle initials before initialisms — see _INITIAL_RE.
    out = _INITIAL_RE.sub(r"\1", out)
    out = _sub_initialisms(out)
    # After the abbreviation passes, which read sentence terminals and must not be shown
    # commas this pass invented.
    out = _strip_typography(out)

    for pattern, replacement in _SYMBOLS:
        out = pattern.sub(replacement, out)

    # Collapse runs of spaces/tabs only. Newlines MUST survive: both chunkers split on
    # blank lines and single newlines, and _chunk_text relies on that to chunk verse
    # (line breaks, no terminal punctuation) correctly.
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"[ \t]+\n", "\n", out)
    return out.strip()
