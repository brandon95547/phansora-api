"""Where a sentence ends.

One answer, shared by the three places that need it, because each of them used
to have none:

  * the script writer, whose reply can be cut off by the model's output ceiling
    (deepseek_client.chat_to_end picks up from the last finished sentence);
  * the script itself, which must never be recorded ending on half a sentence
    (pipeline._phase_sessions applies ``complete_ending`` as the floor);
  * the chunker, whose boundaries are the only places a lesson can begin or end
    (chunking.build_chunks holds an unfinished sentence back for the next chunk).

Deliberately small. This is not a sentence tokenizer — it only has to find the
LAST place a text could stop, and to be conservative about it: calling a real
ending "not an ending" costs one sentence, calling "Dr." an ending leaves a
lesson finishing on a title.
"""
from __future__ import annotations

import re

# A terminator, plus whatever closes the quotation or bracket it sits inside.
_CLOSERS = "\"'”’)\\]»"
_END = re.compile(rf"[.!?…][{_CLOSERS}]*(?=\s|$)")
_ENDS_COMPLETE = re.compile(rf"[.!?…][{_CLOSERS}]*\s*$")

# Words whose period is part of the word. Only consulted mid-text: at the very
# end of a text there is nothing after the period for it to be abbreviating.
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "gen", "col", "lt",
    "sgt", "rev", "hon", "vs", "cf", "fig", "vol", "e.g", "i.e",
})
_WORD_BEFORE = re.compile(r"([A-Za-z]+(?:\.[A-Za-z]+)*)$")


def ends_complete(text: str) -> bool:
    """Does this text stop at the end of a sentence?"""
    return bool(_ENDS_COMPLETE.search(text or ""))


def last_sentence_end(text: str) -> int:
    """Index just past the last finished sentence, or 0 if there is none."""
    text = text or ""
    for match in reversed(list(_END.finditer(text))):
        if match.end() >= len(text.rstrip()) or not _is_abbreviation(text, match.start()):
            return match.end()
    return 0


def complete_ending(text: str) -> tuple[str, str]:
    """``(text ending on a finished sentence, the fragment dropped to get there)``.

    A text with no finished sentence at all is returned untouched: there is
    nothing better to fall back to, and an empty script is a worse failure than
    an unpunctuated one.
    """
    text = text or ""
    if ends_complete(text):
        return text, ""
    cut = last_sentence_end(text)
    if cut <= 0:
        return text, ""
    return text[:cut], text[cut:].strip()


def _is_abbreviation(text: str, period_at: int) -> bool:
    if text[period_at] != ".":
        return False
    word = _WORD_BEFORE.search(text[:period_at])
    if not word:
        return False
    token = word.group(1)
    # "J. R. R. Tolkien", "the U.S. Army": an initial is never the end of a sentence
    # while there is still text after it.
    return token.lower() in _ABBREVIATIONS or len(token.rsplit(".", 1)[-1]) == 1
