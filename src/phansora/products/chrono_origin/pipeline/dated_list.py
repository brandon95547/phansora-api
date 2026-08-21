"""Read the research call's answer straight onto the timeline.

RESEARCH_PROMPT asks for one thing and gets it: ``Title - Date``, one item per line,
oldest first, no prose. That IS the product — what came before what, in order — so it
is read here with code instead of being handed to a second model to be reformatted.

There used to be a second call doing exactly that. It existed because the Gemini API
refuses ``responseMimeType: application/json`` alongside the search tool, so a grounded
call cannot be forced into JSON and something else had to make it. But JSON was never
the requirement; STRUCTURE was, and ``Title - Date`` is structure. A parser cannot
invent an item, cannot quietly drop one, cannot decide the list would read better
shorter, costs nothing, and cannot overrun a token budget — which the synthesis call
did four times in three days, each time failing a trace that had already been researched.

What this deliberately does NOT do is judge. Anything shaped like an item is kept.
Deciding what deserves to be on the timeline is the research call's job, and putting a
second opinion in the way is how a trace lost material it had already found.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Split on a dash with spaces around it. Titles carry their own hyphens — Proto-Sinaitic,
# Al-Andalus, Sub-Saharan — and splitting on a bare "-" cuts those in half. The LAST
# separator wins, because a title may contain a spaced dash of its own but the date is
# always last.
_SEPARATORS = (" — ", " – ", " -- ", " - ")

_BCE = re.compile(r"\b(bce|bc)\b", re.I)
_CENTURY = re.compile(r"(\d{1,2})\s*(?:st|nd|rd|th)?\s*century", re.I)
_MILLENNIUM = re.compile(r"(\d{1,2})\s*(?:st|nd|rd|th)?\s*millennium", re.I)
_RANGE = re.compile(r"(\d{1,4})\s*(?:[-–—]|\bto\b)\s*(\d{1,4})")
_NUMBER = re.compile(r"\d{1,4}")
_PRESENT = re.compile(r"\b(present|today|ongoing|current|now)\b", re.I)
# Leading bullets and numbering. The prompt asks for neither, and models add them anyway.
_LEADING_MARK = re.compile(r"^\s*(?:[-*•·]|\d{1,3}[.)])\s+")
_BOLD = re.compile(r"\*\*|__")


@dataclass(frozen=True)
class DatedItem:
    """One line of the model's list, as a timeline node needs it."""

    title: str
    year: Optional[int] = None
    year_end: Optional[int] = None
    # One of models.DatePrecision. Describes how precisely the DATE is known, which is
    # not the same as how much we trust the item.
    precision: str = "unknown"
    # Kept verbatim when there is no year to be had ("Bronze Age", "present day"), so
    # the node can still say when it is rather than showing an empty slot.
    era_label: Optional[str] = None


def parse_dated_list(text: str) -> List[DatedItem]:
    """Every ``Title - Date`` line in ``text``, in the order the model gave them.

    Lines that are not items — a stray heading, a sentence the model added despite
    being told not to — have no separator and are skipped rather than guessed at.
    """
    items: List[DatedItem] = []
    seen: set = set()

    for raw_line in (text or "").splitlines():
        line = _BOLD.sub("", _LEADING_MARK.sub("", raw_line)).strip()
        if not line:
            continue

        split = _split_title_and_date(line)
        if split is None:
            continue
        title, date_text = split
        if not title:
            continue

        year, year_end, precision, era_label = parse_date(date_text)
        # A separator alone does not make a line an item: prose can contain " - " too.
        # Requiring something date-shaped on the right is what tells them apart, and a
        # short right-hand side ("present day") counts — that is an era, not prose.
        if year is None and era_label is None:
            continue

        key = title.casefold()
        if key in seen:
            continue
        seen.add(key)
        items.append(
            DatedItem(
                title=title,
                year=year,
                year_end=year_end,
                precision=precision,
                era_label=era_label,
            )
        )
    return items


def _split_title_and_date(line: str) -> Optional[Tuple[str, str]]:
    for sep in _SEPARATORS:
        idx = line.rfind(sep)
        if idx > 0:
            return line[:idx].strip(), line[idx + len(sep):].strip()
    return None


def parse_date(text: str) -> Tuple[Optional[int], Optional[int], str, Optional[str]]:
    """``"c. 3400 BCE"`` -> ``(-3400, None, "year", None)``.

    Returns (year, year_end, precision, era_label). BCE years are negative, which is
    what the whole product sorts on. An era_label is set only when there is no year to
    return, so a node never shows both a date and a description of the same date.
    """
    raw = (text or "").strip().strip(".,;:")
    if not raw:
        return None, None, "unknown", None

    bce = bool(_BCE.search(raw))
    sign = -1 if bce else 1

    m = _MILLENNIUM.search(raw)
    if m:
        n = int(m.group(1))
        # 3rd millennium BCE runs 3000-2001 BCE; 3rd millennium CE starts in 2001.
        start = -(n * 1000) if bce else (n - 1) * 1000 + 1
        end = -((n - 1) * 1000 + 1) if bce else n * 1000
        return start, end, "millennium", None

    m = _CENTURY.search(raw)
    if m:
        n = int(m.group(1))
        start = -(n * 100) if bce else (n - 1) * 100 + 1
        end = -((n - 1) * 100 + 1) if bce else n * 100
        return start, end, "century", None

    m = _RANGE.search(raw)
    if m:
        first, second = int(m.group(1)) * sign, int(m.group(2)) * sign
        # A BCE range counts DOWN (1400-1200 BCE), so ordering it by value keeps the
        # earlier end first without the caller having to know which era it is in.
        start, end = (first, second) if first <= second else (second, first)
        return start, end, "year", None

    m = _NUMBER.search(raw)
    if m:
        year = int(m.group(0)) * sign
        # "c. 1750 BCE" is still a year, and precision stays "year". Circa describes
        # confidence IN the year; precision describes the granularity OF it. There is no
        # value for "roughly this year", and inventing a looser one would report the
        # date as vaguer than the answer said it was.
        return year, None, "year", None

    # No number anywhere. If it names a time at all, keep the words: "Bronze Age" and
    # "present day" are real positions on a timeline even without a year to sort by.
    if _PRESENT.search(raw) or len(raw) <= 60:
        return None, None, "era", raw
    return None, None, "unknown", None
