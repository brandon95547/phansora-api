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

RESEARCH_PROMPT asks for a JSON array, so that is read first. It is asked for in the
PROMPT rather than enforced with `responseMimeType`, because the API refuses a forced
JSON mime type alongside the search tool — a grounded call cannot be put in JSON mode,
which is the whole reason a second model call existed. Asking works; a model that
fences its JSON, or writes a sentence before it, is handled here rather than being
failed for it.

The older `Title - Date` line format is still read when there is no array, because the
prompt is retuned often and a trace should not die because its output shape moved.

What this deliberately does NOT do is judge. Anything shaped like an item is kept.
Deciding what deserves to be on the timeline is the research call's job, and putting a
second opinion in the way is how a trace lost material it had already found.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)

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
    # Everything the research reported about the item besides its title and date, as
    # (label, value) in the order the prompt asks for them. Pairs rather than named
    # fields because the fields are the PROMPT's to choose, and it is retuned often —
    # named ones would drop anything it started asking for that this file had not
    # heard of. Empty when the answer carried nothing but a title and a date, which is
    # a fact about the answer rather than a gap to fill.
    details: Tuple[Tuple[str, str], ...] = ()


# The keys the prompt asks for, and the ones models substitute for them. Read leniently:
# a trace should not fail because the model wrote "name" where the prompt said "title".
# The one field that is a statement ABOUT the item rather than a property of it, so it
# reads as the node's claim instead of as another row in the table.
SIGNIFICANCE_LABEL = "Significance"

_TITLE_KEYS = ("title", "item_title", "item", "name")
_DATE_KEYS = ("date", "date_range", "dates", "year", "period")
# Order matters — this is the order the fields are shown in.
_DETAIL_KEYS = (
    ("origin", "Origin"),
    ("geographic_origin", "Origin"),
    ("provenance", "Provenance"),
    ("material", "Material"),
    ("medium", "Medium"),
    ("language", "Language"),
    ("authorship", "Authorship"),
    ("author", "Authorship"),
    ("source_community", "Source community"),
    ("significance", "Significance"),
    ("function", "Function"),
    ("description", "Description"),
    ("notes", "Notes"),
)


def parse_dated_list(text: str) -> List[DatedItem]:
    """Every item in ``text``, in the order the model gave them.

    A JSON array if there is one, falling back to ``Title - Date`` lines. Anything that
    is neither — a stray heading, a sentence added despite being told not to — is
    skipped rather than guessed at.
    """
    items = _parse_json_items(text)
    if items:
        return items
    return _parse_line_items(text)


def _parse_json_items(text: str) -> List[DatedItem]:
    """The array the prompt asks for, however it arrived.

    Fenced, prefaced with a sentence, or followed by one: all three happen, none of them
    means the research failed, and all three are cheaper to read past than to re-run.
    """
    raw = (text or "").strip()
    if not raw:
        return []

    start = raw.find("[")
    if start == -1:
        return []

    end = raw.rfind("]")
    try:
        if end <= start:
            # No closing bracket at all: the answer stopped before the array did.
            raise json.JSONDecodeError("unterminated array", raw, len(raw))
        # strict=False so a literal newline inside a value is read rather than
        # rejected. Models put line breaks in long prose fields, and json.loads
        # calls that an invalid control character — which would throw away an
        # otherwise perfect array over a line break in one significance field.
        parsed = json.loads(raw[start : end + 1], strict=False)
    except json.JSONDecodeError:
        # A truncated array is the expected failure now that each item carries six
        # fields: the answer runs into GEMINI_SEARCH_MAX_TOKENS and stops mid-object,
        # taking every complete item before it down with it. Thirty researched items
        # should not be lost to the thirty-first being cut in half.
        parsed = list(_salvage_objects(raw[start:]))
        if parsed:
            logger.warning(
                "The research answer was not valid JSON — recovered %d complete items "
                "from it. If this repeats, the answer is being cut off: raise "
                "GEMINI_SEARCH_MAX_TOKENS.", len(parsed),
            )
    if not isinstance(parsed, list):
        return []

    items: List[DatedItem] = []
    seen: set = set()
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        lowered = {str(k).strip().lower().replace(" ", "_"): v for k, v in entry.items()}
        title = _first_string(lowered, _TITLE_KEYS)
        if not title:
            continue
        key = title.casefold()
        if key in seen:
            continue
        seen.add(key)

        year, year_end, precision, era_label = parse_date(_first_string(lowered, _DATE_KEYS))
        items.append(
            DatedItem(
                title=title,
                year=year,
                year_end=year_end,
                precision=precision,
                era_label=era_label,
                details=_details(lowered),
            )
        )
    return items


def _salvage_objects(raw: str) -> Iterator[dict]:
    """Every complete ``{...}`` in ``raw``, ignoring anything left half-written.

    Counts braces rather than reaching for a regex, because a brace inside a string —
    which the significance field will contain sooner or later — is not structure.
    """
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(raw):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                try:
                    obj = json.loads(raw[start : i + 1], strict=False)
                except json.JSONDecodeError:
                    pass
                else:
                    if isinstance(obj, dict):
                        yield obj
                start = -1
            elif depth < 0:
                return


def _first_string(entry: dict, keys: tuple) -> str:
    for key in keys:
        value = entry.get(key)
        if isinstance(value, (str, int, float)) and str(value).strip():
            return str(value).strip()
    return ""


def _details(entry: dict) -> Tuple[Tuple[str, str], ...]:
    """The metadata fields as labelled pairs, in the order the prompt asks for them.

    Labelled and kept apart because "Mesopotamia" and "clay tablet" answer different
    questions, and a node that runs them together answers neither. Empty fields are
    dropped rather than carried as blanks: the prompt tells the model to leave a field
    empty instead of guessing, and honouring that means not showing the gap either.
    """
    out: List[Tuple[str, str]] = []
    used: set = set()
    for key, label in _DETAIL_KEYS:
        if label in used:
            continue
        value = _first_string(entry, (key,))
        if value:
            # A value spanning lines would break every consumer that shows these as
            # rows, and nothing in a field this size needs a line break.
            out.append((label, " ".join(value.split())))
            used.add(label)
    return tuple(out)


def _parse_line_items(text: str) -> List[DatedItem]:
    """Every ``Title - Date`` line in ``text``, in the order the model gave them."""
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
