"""Between the model's text and the book's HTML.

The model writes light markdown, which it is good at:

    ## A section heading          (### for a smaller one)
    - A bulleted entry            (1. for a numbered one)
    > A quotation
    A paragraph, with *emphasis* and **strong** words.

The book's text is blocks of INLINE-ONLY HTML: escaped text plus <em> and <strong>, so
the app can put it into a page as it is. Everything the model wrote is escaped except
the two tags made here; a <script> in a reply arrives on the page as the text
"<script>". Models given HTML tend to answer in HTML, so block tags (<p>, <h2>, <li>…)
in a reply are read as the structure they stand for, and inline <em>/<i>/<strong>/<b>
are kept as emphasis.

The other direction, the app's HTML as text for the model to read, is ``as_markdown``
and ``as_text``.
"""

from __future__ import annotations

import html
import re
from typing import Any

FENCE = re.compile(r"^```")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
NUMBERED = re.compile(r"^\d{1,3}[.)]\s+(.*)$")
BULLET = re.compile(r"^[-*•+]\s+(.*)$")
QUOTE = re.compile(r"^>\s?(.*)$")
RULE = re.compile(r"^(?:(?:\*\s*){3,}|(?:-\s*){3,}|(?:_\s*){3,})$")
# A line that is nothing but bold, the way a model often marks a heading it forgot to mark.
BOLD_LINE = re.compile(r"^\*\*([^*]{1,80})\*\*:?$")
SCENE_BREAK = "* * *"

# --- inline ------------------------------------------------------------------

# Stand-ins for the tags this module makes, so escaping cannot touch them. Taken from
# Unicode's private use area, and removed from the input first so a reply cannot forge one.
EM_OPEN, EM_CLOSE, STRONG_OPEN, STRONG_CLOSE = "", "", "", ""
PRIVATE = re.compile("[-]")
OPENS = {EM_OPEN: "em", STRONG_OPEN: "strong"}
CLOSES = {EM_CLOSE: "em", STRONG_CLOSE: "strong"}

EMPHASIS_TAG = re.compile(r"<(/?)(em|i|strong|b)\s*>", re.IGNORECASE)
# Formatting with nothing to say on a book page; the tags go, their text stays.
HARMLESS_TAG = re.compile(r"</?(?:span|u|s|sub|sup|a|font|mark|small|cite|abbr|ins|del)\b[^>]*>", re.IGNORECASE)
MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
MD_CODE = re.compile(r"`([^`]+)`")
STRONG_EM_MD = re.compile(r"\*\*\*(?=\S)(.+?)(?<=\S)\*\*\*")
STRONG_MD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*|(?<!\w)__(?=\S)(.+?)(?<=\S)__(?!\w)")
EM_MD = re.compile(r"(?<![\w*])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![\w*])|(?<![\w_])_(?=[^\s_])(.+?)(?<=[^\s_])_(?![\w_])")
EMPTY_PAIR = re.compile(r"<(em|strong)>\s*</\1>")


def _tag(match: re.Match[str]) -> str:
    closing, name = match.group(1) == "/", match.group(2).lower()
    strong = name in ("strong", "b")
    if closing:
        return STRONG_CLOSE if strong else EM_CLOSE
    return STRONG_OPEN if strong else EM_OPEN


def _balance(text: str) -> str:
    """Turn the stand-ins into tags, closing what is left open and dropping stray closes."""
    out: list[str] = []
    stack: list[str] = []
    for ch in text:
        if ch in OPENS:
            stack.append(OPENS[ch])
            out.append(f"<{OPENS[ch]}>")
        elif ch in CLOSES:
            name = CLOSES[ch]
            if name not in stack:
                continue
            reopen: list[str] = []
            while stack[-1] != name:
                inner = stack.pop()
                out.append(f"</{inner}>")
                reopen.append(inner)
            stack.pop()
            out.append(f"</{name}>")
            for inner in reversed(reopen):
                stack.append(inner)
                out.append(f"<{inner}>")
        else:
            out.append(ch)
    while stack:
        out.append(f"</{stack.pop()}>")
    result = "".join(out)
    while EMPTY_PAIR.search(result):
        result = EMPTY_PAIR.sub("", result)
    return result.strip()


def inline_html(text: str) -> str:
    """One line of model text as inline-only HTML: escaped, with <em> and <strong>."""
    text = PRIVATE.sub("", html.unescape(text or ""))
    text = MD_IMAGE.sub("", text)
    text = MD_LINK.sub(r"\1", text)
    text = MD_CODE.sub(r"\1", text)
    text = HARMLESS_TAG.sub("", text)
    text = EMPHASIS_TAG.sub(_tag, text)
    text = html.escape(" ".join(text.split()), quote=False)
    text = STRONG_EM_MD.sub(lambda m: f"{STRONG_OPEN}{EM_OPEN}{m.group(1)}{EM_CLOSE}{STRONG_CLOSE}", text)
    text = STRONG_MD.sub(lambda m: f"{STRONG_OPEN}{m.group(1) or m.group(2)}{STRONG_CLOSE}", text)
    text = EM_MD.sub(lambda m: f"{EM_OPEN}{m.group(1) or m.group(2)}{EM_CLOSE}", text)
    # A bold marker the model opened and never closed would show as two asterisks.
    text = text.replace("**", "")
    return _balance(text)


def plain(fragment: str) -> str:
    """Inline HTML (or model text) as plain text, for comparing and quoting."""
    text = re.sub(r"</?(?:em|strong)>", "", inline_html(fragment))
    return html.unescape(text)


# --- reading HTML --------------------------------------------------------------

HTML_STRUCTURE = re.compile(r"<(?:p|div|h[1-6]|li|ul|ol|blockquote|br)\b[^>]*>", re.IGNORECASE)
HEADING_TAG = re.compile(r"<h([1-6])\b[^>]*>(.*?)</h\1\s*>", re.IGNORECASE | re.DOTALL)
BLOCKQUOTE_TAG = re.compile(r"<blockquote\b[^>]*>(.*?)</blockquote\s*>", re.IGNORECASE | re.DOTALL)
LIST_TAG = re.compile(r"<(ul|ol)\b[^>]*>(.*?)</\1\s*>", re.IGNORECASE | re.DOTALL)
LIST_ITEM_OPEN = re.compile(r"<li\b[^>]*>", re.IGNORECASE)
LIST_ITEM_CLOSE = re.compile(r"</li\s*>", re.IGNORECASE)
BREAK_TAG = re.compile(r"<br\s*/?>", re.IGNORECASE)
BLOCK_TAG = re.compile(r"</?(?:p|div|section|article|header|footer)\b[^>]*>", re.IGNORECASE)
SCRIPT_TAG = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
# A tag, not a "<" in a sentence ("a < b"): a letter must follow it.
ANY_TAG = re.compile(r"</?[A-Za-z][\w:-]*(?:\s[^<>]*)?/?>")


def _structure_as_markdown(text: str) -> str:
    """Block-level HTML as the markdown it stands for. Inline tags are left alone."""

    def heading(match: re.Match[str]) -> str:
        marks = "##" if int(match.group(1)) <= 2 else "###"
        return f"\n\n{marks} {' '.join(match.group(2).split())}\n\n"

    def quote(match: re.Match[str]) -> str:
        inner = _structure_as_markdown(match.group(1)).strip()
        lines = [f"> {line}" if line.strip() else ">" for line in inner.split("\n")]
        return "\n\n" + "\n".join(lines) + "\n\n"

    def listing(match: re.Match[str]) -> str:
        marker = "1. " if match.group(1).lower() == "ol" else "- "
        inner = LIST_ITEM_CLOSE.sub("\n", LIST_ITEM_OPEN.sub("\n" + marker, match.group(2)))
        inner = BLOCK_TAG.sub(" ", BREAK_TAG.sub(" ", inner))
        return "\n\n" + "\n".join(line.strip() for line in inner.split("\n") if line.strip()) + "\n\n"

    text = HEADING_TAG.sub(heading, text)
    text = BLOCKQUOTE_TAG.sub(quote, text)
    text = LIST_TAG.sub(listing, text)
    text = BREAK_TAG.sub("\n", text)
    text = BLOCK_TAG.sub("\n\n", text)
    return text


def as_markdown(fragment: str) -> str:
    """The app's HTML (or plain text) as the light markdown the model reads.

    Input from the app, so anything that is not text or structure is simply dropped.
    """
    text = fragment or ""
    if "<" in text:
        text = SCRIPT_TAG.sub("", text)
        text = _structure_as_markdown(text)
        text = re.sub(r"<(?:strong|b)\b[^>]*>|</(?:strong|b)\s*>", "**", text, flags=re.IGNORECASE)
        text = re.sub(r"<(?:em|i)\b[^>]*>|</(?:em|i)\s*>", "*", text, flags=re.IGNORECASE)
        text = ANY_TAG.sub("", text)
    text = html.unescape(text).replace("\r\n", "\n").replace("\xa0", " ")
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def as_text(fragment: str) -> str:
    """The app's HTML as plain text: paragraphs separated by blank lines, no markup."""
    paragraphs = []
    for block in parse_blocks(as_markdown(fragment)):
        if block["type"] == "list":
            paragraphs.extend(plain(item) for item in block["items"])
        else:
            paragraphs.append(plain(block["html"]))
    return "\n\n".join(p for p in paragraphs if p)


# --- blocks ------------------------------------------------------------------


def parse_blocks(text: str) -> list[dict[str, Any]]:
    """Model markdown (or HTML) as the book's blocks.

    Each block is a dict in the shape of ``schemas.Block``: heading (level 2 or 3),
    paragraph, quote (html), or list (items, ordered).
    """
    text = (text or "").replace("\r\n", "\n")
    if HTML_STRUCTURE.search(text):
        text = _structure_as_markdown(text)

    blocks: list[dict[str, Any]] = []
    paragraph: list[str] = []
    quote: list[str] = []
    items: list[str] = []
    state = {"ordered": False, "gap": False}

    def flush_paragraph() -> None:
        if paragraph:
            joined = " ".join(paragraph)
            bold = BOLD_LINE.match(joined)
            if bold and len(paragraph) == 1 and len(bold.group(1).split()) <= 12:
                blocks.append({"type": "heading", "level": 3, "html": inline_html(bold.group(1))})
            else:
                rendered = inline_html(joined)
                if rendered:
                    blocks.append({"type": "paragraph", "html": rendered})
            paragraph.clear()

    def flush_quote() -> None:
        if quote:
            rendered = inline_html(" ".join(quote))
            if rendered:
                blocks.append({"type": "quote", "html": rendered})
            quote.clear()

    def flush_list() -> None:
        if items:
            rendered = [item for item in (inline_html(entry) for entry in items) if item]
            if rendered:
                block: dict[str, Any] = {"type": "list", "items": rendered}
                if state["ordered"]:
                    block["ordered"] = True
                blocks.append(block)
            items.clear()

    def flush() -> None:
        flush_paragraph()
        flush_quote()
        flush_list()

    for raw in text.split("\n"):
        line = raw.strip()
        if FENCE.match(line):
            continue
        if not line:
            flush_paragraph()
            flush_quote()
            state["gap"] = True
            continue
        if RULE.match(line):
            flush()
            if blocks and blocks[-1].get("html") != SCENE_BREAK:
                blocks.append({"type": "paragraph", "html": SCENE_BREAK})
            continue
        heading = HEADING.match(line)
        if heading:
            flush()
            words = inline_html(heading.group(2))
            if words:
                blocks.append({"type": "heading", "level": 2 if len(heading.group(1)) <= 2 else 3, "html": words})
            continue
        quoted = QUOTE.match(line)
        if quoted:
            flush_paragraph()
            flush_list()
            if quoted.group(1).strip():
                quote.append(quoted.group(1).strip())
            else:
                flush_quote()
            continue
        numbered, bullet = NUMBERED.match(line), BULLET.match(line)
        if numbered or bullet:
            flush_paragraph()
            flush_quote()
            ordered = bool(numbered)
            if items and ordered != state["ordered"]:
                flush_list()
            items.append((numbered or bullet).group(1))
            state["ordered"], state["gap"] = ordered, False
            continue
        if items and not state["gap"]:
            # A line straight after an entry, with no blank between, continues it.
            items[-1] += " " + line
            continue
        flush_quote()
        flush_list()
        paragraph.append(line)
        state["gap"] = False
    flush()

    # A scene break with nothing before or after it is not a break between anything.
    while blocks and blocks[0].get("html") == SCENE_BREAK:
        blocks.pop(0)
    while blocks and blocks[-1].get("html") == SCENE_BREAK:
        blocks.pop()
    return blocks


def blocks_as_html(blocks: list[dict[str, Any]], *, wrap: bool) -> str:
    """Blocks as a passage of inline HTML. More than one paragraph (or ``wrap``) puts each
    in <p>…</p>; headings become bold paragraphs and list entries paragraphs of their own,
    since a passage carries no other tags."""
    paragraphs: list[str] = []
    for block in blocks:
        if block["type"] == "heading":
            paragraphs.append(f"<strong>{block['html']}</strong>")
        elif block["type"] == "list":
            for number, item in enumerate(block["items"], start=1):
                paragraphs.append(f"{number}. {item}" if block.get("ordered") else f"• {item}")
        else:
            paragraphs.append(block["html"])
    if len(paragraphs) == 1 and not wrap:
        return paragraphs[0]
    return "".join(f"<p>{p}</p>" for p in paragraphs)
