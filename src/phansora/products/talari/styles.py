"""Illustration styles: the prompt language behind each style the app offers.

The ids are the style picker's. Every interior illustration in a book
is drawn with the same style text, which is what keeps the book's art
consistent from page to page.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Style:
    positive: str
    negative: str = ""


STYLES: dict[str, Style] = {
    "watercolor": Style(
        "watercolor illustration, soft translucent washes, visible paper grain, loose wet edges, luminous colour",
        "photograph, 3d render, harsh outlines",
    ),
    "pencil": Style(
        "graphite pencil drawing, fine cross-hatching, delicate sketch lines, monochrome on white paper",
        "colour, photograph, 3d render, painting",
    ),
    "ink": Style(
        "pen and ink illustration, confident black line work, cross-hatching, high contrast, white paper",
        "colour, photograph, 3d render, soft shading",
    ),
    "engraving": Style(
        "antique copperplate engraving, dense fine parallel lines, 19th century book illustration, aged paper tone",
        "colour, photograph, 3d render, modern",
    ),
    "classical": Style(
        "classical oil painting, chiaroscuro lighting, rich glazes, old master composition, visible brushwork",
        "photograph, 3d render, cartoon, flat colour",
    ),
    "fantasy": Style(
        "epic fantasy illustration, dramatic lighting, ornate detail, painterly atmosphere, rich colour",
        "photograph, cartoon, flat colour",
    ),
    "realistic": Style(
        "photorealistic, natural light, true-to-life texture and depth, sharp detail",
        "painting, drawing, cartoon, illustration",
    ),
}

DEFAULT_STYLE = "watercolor"

BASE_NEGATIVE = "text, lettering, words, watermark, signature, logo, frame, border, blurry, low quality, deformed"

COVER_POSITIVE = "book cover art, strong central composition, calm uncluttered area near the top"

# A picture on a dark page should sit in it, not glare out of it as a pale box.
DARK_PAGE_POSITIVE = "set against a deep black background, low-key lighting, rich shadows"


def build_prompts(prompt: str, style_id: str, purpose: str, theme: str = "light") -> tuple[str, str]:
    """The positive and negative prompt for one picture."""
    style = STYLES.get(style_id, STYLES[DEFAULT_STYLE])
    subject = " ".join(prompt.split()).rstrip(".,; ")
    parts = [subject, style.positive]
    if theme == "dark":
        parts.append(DARK_PAGE_POSITIVE)
    if purpose == "cover":
        parts.append(COVER_POSITIVE)
    negative = ", ".join(part for part in (BASE_NEGATIVE, style.negative) if part)
    return ", ".join(part for part in parts if part), negative
