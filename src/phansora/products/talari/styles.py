"""Illustration styles: the prompt language behind each style the app offers.

The ids are the style picker's. Every interior illustration in a book
is drawn with the same style text, which is what keeps the book's art
consistent from page to page.

"custom" is the author's own description of a style, sent with each picture. The palette
and level of detail are choices on top of any style; their defaults (colour, balanced)
add nothing, so a picture asked for without them is drawn exactly as before.
"""

from __future__ import annotations

import re
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
    "childrens": Style(
        "children's picture book illustration, warm friendly characters, soft rounded shapes, gentle colour, playful and inviting",
        "photograph, 3d render, frightening, grotesque, gritty",
    ),
    "lineart": Style(
        "clean line art, crisp even outlines, flat open shapes, no shading, white background",
        "photograph, 3d render, painterly texture, heavy shading, gradients",
    ),
    "comic": Style(
        "graphic novel illustration, bold inked outlines, dynamic composition, cel shading, cinematic framing",
        "photograph, 3d render, speech bubbles, captions, panel borders",
    ),
    "vintage": Style(
        "mid-century print illustration, limited palette, screen-printed texture, bold simplified shapes, 1950s editorial art",
        "photograph, 3d render, glossy, modern digital look",
    ),
    "minimal": Style(
        "flat vector illustration, simple geometric shapes, clean solid fills, generous negative space",
        "photograph, 3d render, texture, gradients, clutter",
    ),
    "digital": Style(
        "digital painting, painterly brushwork, rich atmospheric lighting, detailed rendering",
        "photograph, flat colour, line art",
    ),
}

DEFAULT_STYLE = "watercolor"
CUSTOM_STYLE = "custom"
CUSTOM_STYLE_MAX = 400

BASE_NEGATIVE = "text, lettering, words, watermark, signature, logo, frame, border, blurry, low quality, deformed"

COVER_POSITIVE = "book cover art, strong central composition, calm uncluttered area near the top"

# A picture on a dark page should sit in it, not glare out of it as a pale box.
DARK_PAGE_POSITIVE = "set against a deep black background, low-key lighting, rich shadows"

# Said in the positive prompt, not only the negative: FLUX, the pipeline the stack runs,
# draws at cfg 1 with an empty negative, so a negative alone would change nothing there.
GRAYSCALE_POSITIVE = "black and white, grayscale, monochrome"
GRAYSCALE_NEGATIVE = "colour, color"

DETAIL_POSITIVE = {
    "simple": "simple composition, few elements, clean uncluttered shapes",
    "balanced": "",
    "rich": "richly detailed, intricate textures, layered background detail",
}


def _clean(text: str) -> str:
    return " ".join((text or "").split()).rstrip(".,; ")


def custom_style_text(custom_style: str | None) -> str:
    """The author's own style description as prompt text: one line, bounded, no brackets
    or quotes (quoted words are what an image model tends to letter onto the picture)."""
    text = re.sub(r"[\"“”<>{}\[\]]", " ", custom_style or "")
    return re.sub(r"\s+([,;.])", r"\1", _clean(text))[:CUSTOM_STYLE_MAX]


def style_text(style_id: str, custom_style: str | None = None) -> str:
    """The style's prompt language: what a picture in this style looks like."""
    if style_id == CUSTOM_STYLE:
        custom = custom_style_text(custom_style)
        if custom:
            return custom
    return STYLES.get(style_id, STYLES[DEFAULT_STYLE]).positive


def build_prompts(
    prompt: str,
    style_id: str,
    purpose: str,
    theme: str = "light",
    *,
    custom_style: str | None = None,
    palette: str = "color",
    detail: str = "balanced",
) -> tuple[str, str]:
    """The positive and negative prompt for one picture.

    A custom style with no description falls back to the default style, the same as an
    unknown id does.
    """
    custom = custom_style_text(custom_style) if style_id == CUSTOM_STYLE else ""
    style = Style(custom) if custom else STYLES.get(style_id, STYLES[DEFAULT_STYLE])
    parts = [_clean(prompt), style.positive]
    if palette == "grayscale":
        parts.append(GRAYSCALE_POSITIVE)
    parts.append(DETAIL_POSITIVE.get(detail, ""))
    if theme == "dark":
        parts.append(DARK_PAGE_POSITIVE)
    if purpose == "cover":
        parts.append(COVER_POSITIVE)
    negatives = [BASE_NEGATIVE, style.negative]
    if palette == "grayscale":
        negatives.append(GRAYSCALE_NEGATIVE)
    negative = ", ".join(part for part in negatives if part)
    return ", ".join(part for part in parts if part), negative
