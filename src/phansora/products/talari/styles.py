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

# The look as the app now chooses it: an art style, and a medium (or "auto", which lets the
# art style bring its usual one). Any art style goes with any medium. The ids are the app's
# (ART_STYLES and MEDIUMS in assets/js/admin/talari/model.js).
ART_STYLES: dict[str, str] = {
    "realism": "realistic illustration, lifelike proportions, natural light, true-to-life detail and texture",
    "impressionism": "impressionist painting, visible broken brushstrokes, dappled light, soft edges, vibrant atmospheric color",
    "expressionism": "expressionist art, bold distorted forms, intense emotional color, energetic gestural marks",
    "surrealism": "surrealist art, a dreamlike impossible scene, uncanny juxtapositions, the strange rendered precisely",
    "abstract": "abstract art, bold geometric and organic shapes, expressive fields of color, forms suggested rather than literal",
    "minimalism": "minimalist illustration, simple clean shapes, a limited palette, generous negative space",
    "art-nouveau": "art nouveau illustration, flowing organic lines, ornamental floral borders, elegant decorative curves",
    "art-deco": "art deco style, bold geometric symmetry, sleek stylized forms, gold and black accents, 1920s glamour",
    "classical": "classical academic painting, idealized figures, balanced composition, chiaroscuro, old master technique",
    "cartoon": "cartoon illustration, friendly exaggerated characters, bold clean outlines, bright flat colors",
    "anime": "anime and manga style, expressive large eyes, clean cel-shaded line art, vibrant colors",
    "comic-book": "comic book art, bold inked outlines, dynamic poses, halftone shading, vivid primary colors",
}

MEDIUMS: dict[str, str] = {
    "auto": "",
    "watercolor": "watercolor, soft translucent washes, visible paper grain, loose wet edges",
    "oil-painting": "oil painting, rich impasto brushwork, layered glazes, canvas texture",
    "acrylic": "acrylic painting, opaque vivid color, crisp brushstrokes, matte finish",
    "pencil": "graphite pencil drawing, fine hatching, delicate shading on paper",
    "charcoal": "charcoal drawing, smudged tonal shading, deep blacks, expressive strokes on textured paper",
    "pen-ink": "pen and ink drawing, confident black linework, cross-hatching, high contrast",
    "pastel": "soft pastel drawing, chalky texture, softly blended color, visible paper tooth",
    "engraving": "copperplate engraving, dense fine parallel lines, etched texture",
    "woodcut": "woodcut print, bold carved lines, high-contrast black and white, rough wood grain",
    "collage": "paper collage, cut and torn paper pieces, layered textures, mixed media",
    "digital-painting": "digital painting, smooth painterly rendering, rich lighting, polished detail",
}


def look_text(art_style: str | None, medium: str | None = None) -> str:
    """An art style's words, then its medium's; empty for an art style this API does not know."""
    art = ART_STYLES.get(art_style or "")
    if not art:
        return ""
    return ", ".join(part for part in (art, MEDIUMS.get(medium or "auto", "")) if part)


DEFAULT_STYLE = "watercolor"
CUSTOM_STYLE = "custom"
# No style's words at all: the prompt goes to the model as written. For trying the model
# itself (styleguide/comfyui.html in the app), where any added words would get in the way.
NO_STYLE = "none"
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


def style_text(style_id: str, custom_style: str | None = None, art_style: str | None = None, medium: str | None = None) -> str:
    """The style's prompt language: what a picture in this style looks like. A known art style
    (with its medium) wins over the older single style id."""
    look = look_text(art_style, medium)
    if look:
        return look
    if style_id == NO_STYLE:
        return ""
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
    art_style: str | None = None,
    medium: str | None = None,
) -> tuple[str, str]:
    """The positive and negative prompt for one picture.

    A known art style (with its medium) is the look; without one, the older style id is. A
    custom style with no description falls back to the default style, the same as an
    unknown id does; "none" adds no style's words at all.
    """
    custom = custom_style_text(custom_style) if style_id == CUSTOM_STYLE else ""
    look = look_text(art_style, medium)
    if look:
        style = Style(look)
    elif style_id == NO_STYLE:
        style = Style("")
    else:
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
