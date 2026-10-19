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
    # Each is its default substyle (model.js SUBSTYLE_DEFAULTS): Realism is Photorealism,
    # Abstract is Geometric Abstraction, Comic Book is Modern Comics…
    "realism": "photorealistic illustration, lifelike proportions, natural light, crisp true-to-life detail and texture",
    "impressionism": "impressionist painting, visible broken brushstrokes, dappled light, soft edges, vibrant atmospheric color",
    "tonalism": "tonalist painting, a soft muted palette of grays and browns, misty dusk atmosphere, quiet contemplative mood",
    "expressionism": "expressionist art, bold distorted forms, intense emotional color, energetic gestural marks",
    "abstract-expressionism": "abstract expressionist action painting, large gestural brushstrokes, drips and splatters, raw energetic color",
    "surrealism": "surrealist art, a dreamlike impossible scene, uncanny juxtapositions, the strange rendered precisely",
    "abstract": "abstract art, bold geometric shapes, precise lines and flat planes of color, forms suggested rather than literal",
    "minimalism": "minimalist illustration, simple clean shapes, a limited palette, generous negative space",
    "art-nouveau": "art nouveau illustration, flowing organic lines, ornamental floral borders, elegant decorative curves",
    "art-deco": "art deco style, bold geometric symmetry, sleek stylized forms, gold and black accents, 1920s glamour",
    "classical": "classical academic painting, idealized figures, balanced composition, chiaroscuro, old master technique",
    "cartoon": "cartoon illustration, friendly exaggerated characters, bold clean outlines, bright flat colors",
    "anime": "anime and manga style, expressive large eyes, clean cel-shaded line art, vibrant colors",
    "comic-book": "modern comic book art, bold clean inks, dynamic poses, digital coloring, cinematic lighting",
}

# Each art style's substyles (model.js SUBSTYLES): the art style it belongs to, and its own
# words, used instead of the art style's — a substyle is a more particular way of it. Ids are
# unique across art styles. A substyle sent with another art style is not used.
SUBSTYLES: dict[str, tuple[str, str]] = {
    "fantastic-realism": ("realism", "fantastic realism, mythic and otherworldly subjects painted with meticulous lifelike detail"),
    "biomechanical": ("realism", "biomechanical art, organic forms fused with machinery, ribbed metallic textures, intricate surreal anatomy"),
    "magical-realism": ("realism", "magical realism, an everyday scene painted realistically with one quiet impossible element, calm luminous mood"),
    "neo-impressionism": ("impressionism", "neo-impressionist pointillism, small dots of pure color, divisionist technique, luminous optical mixing"),
    "atmospheric-tonalism": ("tonalism", "atmospheric tonalism, veils of mist and haze, soft glowing light, a near-monochrome palette, dreamy stillness"),
    "fauvism": ("expressionism", "fauvist painting, wild unnatural pure color, flat bold areas, loose expressive brushwork"),
    "neo-expressionism": ("expressionism", "neo-expressionist painting, crude raw figures, thick violent brushwork, vivid clashing color"),
    "color-field": ("abstract-expressionism", "color field painting, broad expanses of flat soaked color, soft edges, meditative calm"),
    "organic-abstraction": ("abstract", "organic abstraction, flowing biomorphic shapes, natural curves, soft layered color"),
    "lyrical-abstraction": ("abstract", "lyrical abstraction, loose poetic brushwork, luminous washes, free flowing color"),
    "hard-edge-painting": ("minimalism", "hard-edge minimalism, crisp sharp-edged shapes of flat color, high contrast, clean abrupt transitions"),
    "glasgow-style": ("art-nouveau", "Glasgow style, elongated vertical lines, stylized roses, a restrained palette of mauve and green, delicate geometric ornament"),
    "streamline-moderne": ("art-deco", "streamline moderne, sleek aerodynamic curves, horizontal speed lines, polished chrome, nautical motifs"),
    "egyptian-revival": ("art-deco", "Egyptian revival art deco, lotus and scarab motifs, pyramids, gold, lapis and black, symmetrical stylized figures"),
    "rococo": ("classical", "Rococo painting, playful pastel colors, light airy brushwork, ornate scrolls, romantic garden scenes"),
    "rubber-hose": ("cartoon", "rubber hose cartoon, 1930s animation, noodle limbs, pie-cut eyes, white gloves, muted vintage color"),
    "storybook-cartoon": ("cartoon", "storybook cartoon illustration, gentle rounded characters, soft textured color, cozy picture-book charm"),
    "modern-cartoon": ("cartoon", "modern cartoon style, flat vector shapes, thick clean lines, bold saturated colors, simple geometric characters"),
    "retro-anime": ("anime", "retro anime, 1980s and 1990s hand-painted cel animation, soft film grain, muted colors, detailed painted backgrounds"),
    "chibi": ("anime", "chibi anime style, tiny super-deformed characters, oversized heads, cute simple features"),
    "painterly-anime": ("anime", "painterly anime style, soft hand-painted backgrounds, lush natural light, gentle watercolor-like color, storybook atmosphere"),
    "realistic-anime": ("anime", "realistic anime style, realistic proportions and anatomy, refined linework, detailed shading, cinematic lighting"),
    "golden-age": ("comic-book", "golden age comic book art, 1940s, simple bold inks, flat primary colors, halftone dots, heroic poses"),
    "silver-age": ("comic-book", "silver age comic book art, 1960s, clean confident inking, bright flat color, dramatic dynamic panels"),
    "european-comics": ("comic-book", "European bande dessinée, ligne claire, clean even outlines, flat clear colors, detailed backgrounds"),
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


def look_text(art_style: str | None, medium: str | None = None, substyle: str | None = None) -> str:
    """An art style's words (its substyle's, when it has one of its own), then its medium's;
    empty for an art style this API does not know."""
    art = ART_STYLES.get(art_style or "")
    if not art:
        return ""
    parent, words = SUBSTYLES.get(substyle or "", ("", ""))
    if words and parent == art_style:
        art = words
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


def style_text(
    style_id: str, custom_style: str | None = None, art_style: str | None = None, medium: str | None = None, substyle: str | None = None
) -> str:
    """The style's prompt language: what a picture in this style looks like. A known art style
    (with its substyle and medium) wins over the older single style id."""
    look = look_text(art_style, medium, substyle)
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
    substyle: str | None = None,
) -> tuple[str, str]:
    """The positive and negative prompt for one picture.

    A known art style (with its medium) is the look; without one, the older style id is. A
    custom style with no description falls back to the default style, the same as an
    unknown id does; "none" adds no style's words at all.
    """
    custom = custom_style_text(custom_style) if style_id == CUSTOM_STYLE else ""
    look = look_text(art_style, medium, substyle)
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
