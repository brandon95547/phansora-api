"""A picture enlarged after ComfyUI draws it, for one that fills a printed page: the model
draws at about a megapixel, and a page with its bleed wants more."""

from __future__ import annotations

import io

from PIL import Image


def upscale(image: bytes, factor: float) -> bytes:
    """The picture enlarged by ``factor`` with Lanczos resampling, as a PNG."""
    if factor <= 1:
        return image
    with Image.open(io.BytesIO(image)) as picture:
        picture.load()
        size = (round(picture.width * factor), round(picture.height * factor))
        out = io.BytesIO()
        picture.resize(size, Image.Resampling.LANCZOS).save(out, format="PNG", optimize=True)
    return out.getvalue()
