"""What is done to a picture after ComfyUI draws it: its background taken away, for a picture
placed over a page, and an enlargement, for one that fills a printed page.

Background removal is rembg (U²-Net family, on the CPU through onnxruntime). It is optional:
without it installed a picture keeps its background, and the caller is told so. The first use
downloads its model (about 170 MB) into ``~/.u2net``.
"""

from __future__ import annotations

import io
import logging
import threading

from PIL import Image

logger = logging.getLogger("talari")

# isnet-general-use keeps fine edges (hair, leaves) better than the default u2net.
MATTING_MODEL = "isnet-general-use"
_session = None
_lock = threading.Lock()


def _rembg():
    """rembg's remove() and a session, made once; None when rembg is not installed."""
    global _session
    try:
        from rembg import new_session, remove  # noqa: PLC0415 — optional, and heavy to import
    except ImportError:
        logger.warning("Talari: rembg is not installed, so a transparent picture keeps its background.")
        return None
    with _lock:
        if _session is None:
            _session = new_session(MATTING_MODEL)
    return remove, _session


def remove_background(image: bytes) -> bytes | None:
    """The picture as a PNG with its background transparent, or None when that cannot be done."""
    tools = _rembg()
    if tools is None:
        return None
    remove, session = tools
    try:
        cut = remove(Image.open(io.BytesIO(image)), session=session, post_process_mask=True)
    except Exception:  # noqa: BLE001 — a picture that cannot be cut keeps its background
        logger.exception("Talari: taking a picture's background away failed")
        return None
    out = io.BytesIO()
    cut.save(out, format="PNG", optimize=True)
    return out.getvalue()


def upscale(image: bytes, factor: float) -> bytes:
    """The picture enlarged by ``factor`` with Lanczos resampling, as a PNG (alpha kept)."""
    if factor <= 1:
        return image
    with Image.open(io.BytesIO(image)) as picture:
        picture.load()
        size = (round(picture.width * factor), round(picture.height * factor))
        out = io.BytesIO()
        picture.resize(size, Image.Resampling.LANCZOS).save(out, format="PNG", optimize=True)
    return out.getvalue()
