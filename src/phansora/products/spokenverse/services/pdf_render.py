# services/pdf_render.py

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List, Tuple

import fitz  # PyMuPDF
from PIL import Image


@dataclass(frozen=True)
class PdfRenderConfig:
    """
    Render config for OCR prep.
    - dpi: higher = better OCR, slower/costlier. 200–300 is typical.
    - image_format: "jpeg" recommended (smaller than png)
    - jpeg_quality: 70–90 usually fine for OCR
    """
    dpi: int = 250
    image_format: str = "jpeg"  # "jpeg" or "png"
    jpeg_quality: int = 85


def render_pdf_to_images_bytes(
    pdf_bytes: bytes,
    cfg: PdfRenderConfig = PdfRenderConfig(),
) -> List[Tuple[int, bytes]]:
    """
    Render a PDF (bytes) into a list of (page_number, image_bytes).
    Page numbers are 1-based.
    """
    if not pdf_bytes:
        return []

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out: List[Tuple[int, bytes]] = []

    # PyMuPDF uses a transformation matrix for scaling.
    # dpi -> scale factor relative to 72 dpi default.
    scale = cfg.dpi / 72.0
    matrix = fitz.Matrix(scale, scale)

    for idx in range(doc.page_count):
        page = doc.load_page(idx)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        # Convert pixmap to PIL Image
        mode = "RGB"  # alpha=False -> RGB
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)

        buf = io.BytesIO()
        fmt = cfg.image_format.lower()

        if fmt == "png":
            img.save(buf, format="PNG", optimize=True)
        else:
            # default jpeg
            img.save(buf, format="JPEG", quality=cfg.jpeg_quality, optimize=True)

        out.append((idx + 1, buf.getvalue()))

    doc.close()
    return out
