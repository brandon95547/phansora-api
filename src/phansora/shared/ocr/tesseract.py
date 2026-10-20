# phansora/shared/ocr/tesseract.py — a page picture's text, by the local Tesseract engine.
#
# System dependency (Pop!_OS/Ubuntu): sudo apt install -y tesseract-ocr
# Python: pytesseract, pillow. pytesseract is imported when a page is read, not with this
# module, so a product that imports the shared OCR still loads on a host without it.

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class TesseractOCRConfig:
    """
    lang: Tesseract language(s), e.g. "eng" or "eng+spa"
    psm: page segmentation mode (3 is a good default for full pages)
    oem: OCR engine mode (3 = default LSTM)
    """
    lang: str = "eng"
    psm: int = 3
    oem: int = 3

    def to_tesseract_config(self) -> str:
        return f"--oem {self.oem} --psm {self.psm}"


def ocr_image_bytes(
    image_bytes: bytes,
    *,
    cfg: TesseractOCRConfig = TesseractOCRConfig(),
) -> str:
    """
    OCR a single image (bytes) and return raw extracted text.
    """
    if not image_bytes:
        return ""
    import pytesseract

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    text = pytesseract.image_to_string(
        img,
        lang=cfg.lang,
        config=cfg.to_tesseract_config(),
    )
    return (text or "").strip()

