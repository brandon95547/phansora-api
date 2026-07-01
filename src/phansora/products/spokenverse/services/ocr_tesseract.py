# src/services/ocr_tesseract.py
#
# OCR page images (bytes) -> raw text using local Tesseract
#
# System dependency (Pop!_OS/Ubuntu):
#   sudo apt update
#   sudo apt install -y tesseract-ocr
#
# Python deps (venv):
#   python -m pip install pytesseract pillow

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

import pytesseract
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

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    text = pytesseract.image_to_string(
        img,
        lang=cfg.lang,
        config=cfg.to_tesseract_config(),
    )
    return (text or "").strip()


def ocr_image_file(
    image_path: str,
    *,
    cfg: TesseractOCRConfig = TesseractOCRConfig(),
) -> str:
    """
    OCR a single image file path and return raw extracted text.
    """
    img = Image.open(image_path).convert("RGB")
    text = pytesseract.image_to_string(
        img,
        lang=cfg.lang,
        config=cfg.to_tesseract_config(),
    )
    return (text or "").strip()
