# txt_to_voice/utils/naming.py

from __future__ import annotations

import re


def sanitize_stem(stem: str) -> str:
    stem = stem.strip()
    stem = re.sub(r"\s+", " ", stem)
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "", stem)
    return stem.strip() or "output"
