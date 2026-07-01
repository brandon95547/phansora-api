# txt_to_voice/utils/naming.py

from __future__ import annotations

import re
from pathlib import Path
from typing import List


def iter_txt_files(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.glob("*.txt") if p.is_file()])


def sanitize_stem(stem: str) -> str:
    stem = stem.strip()
    stem = re.sub(r"\s+", " ", stem)
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "", stem)
    return stem.strip() or "output"
