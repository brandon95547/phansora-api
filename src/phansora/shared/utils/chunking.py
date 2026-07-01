# txt_to_voice/utils/chunking.py

from __future__ import annotations

import re
from typing import List


def chunk_text(text: str, max_chars: int) -> List[str]:
    """
    Chunk text to avoid service limits. Uses paragraph + sentence-ish splitting.
    """
    text = text.strip()
    if not text:
        return []

    paragraphs = re.split(r"\n\s*\n+", text)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) <= max_chars:
            extra = (2 if current else 0) + len(para)
            if current_len + extra <= max_chars:
                current.append(para)
                current_len += extra
            else:
                flush()
                current.append(para)
                current_len = len(para)
            continue

        flush()
        sentences = re.split(r"(?<=[.!?])\s+", para)
        buf = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if not buf:
                buf = s
            elif len(buf) + 1 + len(s) <= max_chars:
                buf = f"{buf} {s}"
            else:
                chunks.append(buf)
                buf = s
        if buf:
            chunks.append(buf)

    flush()
    return [c for c in chunks if c.strip()]
