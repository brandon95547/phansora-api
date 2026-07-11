"""Shared filename + upload helpers used across products (SpokenVerse, Book
Alchemy, …). Kept in ``shared/`` because they are generic file-handling
utilities, not specific to any one product."""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import HTTPException, UploadFile


def safe_ext(filename: str) -> str:
    """Lower-cased file extension (including the dot), or '' if none."""
    return (Path(filename).suffix or "").lower()


def safe_stem(filename: str, fallback: str) -> str:
    """Filesystem-safe stem derived from a filename, or ``fallback``."""
    stem = Path(filename).stem.strip()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or fallback


async def save_upload(upload: UploadFile, dest_path: Path) -> None:
    """Persist an uploaded file to ``dest_path`` (creating parents). Raises a
    400 if the upload is empty."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    dest_path.write_bytes(data)
