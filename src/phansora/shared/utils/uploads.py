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


# Matches `client_max_body_size 352m` in nginx's api.phansora.com server block, so the
# public path keeps behaving exactly as it does today — books, scanned PDFs and long
# audio genuinely run to hundreds of megabytes, and a tighter number here would reject
# uploads nginx had already accepted.
#
# It is enforced here ANYWAY because the Book Alchemy worker posts to 127.0.0.1:8000
# directly. That path never passes through nginx, so until now it had no ceiling at all.
DEFAULT_MAX_UPLOAD_BYTES = 352 * 1024 * 1024
_CHUNK = 1024 * 1024


async def save_upload(
    upload: UploadFile,
    dest_path: Path,
    *,
    max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
) -> int:
    """Stream an uploaded file to ``dest_path`` (creating parents). Returns bytes written.

    STREAMED, not ``await upload.read()``. Reading the whole body into memory meant a
    handful of concurrent uploads sat in RAM at once on a single-worker process — and
    with no ceiling on the internal path, "a handful" had no upper bound either. The
    partial file is removed when the limit is hit, so a rejected upload leaves nothing
    behind to be mistaken for a real one.

    Raises 400 on an empty upload, 413 past ``max_bytes``.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        with dest_path.open("wb") as out:
            while chunk := await upload.read(_CHUNK):
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"That file is larger than the {max_bytes // 1024 // 1024} MB limit.",
                    )
                out.write(chunk)
    except BaseException:
        dest_path.unlink(missing_ok=True)
        raise
    if total == 0:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return total
