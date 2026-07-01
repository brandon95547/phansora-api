"""On-disk layout for Book Alchemy source files and rendered audio.

Lives under the spokenverse project root (same disk as output_audio/), one
folder per user/project:  book_alchemy/<user_id>/<project_id>/
"""
from __future__ import annotations

import re
from pathlib import Path

# spokenverse project root = .../spokenverse (parents: book_alchemy -> src -> root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT / "book_alchemy"


def safe_id(value: str | int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._-")
    if not cleaned:
        raise ValueError("invalid id")
    return cleaned


def project_dir(user_id: str | int, project_id: str | int) -> Path:
    d = BASE_DIR / safe_id(user_id) / safe_id(project_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def session_audio_path(user_id: str | int, project_id: str | int, ordinal: int, fmt: str = "mp3") -> Path:
    return project_dir(user_id, project_id) / f"session_{int(ordinal):03d}.{fmt}"
