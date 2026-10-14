"""Progress of the model download that the ComfyUI container runs on first start.

That container writes a small JSON file as it works; reading it here lets
the app say "downloading, 43%" instead of only "ComfyUI is not reachable".
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# A progress file nobody has touched for this long describes a download that is no longer running.
STALE_AFTER_S = 120.0


def _gb(size: float) -> str:
    return f"{size / 1e9:.1f}"


def download_message(status_file: Path | None) -> str | None:
    """A sentence about the model download, or None when there is nothing to report."""
    if status_file is None:
        return None
    try:
        status = json.loads(status_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(status, dict):
        return None

    state = status.get("state")
    if state == "error":
        return f"The illustration model could not be downloaded: {status.get('message', 'unknown error')}"
    if state != "downloading":
        return None
    if time.time() - float(status.get("updatedAt", 0)) > STALE_AFTER_S:
        return "The illustration model download has stalled; it resumes when the ComfyUI container restarts."

    done = float(status.get("doneBytes", 0))
    total = float(status.get("totalBytes", 0))
    if total <= 0:
        return "Downloading the illustration model (first start only)."
    percent = min(99, int(done * 100 / total))
    return (
        f"Downloading the illustration model, first start only: {percent}% "
        f"({_gb(done)} of {_gb(total)} GB)."
    )
