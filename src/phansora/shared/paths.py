"""Filesystem anchors for runtime and seed data.

Before consolidation, each product resolved its working directories from its own
``__file__`` (which equalled the project root the process ran from). Now that the
code lives inside the installed ``phansora`` package, ``__file__`` points *into*
the package — so products must anchor mutable data on the process's runtime root
instead. Deployments set this via the working directory (systemd
``WorkingDirectory``) or the ``PHANSORA_DATA_DIR`` env var.
"""
from __future__ import annotations

import os
from pathlib import Path


def runtime_root() -> Path:
    """Base directory for mutable runtime data (uploads, generated audio/text,
    Book Alchemy storage, the embeddings DB).

    Defaults to the current working directory; override with ``PHANSORA_DATA_DIR``.
    """
    return Path(os.getenv("PHANSORA_DATA_DIR", os.getcwd())).resolve()


def runtime_dir(*parts: str) -> Path:
    """A path under :func:`runtime_root`, e.g. ``runtime_dir("output_audio")``."""
    return runtime_root().joinpath(*parts)
