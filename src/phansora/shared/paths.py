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


def assets_root() -> Path:
    """Base directory for shipped, read-only asset data that travels with the repo.

    The counterpart to :func:`runtime_root`: that one holds mutable per-user state,
    this one holds content we author and version (the CosyVoice2 reference clip for
    the built-in "default" voice, and the app-wide default voices). Anything here is
    the same on every deployment and arrives via ``git pull``, so it must never be
    written to at runtime.

    Anchored on this file's location — ``<repo>/src/phansora/shared/paths.py`` — so it
    resolves to ``<repo>/assets``. That is deliberately the opposite of
    :func:`runtime_root`: the warning in this module's docstring is about *mutable* data,
    which must follow the deployment rather than the code. Assets are the reverse — they
    ship with the code, so tying them to the package is what keeps them correct no matter
    where the data dir points (on prod, ``PHANSORA_DATA_DIR`` is ``<repo>/data``, so
    deriving assets from the runtime root would wrongly yield ``<repo>/data/assets``).

    Assumes an editable/checkout install, which is how this is deployed. Override with
    ``PHANSORA_ASSETS_DIR`` if the package is ever installed as a copied wheel.
    """
    override = os.getenv("PHANSORA_ASSETS_DIR", "").strip()
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[3] / "assets"


def assets_dir(*parts: str) -> Path:
    """A path under :func:`assets_root`, e.g. ``assets_dir("voices")``."""
    return assets_root().joinpath(*parts)
