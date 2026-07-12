"""Admin storage endpoints — disk/cache usage and safe cache deletion.

Guarded by an ``X-Admin-Key`` header matched against ``PHANSORA_ADMIN_KEY``.
Unset key => locked (403). Mounted on the core app in ``phansora.main``:

    GET  /admin/storage/info           -> disk + per-directory sizes
    POST /admin/storage/clear-model-caches   -> delete re-downloadable model caches
    POST /admin/storage/clear-chrono-cache   -> delete only the Chrono trace cache

Deliberately conservative: model-cache clearing only removes caches that
re-download automatically (HuggingFace, torch hub). It NEVER touches the
CosyVoice2 weights (manual re-download), the Dossier Nova embeddings DB
(expensive to recompute), or the Chrono cache (that has its own button).
"""
from __future__ import annotations

import hmac
import logging
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Header, HTTPException

import phansora
from phansora.shared.paths import runtime_root

logger = logging.getLogger("phansora.admin")

router = APIRouter(prefix="/admin", tags=["admin"])


# ---- auth -------------------------------------------------------------------
def require_admin(x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key")) -> None:
    key = os.getenv("PHANSORA_ADMIN_KEY", "").strip()
    if not key or not x_admin_key or not hmac.compare_digest(x_admin_key, key):
        raise HTTPException(status_code=403, detail="Admin key required")


# ---- path resolution --------------------------------------------------------
def _api_dir() -> Path:
    override = os.getenv("PHANSORA_API_DIR")
    if override:
        return Path(override)
    # .../<repo>/src/phansora/__init__.py -> parents[2] == repo root
    return Path(phansora.__file__).resolve().parents[2]


def _frontend_dir() -> Path:
    return Path(os.getenv("PHANSORA_FRONTEND_DIR", "/var/www/phansora"))


def _chrono_cache_dir() -> Path:
    try:
        from phansora.products.chrono_origin.config import get_settings as chrono_settings

        return Path(chrono_settings().chrono_cache_dir).resolve()
    except Exception:  # noqa: BLE001 — product may not be loaded
        return Path(os.getenv("CHRONO_CACHE_DIR", "./data/chrono_origin/cache")).resolve()


def _hf_cache_dir() -> Path:
    hf = os.getenv("HF_HOME")
    return Path(hf) if hf else Path.home() / ".cache" / "huggingface"


def _torch_cache_dir() -> Path:
    th = os.getenv("TORCH_HOME")
    return Path(th) if th else Path.home() / ".cache" / "torch"


def _cosyvoice_dir() -> Path:
    d = os.getenv("COSYVOICE2_MODEL_DIR")
    if d:
        return Path(d)
    repo = os.getenv("COSYVOICE2_REPO", "/var/www/CosyVoice")
    return Path(repo) / "pretrained_models" / "CosyVoice2-0.5B"


def _dossier_embeddings() -> Path:
    return runtime_root() / "data" / "dossier_nova" / "embeddings.db"


# ---- size / delete helpers --------------------------------------------------
def _dir_size(p: Path) -> Optional[int]:
    """Fast, bounded directory size via `du`. Returns bytes, or None if it times
    out / errors — a huge tree (node_modules, .venv) must not hang the request."""
    try:
        if not p.exists():
            return 0
        if p.is_file():
            return p.stat().st_size
        # `du -sk` is portable (Linux + macOS): summary size in 1K blocks.
        out = subprocess.run(
            ["du", "-sk", str(p)], capture_output=True, text=True, timeout=12
        )
        if out.returncode != 0 or not out.stdout.strip():
            return None
        return int(out.stdout.split()[0]) * 1024
    except (subprocess.TimeoutExpired, ValueError, OSError):
        return None


def _clear_contents(p: Path) -> int:
    """Delete everything inside p (keep the dir). Returns bytes freed."""
    if not p.exists():
        return 0
    freed = _dir_size(p) or 0
    for child in p.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("could not remove %s: %s", child, exc)
    return freed


# ---- endpoints --------------------------------------------------------------
@router.get("/storage/info")
def storage_info(x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key")) -> dict:
    require_admin(x_admin_key)
    du = shutil.disk_usage("/")
    # (key, label, path, deletable, kind)
    specs = [
        ("frontend", "Frontend", _frontend_dir(), False, ""),
        ("api", "API code + data", _api_dir(), False, ""),
        ("runtime_data", "Runtime data", runtime_root(), False, ""),
        ("chrono_cache", "Chrono-Origin cache", _chrono_cache_dir(), True, "chrono"),
        ("hf_cache", "HuggingFace model cache", _hf_cache_dir(), True, "model"),
        ("torch_cache", "Torch hub cache", _torch_cache_dir(), True, "model"),
        ("cosyvoice", "CosyVoice2 weights (do not delete)", _cosyvoice_dir(), False, ""),
        ("dossier_embeddings", "Dossier embeddings (expensive to rebuild)", _dossier_embeddings(), False, ""),
    ]
    # Size each dir concurrently so the whole call is bounded by the slowest `du`
    # (~12s), not the sum. bytes=null means it timed out / was too large to size.
    with ThreadPoolExecutor(max_workers=len(specs)) as ex:
        sizes = list(ex.map(lambda s: _dir_size(s[2]), specs))
    entries = [
        {
            "key": key,
            "label": label,
            "path": str(path),
            "exists": path.exists(),
            "bytes": size,
            "deletable": deletable,
            "kind": kind,
        }
        for (key, label, path, deletable, kind), size in zip(specs, sizes)
    ]
    return {
        "ok": True,
        "disk": {"total": du.total, "used": du.used, "free": du.free, "path": "/"},
        "entries": entries,
    }


@router.post("/storage/clear-model-caches")
def clear_model_caches(x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key")) -> dict:
    require_admin(x_admin_key)
    # Only caches that re-download automatically. NOT CosyVoice2 weights, NOT the
    # Dossier embeddings DB, NOT the Chrono cache.
    targets = [_hf_cache_dir(), _torch_cache_dir()]
    freed = sum(_clear_contents(p) for p in targets)
    logger.info("cleared model caches, freed %d bytes", freed)
    return {
        "ok": True,
        "freed_bytes": freed,
        "cleared": [str(p) for p in targets],
        "note": "Model files re-download on next use; restart the service to reload cleanly.",
    }


@router.post("/storage/clear-chrono-cache")
def clear_chrono_cache(x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key")) -> dict:
    require_admin(x_admin_key)
    d = _chrono_cache_dir()
    freed = _clear_contents(d)
    logger.info("cleared chrono cache at %s, freed %d bytes", d, freed)
    return {"ok": True, "freed_bytes": freed, "cleared": [str(d)]}
