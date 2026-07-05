"""Unified Phansora API application.

Each product ships a self-contained FastAPI app (``phansora.products.<name>``).
This module composes them into one process, mounting each under a path prefix:

    /spokenverse/*   -> SpokenVerse (PDF/OCR, text->audio, Book Alchemy)
    /chrono/*        -> Chrono-Origin (story/myth origin tracing)
    /dossier/*       -> Dossier Nova (AI research & dossier generation)

A product is mounted only if it imports cleanly, so a host that is missing one
product's optional heavy dependencies (torch, cosyvoice, asyncpg, ...)
still serves the others instead of failing to boot. Each mounted sub-app keeps
its own middleware and startup/shutdown lifespan — we propagate those from the
parent lifespan below.

Run with:  uvicorn phansora.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import importlib
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from phansora.config import settings

logger = logging.getLogger("phansora")

# prefix -> "module:attr" of the product's FastAPI app
_PRODUCT_APPS = {
    "/spokenverse": "phansora.products.spokenverse.server:app",
    "/chrono": "phansora.products.chrono_origin.server:app",
    "/dossier": "phansora.products.dossier_nova.api:app",
}
# prefix -> product key used by Settings.enabled_products
_PREFIX_TO_KEY = {
    "/spokenverse": "spokenverse",
    "/chrono": "chrono_origin",
    "/dossier": "dossier_nova",
}


def _load_products() -> Dict[str, FastAPI]:
    """Import each product app, skipping (with a warning) any that fail to load."""
    loaded: Dict[str, FastAPI] = {}
    enabled = set(settings.enabled_products)
    for prefix, target in _PRODUCT_APPS.items():
        if enabled and _PREFIX_TO_KEY[prefix] not in enabled:
            logger.info("Skipping %s (not in PHANSORA_ENABLED_PRODUCTS)", prefix)
            continue
        module_name, attr = target.split(":")
        try:
            module = importlib.import_module(module_name)
            loaded[prefix] = getattr(module, attr)
            logger.info("Mounted product at %s", prefix)
        except Exception as exc:  # noqa: BLE001 — resilient boot by design
            logger.warning("Could not load product %s (%s): %s", prefix, target, exc)
    return loaded


_products = _load_products()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Propagate each mounted sub-app's lifespan (startup/shutdown)."""
    async with AsyncExitStack() as stack:
        for prefix, sub in _products.items():
            try:
                await stack.enter_async_context(sub.router.lifespan_context(sub))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Lifespan startup failed for %s: %s", prefix, exc)
        yield


app = FastAPI(title=settings.app_name, version=settings.version, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for _prefix, _sub in _products.items():
    app.mount(_prefix, _sub)


@app.get("/")
def root():
    return {
        "name": settings.app_name,
        "version": settings.version,
        "products": {prefix: _PREFIX_TO_KEY[prefix] for prefix in _products},
    }


@app.get("/health")
def health():
    return {"status": "ok", "mounted": sorted(_products.keys())}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("phansora.main:app", host=settings.host, port=settings.port, reload=True)
