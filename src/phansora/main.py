"""Unified Phansora API application.

Each product ships a self-contained FastAPI app (``phansora.products.<name>``).
This module composes them into one process, mounting each under a path prefix:

    /spokenverse/*   -> SpokenVerse (PDF/OCR, text->audio, Book Alchemy)
    /chrono/*        -> Chrono-Origin (story/myth origin tracing)
    /research-atlas/* -> Research Atlas (organizes research material into a report)
    /dossier/*       -> alias of /research-atlas, kept so the Node worker keeps
                        resolving during the rename; remove once it points at the new prefix

A product is mounted only if it imports cleanly, so a host that is missing one
product's optional heavy dependencies (torch, cosyvoice2, vllm, asyncpg, ...)
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
from phansora.shared.admin.router import router as admin_router
from phansora.shared.auth import AuthGate
from phansora.shared.contact import router as contact_router
from phansora.shared.conversions.router import router as conversions_router

logger = logging.getLogger("phansora")

# prefix -> "module:attr" of the product's FastAPI app
_PRODUCT_APPS = {
    "/spokenverse": "phansora.products.spokenverse.server:app",
    "/chrono": "phansora.products.chrono_origin.server:app",
    "/research-atlas": "phansora.products.research_atlas.api:app",
    # Same app under the old prefix. Mounting both means the API can be deployed
    # before the Node worker is updated, instead of the two having to land together.
    "/dossier": "phansora.products.research_atlas.api:app",
    "/book-alchemy": "phansora.products.book_alchemy.server:app",
    "/studio": "phansora.products.narrava_studio.server:app",
}
# prefix -> product key (used in the root() response)
_PREFIX_TO_KEY = {
    "/spokenverse": "spokenverse",
    "/chrono": "chrono_origin",
    "/research-atlas": "research_atlas",
    "/dossier": "research_atlas",
    "/book-alchemy": "book_alchemy",
    "/studio": "narrava_studio",
}


def _load_products() -> Dict[str, FastAPI]:
    """Import every product app, skipping (with a warning) any that fail to import
    (e.g. a host missing that product's optional heavy deps)."""
    loaded: Dict[str, FastAPI] = {}
    for prefix, target in _PRODUCT_APPS.items():
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

# The gate is added FIRST so CORS wraps it: an unauthenticated 401 must still carry
# CORS headers, or the browser reports an opaque network error instead of the truth.
app.add_middleware(AuthGate)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    # No cookies cross this API — auth is a bearer token or an internal header —
    # and wildcard origins with credentials is the one combination the CORS spec
    # forbids outright. The sub-apps already say False for the same reason.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

for _prefix, _sub in _products.items():
    app.mount(_prefix, _sub)

# Cross-cutting endpoints that belong to no product live on the core app.
app.include_router(contact_router)
app.include_router(conversions_router)
app.include_router(admin_router)


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
