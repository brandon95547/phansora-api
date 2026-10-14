"""Talari Book Creator — FastAPI app, mounted at /talari.

Illustrations only, for now:

    GET  /status                                   is ComfyUI ready to draw, and if not, why
    POST /users/{user_id}/illustrations            draw one picture (about 85 s on the A4000)
    GET  /users/{user_id}/images/{name}            a picture drawn earlier

Pictures are kept per user under the runtime root (``talari/images/<user_id>/``). The
user id travels in the path, so ``enforce_user_scope`` holds a browser token to its own
pictures; the Node app, calling with the internal key, may name any user — which is why
the id is also checked for shape before it becomes part of a path.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from phansora.shared.auth import enforce_user_scope

from .comfy import ComfyUI
from .config import Settings, load_settings
from .downloads import download_message
from .errors import ServiceError
from .schemas import IllustrationRequest, IllustrationResponse, ServiceState, Status

logger = logging.getLogger("talari")

USER_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
IMAGE_NAME = re.compile(r"^[0-9a-f]{32}\.png$")


def create_app(settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    """Build the app. Tests pass their own settings and a fake HTTP transport."""
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with httpx.AsyncClient(transport=transport) as client:
            app.state.comfy = ComfyUI(settings, client)
            yield

    app = FastAPI(title="Talari Book Creator API", lifespan=lifespan, dependencies=[Depends(enforce_user_scope)])

    @app.exception_handler(ServiceError)
    async def service_error(_: Request, error: ServiceError) -> JSONResponse:
        logger.warning("Talari: %s", error.message)
        return JSONResponse(status_code=error.status, content={"detail": error.message})

    def user_images(user_id: str):
        if not USER_ID.match(user_id):
            raise HTTPException(status_code=400, detail="That user id is not valid.")
        return settings.images_dir / user_id

    @app.get("/")
    def root() -> dict[str, str]:
        return {"name": "talari", "status": "ok"}

    @app.get("/status", response_model=Status, response_model_by_alias=True)
    async def status(request: Request) -> Status:
        problem = await request.app.state.comfy.problem()
        if problem is None:
            return Status(illustration=ServiceState(available=True))
        # While the models are still arriving, say that rather than "not reachable".
        detail = download_message(settings.model_status_file) or problem
        return Status(illustration=ServiceState(available=False, detail=detail))

    @app.post("/users/{user_id}/illustrations", response_model=IllustrationResponse, response_model_by_alias=True)
    async def illustrate(user_id: str, body: IllustrationRequest, request: Request) -> IllustrationResponse:
        folder = user_images(user_id)
        image, seed = await request.app.state.comfy.generate(body)
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{uuid.uuid4().hex}.png"
        (folder / name).write_bytes(image)
        return IllustrationResponse(src=f"/talari/users/{user_id}/images/{name}", seed=seed)

    @app.get("/users/{user_id}/images/{name}")
    def image(user_id: str, name: str) -> FileResponse:
        if not IMAGE_NAME.match(name):
            raise HTTPException(status_code=404, detail="No such picture.")
        path = user_images(user_id) / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="No such picture.")
        return FileResponse(path, media_type="image/png")

    return app


app = create_app()
