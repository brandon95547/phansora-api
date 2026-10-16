"""Talari Book Creator — FastAPI app, mounted at /talari.

Illustrations, drawn by ComfyUI:

    GET  /status                                   is ComfyUI ready to draw, and if not, why
    POST /users/{user_id}/illustrations            draw one picture (about 85 s on the A4000)
    GET  /users/{user_id}/images/{name}            a picture drawn earlier

The ComfyUI lab, for the styleguide's test page — every workflow input passed through as
given, open only to the accounts in TALARI_LAB_USERS (and the app's internal key):

    GET  /users/{user_id}/lab/options              the models, samplers, schedulers, style files
    POST /users/{user_id}/lab/illustrations        draw one picture with exactly these settings

Text, written by DeepSeek (writing.py), and imported documents (sources.py):

    POST /users/{user_id}/sources/extract          an uploaded PDF/EPUB/DOCX/TXT/MD, as chapters
    POST /users/{user_id}/sources/analyze          what imported material holds: summary, themes, concepts
    POST /users/{user_id}/blueprints               one to three plans for a book
    POST /users/{user_id}/blueprints/refine        a plan changed by a conversation
    POST /users/{user_id}/chapters/write           one chapter, as blocks of inline HTML
    POST /users/{user_id}/text/revise              a passage rewritten, expanded, checked…
    POST /users/{user_id}/illustration-prompts     what to draw for a chapter or a cover

The text routes keep nothing: the app holds the book and sends what each call needs.

Pictures are kept per user under the runtime root (``talari/images/<user_id>/``). The
user id travels in the path, so ``enforce_user_scope`` holds a browser token to its own
pictures and text calls; the Node app, calling with the internal key, may name any user —
which is why the id is also checked for shape before it becomes part of a path.
"""

from __future__ import annotations

import asyncio
import logging
import re
import tempfile
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from phansora.shared.auth import enforce_user_scope

from . import sources, writing
from .comfy import ComfyUI
from .config import Settings, load_settings
from .downloads import download_message
from .errors import ServiceError
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BlueprintRequest,
    BlueprintResponse,
    ChapterRequest,
    ChapterResponse,
    ExtractResponse,
    IllustrationPromptsRequest,
    IllustrationPromptsResponse,
    IllustrationRequest,
    IllustrationResponse,
    LabOptions,
    LabRequest,
    LabResponse,
    RefineRequest,
    RefineResponse,
    ReviseRequest,
    ReviseResponse,
    ServiceState,
    Status,
)

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

    # --- the ComfyUI lab --------------------------------------------------------------

    def lab_user(user_id: str, request: Request) -> None:
        """The lab runs any settings on the GPU, unmetered: only the app and named accounts."""
        # No auth on the request means no gate in front (tests, local development), the same
        # as enforce_user_scope reads it.
        auth = getattr(request.state, "auth", None)
        if auth is None or auth.get("internal") or user_id in settings.lab_users:
            return
        raise HTTPException(status_code=403, detail="The ComfyUI lab is not open to this account.")

    @app.get("/users/{user_id}/lab/options", response_model=LabOptions, response_model_by_alias=True)
    async def lab_options(user_id: str, request: Request) -> LabOptions:
        lab_user(user_id, request)
        return await request.app.state.comfy.lab_options()

    @app.post("/users/{user_id}/lab/illustrations", response_model=LabResponse, response_model_by_alias=True)
    async def lab_illustrate(user_id: str, body: LabRequest, request: Request) -> LabResponse:
        lab_user(user_id, request)
        folder = user_images(user_id)
        image, seed, width, height, workflow = await request.app.state.comfy.lab(body)
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{uuid.uuid4().hex}.png"
        (folder / name).write_bytes(image)
        return LabResponse(src=f"/talari/users/{user_id}/images/{name}", seed=seed, width=width, height=height, workflow=workflow)

    @app.get("/users/{user_id}/images/{name}")
    def image(user_id: str, name: str) -> FileResponse:
        if not IMAGE_NAME.match(name):
            raise HTTPException(status_code=404, detail="No such picture.")
        path = user_images(user_id) / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="No such picture.")
        return FileResponse(path, media_type="image/png")

    # --- text -------------------------------------------------------------------
    # Nothing here is stored, so the user id is only checked for shape (and, by the
    # app-level dependency, against a browser token's own id).

    def check_user(user_id: str) -> None:
        if not USER_ID.match(user_id):
            raise HTTPException(status_code=400, detail="That user id is not valid.")

    @app.post("/users/{user_id}/sources/extract", response_model=ExtractResponse, response_model_by_alias=True)
    async def extract(user_id: str, file: UploadFile = File(...), title: str = Form("")) -> ExtractResponse:
        check_user(user_id)
        fmt = sources.source_format(file.filename)
        if fmt is None:
            raise ServiceError(sources.UNSUPPORTED, status=415)
        with tempfile.TemporaryDirectory(prefix="talari-") as folder:
            path = Path(folder) / f"upload.{fmt}"
            await sources.save_upload(file, path)
            # Parsing a long PDF is seconds of CPU; off the event loop, so other requests carry on.
            return await asyncio.to_thread(sources.extract, path, fmt, filename=file.filename, title=title)

    @app.post("/users/{user_id}/sources/analyze", response_model=AnalyzeResponse, response_model_by_alias=True)
    async def analyze(user_id: str, body: AnalyzeRequest) -> AnalyzeResponse:
        check_user(user_id)
        return await writing.analyze(body)

    @app.post("/users/{user_id}/blueprints", response_model=BlueprintResponse, response_model_by_alias=True)
    async def blueprints(user_id: str, body: BlueprintRequest) -> BlueprintResponse:
        check_user(user_id)
        return await writing.blueprints(body)

    @app.post("/users/{user_id}/blueprints/refine", response_model=RefineResponse, response_model_by_alias=True)
    async def refine(user_id: str, body: RefineRequest) -> RefineResponse:
        check_user(user_id)
        return await writing.refine(body)

    @app.post(
        "/users/{user_id}/chapters/write",
        response_model=ChapterResponse,
        response_model_by_alias=True,
        # A block carries only the fields its type uses: no "level": null on a paragraph.
        response_model_exclude_none=True,
    )
    async def write_chapter(user_id: str, body: ChapterRequest) -> ChapterResponse:
        check_user(user_id)
        return await writing.write_chapter(body)

    @app.post("/users/{user_id}/text/revise", response_model=ReviseResponse, response_model_by_alias=True)
    async def revise(user_id: str, body: ReviseRequest) -> ReviseResponse:
        check_user(user_id)
        return await writing.revise(body)

    @app.post(
        "/users/{user_id}/illustration-prompts",
        response_model=IllustrationPromptsResponse,
        response_model_by_alias=True,
    )
    async def illustration_prompts(user_id: str, body: IllustrationPromptsRequest) -> IllustrationPromptsResponse:
        check_user(user_id)
        return await writing.illustration_prompts(body)

    return app


app = create_app()
