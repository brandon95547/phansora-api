"""Narrava Studio FastAPI entrypoint.

Mounted by phansora.main under ``/studio``. Endpoints:

  GET  /voices           -> preset narration voices
  POST /script/generate  -> prompt      -> narrator script (timed beats)   [LLM]
  POST /script/segment   -> pasted text -> timed beats                     [no LLM]
  POST /timeline/build   -> beats       -> preliminary media timeline      [LLM+web]

All blocking work (LLM SDK calls, media HTTP) runs in a thread executor so the
event loop stays responsive.
"""
from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from .config import get_settings  # noqa: E402
from .models import (  # noqa: E402
    MediaSearchRequest,
    MediaSearchResponse,
    ScriptGenerateRequest,
    ScriptGenerateResponse,
    SegmentRequest,
    SegmentResponse,
    TimelineBuildRequest,
    TimelineBuildResponse,
)
from .services import llm, media, script, timeline, voices  # noqa: E402

logger = logging.getLogger("narrava-studio")

app = FastAPI(
    title="Narrava Studio API",
    description="AI-assisted video production: script -> timed beats -> media timeline.",
    version="0.1.0",
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"name": "narrava-studio", "status": "ok", "version": "0.1.0"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": settings.provider,
        "configured": llm.provider_configured(),
    }


def _ensure_llm() -> None:
    if not llm.provider_configured():
        raise HTTPException(
            status_code=503,
            detail=f"{llm.required_key_name()} is not configured.",
        )


@app.get("/voices")
def get_voices():
    return {"voices": voices.list_voices()}


@app.post("/script/generate", response_model=ScriptGenerateResponse)
async def generate_script(req: ScriptGenerateRequest):
    _ensure_llm()
    try:
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: script.generate_script(
                req.prompt,
                style=req.style,
                tone=req.tone,
                target_duration_sec=req.target_duration_sec,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Script generation failed")
        raise HTTPException(status_code=502, detail=f"Script generation failed: {exc}")
    return ScriptGenerateResponse(script=result)


@app.post("/script/segment", response_model=SegmentResponse)
async def segment_script(req: SegmentRequest):
    result = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: script.segment_script(
            req.script_text,
            title=req.title or "",
            wpm=req.words_per_minute,
            source="provided",
        ),
    )
    return SegmentResponse(script=result)


@app.post("/media/search", response_model=MediaSearchResponse)
async def search_media(req: MediaSearchRequest):
    """Search fair-use media for a query — powers 'replace this clip'."""
    clips = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: media.search_media(
            req.query,
            segment_id="search",
            media_type=req.media_type,
            limit=req.limit,
        ),
    )
    return MediaSearchResponse(clips=clips)


@app.post("/timeline/build", response_model=TimelineBuildResponse)
async def build_timeline(req: TimelineBuildRequest):
    if not req.segments:
        raise HTTPException(status_code=400, detail="No segments provided.")
    result = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: timeline.build_timeline(
            req.segments,
            voice_id=req.voice_id,
            media_types=req.media_types,
            per_segment=req.per_segment,
        ),
    )
    return TimelineBuildResponse(timeline=result)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8010"))
    uvicorn.run("phansora.products.narrava_studio.server:app", host="0.0.0.0", port=port, reload=True)
