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
import tempfile
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from .config import get_settings  # noqa: E402
from .models import (  # noqa: E402
    MediaSearchRequest,
    MediaSearchResponse,
    SceneSuggestRequest,
    SceneSuggestResponse,
    ScriptGenerateRequest,
    ScriptGenerateResponse,
    SegmentRequest,
    SegmentResponse,
    StoryboardRequest,
    StoryboardResponse,
    TimelineBuildRequest,
    TimelineBuildResponse,
)
from .services import align, llm, media, script, storyboard, timeline, voices  # noqa: E402

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
                doc_style=req.doc_style,
                research=req.research,
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
            # Interactive search reports "no results" itself — a fake placeholder tile
            # would look like a real result the editor could place.
            allow_placeholder=False,
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


@app.post("/scene/suggest", response_model=SceneSuggestResponse)
async def suggest_scene(req: SceneSuggestRequest):
    """Fresh visual direction for ONE placeholder that is already correctly placed.

    Deliberately not /storyboard: that one re-cuts the narration and demands measured word
    timings, neither of which makes sense when the user is asking "give me better ideas for
    this shot" about a scene whose position is already right.
    """
    _ensure_llm()
    try:
        suggestion = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: storyboard.suggest_for_span(
                req.text, count=req.count, have=req.have, style=req.style,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Scene suggestion failed")
        raise HTTPException(status_code=502, detail=f"Scene suggestion failed: {exc}")
    return SceneSuggestResponse(
        rationale=suggestion.get("rationale", ""),
        media_type=suggestion.get("media_type", "image"),
        search_terms=suggestion.get("search_terms", []),
        visual_prompt=suggestion.get("visual_prompt", ""),
        visual_ideas=suggestion.get("visual_ideas", []),
        ideas=suggestion.get("ideas", []),
        degraded=bool(suggestion.get("degraded", False)),
    )


@app.post("/narration/align")
async def align_narration(
    file: UploadFile = File(...),
    full_text: str = Form(...),
    language: Optional[str] = Form(None),
    total_duration_sec: Optional[float] = Form(None),
):
    """Measured word timings for a rendered narration, for /storyboard to lay scenes on.

    Kept separate from /storyboard rather than folded into it: the timings are useful on
    their own (captions, subtitles, a word-accurate scrubber) and this way the storyboard
    endpoint stays a plain JSON call.

    Failure is an error, not a degraded result. 503 means the host cannot do this at all
    and an operator has to fix it; 422 means this particular audio and script do not go
    together and the user has to.
    """
    suffix = os.path.splitext(file.filename or "")[1][:8] or ".mp3"
    fd, temp_path = tempfile.mkstemp(prefix="narrava_align_", suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := await file.read(1 << 20):
                out.write(chunk)
        times = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: align.word_times(
                temp_path,
                full_text,
                language=language,
                total_duration_sec=total_duration_sec,
            ),
        )
    except align.AlignmentUnavailable as exc:
        logger.error("Narration alignment unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except align.AlignmentFailed as exc:
        logger.warning("Narration alignment failed: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
    return {"aligned": True, "words": times}


@app.post("/storyboard", response_model=StoryboardResponse)
async def build_storyboard(req: StoryboardRequest):
    """AI first visual pass: narration -> media-placeholder scenes at story-flow
    boundaries, each with media suggestions."""
    _ensure_llm()
    try:
        scenes = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: storyboard.build_storyboard(
                req.full_text,
                total_duration_sec=req.total_duration_sec,
                max_scenes=req.max_scenes,
                word_times=req.word_times,
                style=req.style,
            ),
        )
    except storyboard.NarrationNotTimed as exc:
        # Placeholders that are only nearly on the voice are the bug, so an untimed
        # narration is refused rather than approximated.
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Storyboard build failed")
        raise HTTPException(status_code=502, detail=f"Storyboard build failed: {exc}")
    return StoryboardResponse(scenes=scenes)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8010"))
    uvicorn.run("phansora.products.narrava_studio.server:app", host="0.0.0.0", port=port, reload=True)
