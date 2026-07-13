# src/server.py
#
# FastAPI backend for:
#  - PDF -> TXT (PDF rendered to images -> Tesseract OCR -> DeepSeek cleanup/merge)
#  - Audio -> TXT (speech transcription)
#  - TXT -> Audio (CosyVoice2)
#
# Run:
#   uvicorn server:app --host 0.0.0.0 --port 8000 --reload
#
# Notes:
# - This file intentionally contains NO frontend code.
# - PdfConverter loads .env from project root automatically (see pdf_pipeline.py).

from __future__ import annotations

import asyncio
import os
import re
import uuid
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse

from phansora.shared.paths import runtime_root
from phansora.shared.utils.uploads import (
    safe_ext as _safe_ext,
    safe_stem as _safe_stem,
    save_upload as _save_upload,
)
from phansora.products.spokenverse.txt_to_voice.adapters.backend import discover_voices, get_synthesizer
from phansora.products.spokenverse.txt_to_voice.pdf_pipeline import PdfConverter, PdfToTxtConfig
from phansora.products.spokenverse.txt_to_voice.pipeline import BatchConverter, TTSConfig
from phansora.products.spokenverse import voices as voice_store


# ----------------------------
# Paths
# ----------------------------

PROJECT_ROOT = runtime_root()
OUTPUT_TXT_DIR = PROJECT_ROOT / "output_txt"
OUTPUT_AUDIO_DIR = PROJECT_ROOT / "output_audio"
TMP_UPLOADS_DIR = PROJECT_ROOT / ".tmp_uploads"

OUTPUT_TXT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TMP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


_WHISPER_MODEL = None
_WHISPER_MODEL_NAME = None
_WHISPER_MODEL_LOCK = Lock()


# ----------------------------
# App
# ----------------------------

app = FastAPI(title="spokenverse-api", version="1.0.0")

_cors_origins_raw = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
)
_cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.on_event("startup")
async def _sweep_stale_pending_voices() -> None:
    # Clear pending voice clips abandoned before the last restart. Best-effort:
    # never let cleanup failures block startup.
    try:
        voice_store.prune_all_pending()
    except Exception:
        pass


@app.on_event("startup")
async def _preload_tts_model() -> None:
    # Warm the TTS engine in the background so the first generation doesn't pay the
    # one-time load. For CosyVoice2 this is the big one — weights + vLLM CUDA-graph
    # capture (~80s) + first-run TensorRT engine build. Off-thread so startup and
    # /health stay responsive; the load is lock-guarded, so a request that races the
    # warmup just waits on the same load instead of starting a second one.
    import threading

    from phansora.products.spokenverse.txt_to_voice.adapters import backend as tts_backend

    threading.Thread(
        target=tts_backend.preload, name="cosyvoice2-preload", daemon=True
    ).start()


# ----------------------------
# Helpers
# ----------------------------

def _safe_user_id(user_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (user_id or "").strip()).strip("._-")
    if not cleaned:
        raise HTTPException(status_code=400, detail="user_id is required.")
    return cleaned


def _is_supported_audio_ext(ext: str) -> bool:
    return ext in {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".webm"}


def _user_txt_dir(user_id: str) -> Path:
    out = OUTPUT_TXT_DIR / _safe_user_id(user_id)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _user_audio_dir(user_id: str) -> Path:
    out = OUTPUT_AUDIO_DIR / _safe_user_id(user_id)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _append_history_event(user_id: str, event: dict) -> None:
    history_file = PROJECT_ROOT / "history" / _safe_user_id(user_id) / "events.jsonl"
    history_file.parent.mkdir(parents=True, exist_ok=True)
    event_with_time = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **event,
    }
    with history_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event_with_time, ensure_ascii=True) + "\n")


def _load_whisper_model(model_name: str):
    global _WHISPER_MODEL, _WHISPER_MODEL_NAME

    with _WHISPER_MODEL_LOCK:
        if _WHISPER_MODEL is not None and _WHISPER_MODEL_NAME == model_name:
            return _WHISPER_MODEL

        try:
            from faster_whisper import WhisperModel
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=(
                    "faster-whisper is not installed. Install dependencies in spokenverse "
                    "to enable local audio transcription."
                ),
            ) from e

        device = os.getenv("WHISPER_DEVICE", "cpu")
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
        cpu_threads_raw = os.getenv("WHISPER_CPU_THREADS", "").strip()
        num_workers_raw = os.getenv("WHISPER_NUM_WORKERS", "").strip()

        kwargs = {
            "device": device,
            "compute_type": compute_type,
        }
        if cpu_threads_raw.isdigit():
            kwargs["cpu_threads"] = max(1, int(cpu_threads_raw))
        if num_workers_raw.isdigit():
            kwargs["num_workers"] = max(1, int(num_workers_raw))

        _WHISPER_MODEL = WhisperModel(model_name, **kwargs)
        _WHISPER_MODEL_NAME = model_name
        return _WHISPER_MODEL


def _transcribe_audio_to_text_sync(
    audio_path: Path,
    model: str,
    language: Optional[str],
) -> str:
    whisper_model = _load_whisper_model(model)

    beam_size = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
    vad_filter = os.getenv("WHISPER_VAD_FILTER", "true").strip().lower() in {"1", "true", "yes", "on"}

    segments, _ = whisper_model.transcribe(
        str(audio_path),
        beam_size=max(1, beam_size),
        vad_filter=vad_filter,
        language=(language or None),
    )

    text_parts = []
    for segment in segments:
        seg_text = str(getattr(segment, "text", "")).strip()
        if seg_text:
            text_parts.append(seg_text)

    transcript = " ".join(text_parts).strip()
    if not transcript:
        raise HTTPException(status_code=502, detail="Local transcription returned empty text.")
    return transcript


async def _transcribe_audio_to_text(
    audio_path: Path,
    model: str,
    language: Optional[str] = None,
) -> str:
    return await asyncio.to_thread(
        _transcribe_audio_to_text_sync,
        audio_path,
        model,
        language,
    )


# ----------------------------
# PDF -> TXT
# ----------------------------

@app.post("/pdf-to-txt", response_model=None)
async def pdf_to_txt(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    dpi: int = Form(250),
    ocr_lang: str = Form("eng"),
    batch_pages: int = Form(5),
    keep_page_breaks: bool = Form(True),
    to_chapters: bool = Form(False),
    ocr_concurrency: int = Form(4),
    clean_concurrency: int = Form(2),
    target_chapter_chars: int = Form(18000),
    return_file: bool = Form(True),
) -> FileResponse | PlainTextResponse | dict:
    """
    Upload a PDF and return cleaned .txt.
    - return_file=true: returns the .txt as a file download
    - return_file=false: returns plain text in the response body
    """
    ext = _safe_ext(file.filename or "")
    if ext != ".pdf":
        raise HTTPException(status_code=400, detail="Please upload a .pdf file.")

    safe_user = _safe_user_id(user_id)
    job_id = uuid.uuid4().hex
    source_stem = _safe_stem(file.filename or "", fallback=f"pdf_{job_id}")
    pdf_path = TMP_UPLOADS_DIR / f"{job_id}.pdf"
    user_txt_dir = _user_txt_dir(safe_user)
    out_txt_path = user_txt_dir / f"{source_stem}__{job_id}.txt"

    await _save_upload(file, pdf_path)

    try:
        cfg = PdfToTxtConfig(
            render_dpi=dpi,
            tesseract_lang=ocr_lang,
            batch_pages=batch_pages,
            keep_page_breaks=keep_page_breaks,
            to_chapters=to_chapters,
            ocr_concurrency=ocr_concurrency,
            clean_concurrency=clean_concurrency,
            target_chapter_chars=target_chapter_chars,
        )
        converter = PdfConverter(cfg)
        result_path = await converter.convert_pdf_to_txt_async(pdf_path, out_txt_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF->TXT failed: {e}") from e
    finally:
        # keep temp PDF only if you want to debug; otherwise remove it
        try:
            pdf_path.unlink(missing_ok=True)
        except Exception:
            pass

    _append_history_event(
        safe_user,
        {
            "type": "pdf_to_txt",
            "job_id": job_id,
            "input_filename": file.filename,
            "output_path": str(result_path),
            "options": {
                "dpi": dpi,
                "ocr_lang": ocr_lang,
                "batch_pages": batch_pages,
                "keep_page_breaks": keep_page_breaks,
                "to_chapters": to_chapters,
                "ocr_concurrency": ocr_concurrency,
                "clean_concurrency": clean_concurrency,
                "target_chapter_chars": target_chapter_chars,
            },
        },
    )

    if result_path.is_dir():
        txt_files = sorted(str(p.name) for p in result_path.glob("*.txt") if p.is_file())
        return {
            "user_id": safe_user,
            "job_id": job_id,
            "output_dir": str(result_path),
            "files": txt_files,
        }

    if return_file:
        return FileResponse(
            path=str(result_path),
            media_type="text/plain; charset=utf-8",
            filename=result_path.name,
        )

    text = result_path.read_text(encoding="utf-8", errors="replace")
    return PlainTextResponse(text)


# ----------------------------
# TXT -> Audio
# ----------------------------

@app.post("/txt-to-audio", response_model=None)
async def txt_to_audio(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    voice: str = Form("default"),
    use_gpu: bool = Form(False),
    rate: str = Form("+0%"),
    volume: str = Form("+0%"),
    speaker: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    output_format: Literal["mp3", "wav"] = Form("mp3"),
    chunk_chars: int = Form(2500),
    max_concurrency: int = Form(4),
    file_concurrency: int = Form(1),
    speed: Optional[float] = Form(None),  # 0.5-2.0; native CosyVoice2 speed
) -> FileResponse | dict:
    """
    Upload a .txt and return an audio file (mp3/wav).
    """
    ext = _safe_ext(file.filename or "")
    if ext != ".txt":
        raise HTTPException(status_code=400, detail="Please upload a .txt file.")

    safe_user = _safe_user_id(user_id)
    job_id = uuid.uuid4().hex
    source_stem = _safe_stem(file.filename or "", fallback=f"txt_{job_id}")

    # Put the single uploaded txt into a temp folder so we can reuse convert_folder()
    tmp_in_dir = TMP_UPLOADS_DIR / f"in_{job_id}"
    tmp_in_dir.mkdir(parents=True, exist_ok=True)

    txt_path = tmp_in_dir / f"{source_stem}.txt"
    await _save_upload(file, txt_path)
    user_audio_dir = _user_audio_dir(safe_user)

    # A non-"default" voice may be one of the user's saved cloned voices; resolve its id to
    # the on-disk reference clip CosyVoice2 clones from, plus its stored transcript
    # (``ref_text``) — CosyVoice conditions on that transcript (prompt_text). Fall back to
    # the voice's saved language when the request doesn't override it.
    resolved_voice = voice
    prompt_text: Optional[str] = None
    if voice and voice != "default":
        clip = voice_store.voice_path(safe_user, voice)
        if clip is not None:
            resolved_voice = str(clip)
        rec = next((v for v in voice_store.list_voices(safe_user) if v.get("id") == voice), None)
        if rec:
            if not language:
                language = rec.get("language")
            prompt_text = rec.get("ref_text") or None

    cfg = TTSConfig(
        voice=resolved_voice,
        use_gpu=use_gpu,
        rate=rate,
        volume=volume,
        output_format=output_format,
        chunk_chars=chunk_chars,
        speaker=speaker,
        language=language,
        prompt_text=prompt_text,
        max_concurrency=max_concurrency,
        file_concurrency=file_concurrency,
        speed=speed,
    )

    try:
        converter = BatchConverter(cfg)
        rc = await converter.convert_folder(tmp_in_dir, user_audio_dir)
        if rc != 0:
            raise RuntimeError(f"TTS conversion returned non-zero status: {rc}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TXT->Audio failed: {e}") from e
    finally:
        # Cleanup temp input folder
        try:
            for p in tmp_in_dir.glob("*"):
                p.unlink(missing_ok=True)
            tmp_in_dir.rmdir()
        except Exception:
            pass

    # The audio filename is based on the txt stem; here it's job_id
    audio_path = user_audio_dir / f"{source_stem}.{output_format}"
    if not audio_path.exists():
        raise HTTPException(status_code=500, detail="Audio file was not created.")

    _append_history_event(
        safe_user,
        {
            "type": "txt_to_audio",
            "job_id": job_id,
            "input_filename": file.filename,
            "output_path": str(audio_path),
            "options": {
                "voice": voice,
                "speaker": speaker,
                "language": language,
                "use_gpu": use_gpu,
                "rate": rate,
                "volume": volume,
                "output_format": output_format,
                "chunk_chars": chunk_chars,
                "max_concurrency": max_concurrency,
                "file_concurrency": file_concurrency,
                "speed": speed,
            },
        },
    )

    return FileResponse(
        path=str(audio_path),
        media_type="audio/mpeg" if output_format == "mp3" else "audio/wav",
        filename=audio_path.name,
    )


# ----------------------------
# Audio -> TXT
# ----------------------------

@app.post("/audio-to-txt", response_model=None)
async def audio_to_txt(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    model: str = Form(os.getenv("WHISPER_MODEL", "small")),
    language: Optional[str] = Form(None),
    return_file: bool = Form(True),
) -> FileResponse | PlainTextResponse | dict:
    """
    Upload an audio file and return transcribed .txt.
    - return_file=true: returns the .txt as a file download
    - return_file=false: returns plain text in the response body
    """
    ext = _safe_ext(file.filename or "")
    if not _is_supported_audio_ext(ext):
        raise HTTPException(
            status_code=400,
            detail="Please upload a supported audio file: mp3, wav, m4a, aac, ogg, flac, webm.",
        )

    safe_user = _safe_user_id(user_id)
    job_id = uuid.uuid4().hex
    source_stem = _safe_stem(file.filename or "", fallback=f"audio_{job_id}")
    audio_path = TMP_UPLOADS_DIR / f"{job_id}{ext}"
    user_txt_dir = _user_txt_dir(safe_user)
    out_txt_path = user_txt_dir / f"{source_stem}__{job_id}.txt"

    await _save_upload(file, audio_path)

    try:
        transcript = await _transcribe_audio_to_text(
            audio_path=audio_path,
            model=model,
            language=language,
        )
        out_txt_path.write_text(transcript, encoding="utf-8")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio->TXT failed: {e}") from e
    finally:
        try:
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass

    _append_history_event(
        safe_user,
        {
            "type": "audio_to_txt",
            "job_id": job_id,
            "input_filename": file.filename,
            "output_path": str(out_txt_path),
            "options": {
                "model": model,
                "language": language,
            },
        },
    )

    if return_file:
        return FileResponse(
            path=str(out_txt_path),
            media_type="text/plain; charset=utf-8",
            filename=out_txt_path.name,
        )

    return PlainTextResponse(transcript)


@app.get("/tts-options")
async def tts_options() -> dict:
    """
    Options currently available from the CosyVoice2 backend.
    """
    from phansora.products.spokenverse.txt_to_voice.adapters import cosyvoice2_client as cv

    voices = discover_voices()
    return {
        "backend": "cosyvoice2",
        "voices": voices,
        "languages": cv.LANGUAGES,
        "audio_formats": ["mp3", "wav"],
        "voice_cloning": "supported — clone from a reference clip + its transcript (ref_text)",
        "controls_available": {
            "language": {"values": cv.LANGUAGES, "default": cv.LANGUAGE_DEFAULT,
                         "description": "language of the synthesized text"},
            "speed": {"min": cv.SPEED_MIN, "max": cv.SPEED_MAX,
                      "default": cv.SPEED_DEFAULT, "description": "native CosyVoice2 speed (mel time-scaling)"},
            "rate_volume": "`rate`/`volume` accepted for compatibility; ignored by backend",
        },
        "env_overrides": [
            "COSYVOICE2_REPO", "COSYVOICE2_MODEL_DIR", "COSYVOICE2_FP16",
            "COSYVOICE2_USE_VLLM", "COSYVOICE2_USE_TRT", "COSYVOICE2_DEFAULT_REF",
            "COSYVOICE2_DEFAULT_REF_TEXT", "COSYVOICE2_LANGUAGE", "COSYVOICE2_SPEED",
            "COSYVOICE2_MAX_CHARS",
        ],
    }


# ----------------------------
# Custom voices (CosyVoice2 cloning)
# ----------------------------

# Spoken during the create-voice preview so the user hears the cloned voice, not
# their own raw upload.
VOICE_SAMPLE_TEXT = "This is a sample of your voice. Approve to save in your voices."


@app.post("/voices/preview", response_model=None)
async def voice_preview(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    language: Optional[str] = Form(None),  # en/zh/ja/ko/yue/auto
    speed: Optional[float] = Form(None),  # 0.5-2.0; native CosyVoice2 speed
) -> dict:
    """Upload a reference clip, then synthesize a sample the user can preview.

    The upload is trimmed and normalized to a 24kHz mono WAV reference clip.
    CosyVoice2 clones from the clip PLUS its transcript, so we auto-transcribe it
    (whisper) and store the text as ``ref_text`` — that transcript is required at
    synthesis (passed as prompt_text). We then run the engine to speak
    ``VOICE_SAMPLE_TEXT``; that synthesized sample is what the user hears. Returns a
    token used to preview and then approve/discard. Nothing is saved as a usable
    voice until it is approved.
    """
    safe_user = _safe_user_id(user_id)
    # Opportunistically drop abandoned previews (uploaded but never approved or
    # discarded) so pending clips don't accumulate on disk.
    voice_store.prune_pending(safe_user)
    job_id = uuid.uuid4().hex
    ext = _safe_ext(file.filename or "") or ".bin"
    tmp_path = TMP_UPLOADS_DIR / f"voice_{job_id}{ext}"
    await _save_upload(file, tmp_path)
    try:
        result = voice_store.create_pending(safe_user, tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process audio: {e}") from e
    finally:
        tmp_path.unlink(missing_ok=True)

    token = result["token"]
    ref_clip = voice_store.pending_path(safe_user, token)
    if ref_clip is None:
        raise HTTPException(status_code=400, detail="Could not process audio.")

    knobs = voice_store.clamp_settings(language=language, speed=speed)

    # Auto-transcribe the reference clip and store it as ref_text. CosyVoice2 REQUIRES this
    # transcript at synthesis (prompt_text). Best-effort: if it fails, ref_text stays empty
    # and the preview below will surface the "needs transcript" error.
    ref_text = ""
    try:
        model = os.getenv("WHISPER_MODEL", "base")
        # Let whisper auto-detect the reference clip's language (None = auto).
        wlang = knobs["language"] if knobs["language"] in ("en", "zh", "ja", "ko") else None
        ref_text = (await asyncio.to_thread(
            _transcribe_audio_to_text_sync, ref_clip, model, wlang
        )).strip()
    except Exception as tr_err:  # noqa: BLE001
        print(f"[create-voice] ref transcription skipped: {tr_err}", flush=True)

    try:
        synthesize_to_file = get_synthesizer()
        sample_out = voice_store.pending_sample_path(safe_user, token)
        engine_call = {
            "text": VOICE_SAMPLE_TEXT, "out_path": str(sample_out), "voice": str(ref_clip),
            "prompt_text": ref_text, **knobs,
        }
        # TESTING: log the exact call sent to the engine on create-voice generate.
        print(f"[create-voice] synthesize_to_file <- {json.dumps(engine_call)}", flush=True)
        await synthesize_to_file(
            text=VOICE_SAMPLE_TEXT,
            out_path=sample_out,
            voice=str(ref_clip),
            use_gpu=False,
            rate="+0%",
            volume="+0%",
            language=knobs["language"],
            prompt_text=ref_text,  # CosyVoice conditions on the ref clip's transcript
            speed=knobs["speed"],
        )
    except Exception as e:
        import traceback
        print("[create-voice] synthesis failed:", flush=True)
        traceback.print_exc()
        voice_store.discard_pending(safe_user, token)
        raise HTTPException(status_code=500, detail=f"Could not generate a voice sample: {e}") from e
    # Remember the knobs + reference transcript so approval can persist them.
    voice_store.save_pending_settings(safe_user, token, ref_text=ref_text, **knobs)
    return result


@app.get("/voices/preview/{token}", response_model=None)
async def voice_preview_audio(token: str, user_id: str) -> FileResponse:
    safe_user = _safe_user_id(user_id)
    # Serve the synthesized sample (what the user approves on), not the raw upload.
    sample = voice_store.pending_sample_path(safe_user, token)
    if sample.exists():
        return FileResponse(path=str(sample), media_type="audio/wav", filename=f"{token}.wav")
    raise HTTPException(status_code=404, detail="Preview not found or expired.")


@app.post("/voices/{token}/approve", response_model=None)
async def voice_approve(
    token: str,
    user_id: str = Form(...),
    name: str = Form(""),
) -> dict:
    try:
        rec = voice_store.approve(_safe_user_id(user_id), token, name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if rec is None:
        raise HTTPException(status_code=404, detail="Preview not found or expired.")
    return {"ok": True, "voice": rec}


@app.post("/voices/{token}/discard", response_model=None)
async def voice_discard(token: str, user_id: str = Form(...)) -> dict:
    voice_store.discard_pending(_safe_user_id(user_id), token)
    return {"ok": True}


@app.get("/voices", response_model=None)
async def voice_list(user_id: str, response: Response) -> dict:
    # Per-user, mutates whenever a voice is approved/deleted. Must never be cached, or a
    # freshly saved voice won't appear until a hard refresh (browser/nginx served a stale
    # list). no-store also stops any intermediate proxy from caching it.
    response.headers["Cache-Control"] = "no-store"
    return {"ok": True, "voices": voice_store.list_voices(_safe_user_id(user_id))}


@app.get("/voices/{voice_id}/audio", response_model=None)
async def voice_audio(voice_id: str, user_id: str) -> FileResponse:
    safe_user = _safe_user_id(user_id)
    # Play back the synthesized sample (what the user approved). Fall back to the
    # reference clip for voices saved before samples were stored.
    sample = voice_store.voice_sample_path(safe_user, voice_id)
    if sample.exists():
        return FileResponse(path=str(sample), media_type="audio/wav", filename=f"{voice_id}.wav")
    p = voice_store.voice_path(safe_user, voice_id)
    if p is None:
        raise HTTPException(status_code=404, detail="Voice not found.")
    return FileResponse(path=str(p), media_type="audio/wav", filename=f"{voice_id}.wav")


@app.delete("/voices/{voice_id}", response_model=None)
async def voice_delete(voice_id: str, user_id: str) -> dict:
    existed = voice_store.delete_voice(_safe_user_id(user_id), voice_id)
    if not existed:
        raise HTTPException(status_code=404, detail="Voice not found.")
    return {"ok": True}


@app.patch("/voices/{voice_id}", response_model=None)
async def voice_rename(voice_id: str, user_id: str = Form(...), name: str = Form(...)) -> dict:
    try:
        rec = voice_store.rename_voice(_safe_user_id(user_id), voice_id, name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if rec is None:
        raise HTTPException(status_code=404, detail="Voice not found.")
    return {"ok": True, "voice": rec}


@app.get("/users/{user_id}/history")
async def get_user_history(user_id: str) -> dict:
    safe_user = _safe_user_id(user_id)
    user_txt_dir = _user_txt_dir(safe_user)
    user_audio_dir = _user_audio_dir(safe_user)

    txt_files = sorted(
        [
            {
                "name": p.name,
                "path": str(p),
                "size_bytes": p.stat().st_size,
                "modified_utc": datetime.fromtimestamp(
                    p.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            }
            for p in user_txt_dir.rglob("*.txt")
            if p.is_file()
        ],
        key=lambda x: x["modified_utc"],
        reverse=True,
    )
    audio_files = sorted(
        [
            {
                "name": p.name,
                "path": str(p),
                "size_bytes": p.stat().st_size,
                "modified_utc": datetime.fromtimestamp(
                    p.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            }
            for p in user_audio_dir.rglob("*.mp3")
            if p.is_file()
        ]
        + [
            {
                "name": p.name,
                "path": str(p),
                "size_bytes": p.stat().st_size,
                "modified_utc": datetime.fromtimestamp(
                    p.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            }
            for p in user_audio_dir.rglob("*.wav")
            if p.is_file()
        ],
        key=lambda x: x["modified_utc"],
        reverse=True,
    )

    return {
        "user_id": safe_user,
        "txt_files": txt_files,
        "audio_files": audio_files,
    }


@app.get("/users/{user_id}/audio/{filename}")
async def get_user_audio_file(user_id: str, filename: str) -> FileResponse:
    safe_user = _safe_user_id(user_id)
    safe_name = Path(filename).name
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    ext = (Path(safe_name).suffix or "").lower()
    if ext not in {".mp3", ".wav"}:
        raise HTTPException(status_code=400, detail="Only mp3/wav files are supported.")

    audio_path = _user_audio_dir(safe_user) / safe_name
    if not audio_path.exists() or not audio_path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found.")

    return FileResponse(
        path=str(audio_path),
        media_type="audio/mpeg" if ext == ".mp3" else "audio/wav",
        filename=safe_name,
    )


@app.delete("/users/{user_id}/audio/{filename}")
async def delete_user_audio_file(user_id: str, filename: str) -> dict:
    safe_user = _safe_user_id(user_id)
    safe_name = Path(filename).name
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    ext = (Path(safe_name).suffix or "").lower()
    if ext not in {".mp3", ".wav"}:
        raise HTTPException(status_code=400, detail="Only mp3/wav files are supported.")

    audio_path = _user_audio_dir(safe_user) / safe_name
    if not audio_path.exists() or not audio_path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found.")

    try:
        audio_path.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete audio file: {e}") from e

    return {"ok": True, "deleted": safe_name}


# ----------------------------
# Run directly (optional)
# ----------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
