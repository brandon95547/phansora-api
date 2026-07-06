# src/server.py
#
# FastAPI backend for:
#  - PDF -> TXT (PDF rendered to images -> Tesseract OCR -> DeepSeek cleanup/merge)
#  - Audio -> TXT (speech transcription)
#  - TXT -> Audio (IndexTTS2)
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

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse

from phansora.shared.paths import runtime_root
from phansora.shared.utils.email import send_email
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


# ----------------------------
# Contact email
# ----------------------------

@app.post("/send-email")
async def send_email_endpoint(request: Request) -> dict:
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    try:
        result = await send_email(data)
    except ValueError as e:
        # Validation problem with the request payload.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # SMTP / delivery failure — must not report success.
        raise HTTPException(status_code=502, detail=f"Failed to send email: {str(e)}")

    return {"status": result}


# ----------------------------
# Helpers
# ----------------------------

def _safe_ext(filename: str) -> str:
    return (Path(filename).suffix or "").lower()


def _safe_stem(filename: str, fallback: str) -> str:
    stem = Path(filename).stem.strip()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or fallback


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


async def _save_upload(upload: UploadFile, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    dest_path.write_bytes(data)


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

def _parse_emo_vector(raw: Optional[str]) -> Optional[list]:
    """Parse an emotion vector from a form field: JSON array or comma-separated floats.
    Returns a list of floats, or None if absent/unparseable."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        data = json.loads(s)
        if isinstance(data, list):
            return [float(x) for x in data]
    except (ValueError, TypeError):
        pass
    try:
        return [float(x) for x in s.split(",") if x.strip() != ""]
    except ValueError:
        return None


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
    speed: Optional[float] = Form(None),  # 0.5-2.0; playback speed (ffmpeg atempo)
    emo_alpha: Optional[float] = Form(None),  # expressiveness weight 0-1
    emo_vector: Optional[str] = Form(None),  # 8 comma/JSON floats (EMO_LABELS order)
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

    # A non-"default" voice may be one of the user's saved cloned voices; resolve
    # its id to the on-disk reference clip IndexTTS2 clones from, and fall back to the
    # voice's approved emotion settings when the request doesn't override them.
    resolved_voice = voice
    emo_vec = _parse_emo_vector(emo_vector)
    if voice and voice != "default":
        clip = voice_store.voice_path(safe_user, voice)
        if clip is not None:
            resolved_voice = str(clip)
        rec = next((v for v in voice_store.list_voices(safe_user) if v.get("id") == voice), None)
        if rec:
            if not language:
                language = rec.get("language")
            if emo_alpha is None and rec.get("emo_alpha") is not None:
                emo_alpha = rec.get("emo_alpha")
            if emo_vec is None and rec.get("emo_vector"):
                emo_vec = rec.get("emo_vector")

    cfg = TTSConfig(
        voice=resolved_voice,
        use_gpu=use_gpu,
        rate=rate,
        volume=volume,
        output_format=output_format,
        chunk_chars=chunk_chars,
        speaker=speaker,
        language=language,
        max_concurrency=max_concurrency,
        file_concurrency=file_concurrency,
        speed=speed,
        emo_alpha=emo_alpha,
        emo_vector=emo_vec,
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
                "emo_alpha": emo_alpha,
                "emo_vector": emo_vec,
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
    Options currently available from the IndexTTS2 backend.
    """
    from phansora.products.spokenverse.txt_to_voice.adapters import indextts2_client as ix

    voices = discover_voices()
    return {
        "backend": "indextts2",
        "voices": voices,
        "languages": ix.LANGUAGES,
        "audio_formats": ["mp3", "wav"],
        "voice_cloning": "supported — pass a reference-clip path as `voice`/`speaker`",
        "controls_available": {
            "language": {"values": ix.LANGUAGES, "default": ix.LANGUAGE_DEFAULT,
                         "description": "language of the synthesized text"},
            "speed": {"min": ix.SPEED_MIN, "max": ix.SPEED_MAX,
                      "default": ix.SPEED_DEFAULT, "description": "playback speed (ffmpeg atempo)"},
            "emo_alpha": {"min": ix.EMO_ALPHA_MIN, "max": ix.EMO_ALPHA_MAX,
                          "default": ix.EMO_ALPHA_DEFAULT, "description": "expressiveness weight"},
            "emo_vector": {"labels": ix.EMO_LABELS, "length": ix.EMO_VECTOR_LEN, "each": [0.0, 1.0],
                           "description": "per-emotion mix (8 weights 0-1); all-zero => inherent emotion"},
            "rate_volume": "`rate`/`volume` accepted for compatibility; ignored by backend",
        },
        "env_overrides": [
            "INDEXTTS2_REPO", "INDEXTTS2_MODEL_DIR", "INDEXTTS2_CONFIG", "INDEXTTS2_FP16",
            "INDEXTTS2_USE_CUDA_KERNEL", "INDEXTTS2_USE_DEEPSPEED", "INDEXTTS2_DEFAULT_REF",
            "INDEXTTS2_LANGUAGE", "INDEXTTS2_SPEED", "INDEXTTS2_EMO_ALPHA",
        ],
    }


# ----------------------------
# Custom voices (IndexTTS2 cloning)
# ----------------------------

# Spoken during the create-voice preview so the user hears the cloned voice, not
# their own raw upload.
VOICE_SAMPLE_TEXT = "This is a sample of your voice. Approve to save in your voices."


@app.post("/voices/preview", response_model=None)
async def voice_preview(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    language: Optional[str] = Form(None),  # en/zh/ja/ko/yue/auto
    speed: Optional[float] = Form(None),  # 0.5-2.0; playback speed (ffmpeg atempo)
    emo_alpha: Optional[float] = Form(None),  # expressiveness weight 0-1
    emo_vector: Optional[str] = Form(None),  # 8 comma/JSON floats (EMO_LABELS order)
) -> dict:
    """Upload a reference clip, then synthesize a sample the user can preview.

    The upload is trimmed and normalized to a 24kHz mono WAV reference clip.
    IndexTTS2 clones from the clip; we still auto-transcribe it (whisper) and store
    the text as ``ref_text`` for reference, then run the engine to speak
    ``VOICE_SAMPLE_TEXT`` with the chosen emotion. That synthesized sample is what
    the user hears. Returns a token used to preview and then approve/discard.
    Nothing is saved as a usable voice until it is approved.
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

    knobs = voice_store.clamp_settings(
        language=language, speed=speed, emo_alpha=emo_alpha,
        emo_vector=_parse_emo_vector(emo_vector),
    )

    # Auto-transcribe the reference clip and store it as ref_text (informational;
    # IndexTTS2 clones from the clip alone). Best-effort: skipped on failure.
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
            **knobs,
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
            speed=knobs["speed"],
            emo_alpha=knobs["emo_alpha"],
            emo_vector=knobs["emo_vector"],
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
async def voice_list(user_id: str) -> dict:
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
# Book Alchemy (book/long-form -> structured audio course)
# ----------------------------
#
# Browser talks to these endpoints directly with the session user_id (same
# convention as the rest of the spokenverse dashboard). Durable job state lives
# in Postgres; the standalone book_alchemy_worker.py does the heavy processing.
# Imports are guarded so that if the Book Alchemy deps (asyncpg, ebooklib, ...)
# are not yet installed, the rest of the spokenverse API still boots.
try:
    from phansora.products.spokenverse.book_alchemy import db as ba_db
    from phansora.products.spokenverse.book_alchemy import storage as ba_storage
    _BOOK_ALCHEMY_OK = True
except Exception as _ba_exc:  # noqa: BLE001
    _BOOK_ALCHEMY_OK = False
    import logging as _logging

    _logging.getLogger("book_alchemy").warning(
        "Book Alchemy routes disabled (import failed): %s", _ba_exc
    )

if _BOOK_ALCHEMY_OK:
    _BA_FILE_FORMATS = {
        ".pdf": "pdf", ".epub": "epub", ".mobi": "mobi", ".azw": "mobi", ".azw3": "mobi",
        ".docx": "docx", ".txt": "txt", ".md": "markdown", ".markdown": "markdown",
        ".html": "html", ".htm": "html",
    }

    def _ba_user_id(user_id: str) -> int:
        try:
            return int(str(user_id).strip())
        except Exception:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="Valid numeric user_id is required.")

    def _ba_json(value):
        if value is None or isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value)
        except Exception:  # noqa: BLE001
            return value

    def _ba_project_wire(row) -> dict:
        d = dict(row)
        return {
            "project_id": d["id"],
            "name": d["name"],
            "source_format": d["source_format"],
            "status": d["status"],
            "stage": d["stage"],
            "progress": d["progress"],
            "validation_status": d["validation_status"],
            "total_audio_seconds": d["total_audio_seconds"],
            "sessions_complete": int(d.get("sessions_complete") or 0),
            "sessions_total": int(d.get("sessions_total") or 0),
            "curriculum": _ba_json(d.get("curriculum")),
            "error": d.get("error_message"),
            "created_at": d["created_at"].isoformat() if d.get("created_at") else None,
            "updated_at": d["updated_at"].isoformat() if d.get("updated_at") else None,
        }

    def _ba_session_wire(row, ref_map: dict | None = None) -> dict:
        d = dict(row)
        chunk_ids = list(d.get("source_chunk_ids") or [])
        sources = []
        if ref_map:
            seen = set()
            for cid in chunk_ids:
                ref = ref_map.get(cid)
                if not ref:
                    continue
                key = (ref.get("chapter"), ref.get("page_start"), ref.get("page_end"))
                if key in seen:
                    continue
                seen.add(key)
                sources.append(ref)
        return {
            "session_id": d["id"],
            "ordinal": d["ordinal"],
            "title": d["title"],
            "summary": d["summary"],
            "status": d["status"],
            "validation_status": d["validation_status"],
            "validation_notes": _ba_json(d.get("validation_notes")),
            "outline": _ba_json(d.get("outline")),
            "script": d.get("script"),
            "source_chunk_ids": chunk_ids,
            "sources": sources,
            "audio_seconds": d.get("audio_seconds"),
            "has_audio": bool(d.get("audio_path")),
            "generated_at": d["generated_at"].isoformat() if d.get("generated_at") else None,
        }

    @app.post("/book-alchemy/projects")
    async def ba_create_project(
        user_id: str = Form(...),
        name: str = Form(""),
        source_format: str = Form(""),
        url: str = Form(""),
        text: str = Form(""),
        voice: str = Form("default"),
        file: Optional[UploadFile] = File(None),
    ) -> dict:
        uid = _ba_user_id(user_id)
        url = (url or "").strip()
        text = (text or "").strip()

        # Determine source format + a default project name.
        if file is not None and file.filename:
            ext = _safe_ext(file.filename)
            fmt = source_format or _BA_FILE_FORMATS.get(ext)
            if not fmt:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext or 'unknown'}")
            default_name = _safe_stem(file.filename, "Untitled")
        elif url:
            fmt = "url"
            default_name = url
        elif text:
            fmt = "text"
            default_name = "Pasted text"
        else:
            raise HTTPException(status_code=400, detail="Provide a file, a URL, or pasted text.")

        proj_name = (name or default_name or "Untitled").strip()[:200]
        project_id = await ba_db.create_project(
            user_id=uid, name=proj_name, source_format=fmt,
            source_path=None, source_url=(url or None), options={"voice": voice},
        )

        # Persist the source so processing is fully resumable from disk + DB.
        if file is not None and file.filename:
            dest = ba_storage.project_dir(uid, project_id) / f"source{_safe_ext(file.filename) or '.bin'}"
            await _save_upload(file, dest)
            await ba_db.set_project(project_id, source_path=str(dest))
        elif text:
            dest = ba_storage.project_dir(uid, project_id) / "source.txt"
            dest.write_text(text, encoding="utf-8")
            await ba_db.set_project(project_id, source_path=str(dest))

        return {"ok": True, "project_id": project_id}

    @app.get("/book-alchemy/projects")
    async def ba_list_projects(user_id: str) -> dict:
        uid = _ba_user_id(user_id)
        rows = await ba_db.list_projects(uid)
        return {"ok": True, "projects": [_ba_project_wire(r) for r in rows]}

    @app.get("/book-alchemy/projects/{project_id}")
    async def ba_get_project(project_id: int, user_id: str) -> dict:
        uid = _ba_user_id(user_id)
        row = await ba_db.get_project(project_id, uid)
        if row is None:
            raise HTTPException(status_code=404, detail="Project not found.")

        chunks = await ba_db.get_all_chunks(project_id)
        ref_map = {
            int(c["id"]): {
                "chapter": c["chapter"], "section": c["section"],
                "page_start": c["page_start"], "page_end": c["page_end"],
            }
            for c in chunks
        }
        sessions = await ba_db.get_sessions(project_id)
        out = _ba_project_wire(row)
        out["sessions"] = [_ba_session_wire(s, ref_map) for s in sessions]
        return {"ok": True, "project": out}

    @app.get("/book-alchemy/projects/{project_id}/sessions/{session_id}/audio", response_model=None)
    async def ba_session_audio(project_id: int, session_id: int, user_id: str) -> FileResponse:
        uid = _ba_user_id(user_id)
        project = await ba_db.get_project(project_id, uid)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found.")
        sess = await ba_db.get_session(session_id, project_id)
        if sess is None or not sess["audio_path"]:
            raise HTTPException(status_code=404, detail="Audio not found.")
        path = Path(sess["audio_path"])
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="Audio file missing on disk.")
        return FileResponse(path=str(path), media_type="audio/mpeg", filename=path.name)

    def _ba_safe_filename(name: str) -> str:
        """Sanitize a title for use as a file / zip-entry name (keeps spaces)."""
        cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', " ", str(name or "")).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned[:120] or "course"

    def _ba_build_zip(items, course_name: str) -> str:
        """Build a zip of session audio (named by session title). Blocking; run
        in a thread. Returns the temp zip path."""
        import tempfile
        import zipfile

        tmp = tempfile.NamedTemporaryFile(prefix="ba_zip_", suffix=".zip", delete=False)
        tmp.close()
        with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as zf:
            for ordinal, title, path in items:
                ext = path.suffix or ".mp3"
                arcname = f"{ordinal:02d} - {_ba_safe_filename(title)}{ext}"
                zf.write(str(path), arcname=arcname)
        return tmp.name

    def _ba_unlink_quiet(path: str) -> None:
        try:
            os.unlink(path)
        except Exception:  # noqa: BLE001
            pass

    @app.get("/book-alchemy/projects/{project_id}/download", response_model=None)
    async def ba_download_project(project_id: int, user_id: str) -> FileResponse:
        from starlette.background import BackgroundTask

        uid = _ba_user_id(user_id)
        project = await ba_db.get_project(project_id, uid)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found.")

        sessions = await ba_db.get_sessions(project_id)
        items = []
        for s in sessions:
            ap = s["audio_path"]
            if ap and Path(ap).exists() and Path(ap).is_file():
                items.append((int(s["ordinal"]), s["title"], Path(ap)))
        if not items:
            raise HTTPException(status_code=404, detail="No audio is available to download yet.")

        course_name = _ba_safe_filename(project["name"] or f"course_{project_id}")
        zip_path = await asyncio.to_thread(_ba_build_zip, items, course_name)
        return FileResponse(
            path=zip_path,
            media_type="application/zip",
            filename=f"{course_name}.zip",
            background=BackgroundTask(_ba_unlink_quiet, zip_path),
        )

    @app.post("/book-alchemy/projects/{project_id}/sessions/{session_id}/regenerate")
    async def ba_regenerate_session(project_id: int, session_id: int, user_id: str) -> dict:
        uid = _ba_user_id(user_id)
        project = await ba_db.get_project(project_id, uid)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found.")
        sess = await ba_db.get_session(session_id, project_id)
        if sess is None:
            raise HTTPException(status_code=404, detail="Session not found.")

        if sess["audio_path"]:
            try:
                Path(sess["audio_path"]).unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass

        await ba_db.set_session(
            session_id, status="pending", validation_status="pending",
            script=None, audio_path=None, audio_seconds=None, validation_notes=None,
        )
        # Re-open the project at the sessions phase so the worker re-scripts,
        # re-validates and re-renders just this session.
        await ba_db.set_project(
            project_id, status="processing", phase="sessions",
            stage="Regenerating session", lease_owner=None, lease_expires_at=None,
        )
        return {"ok": True}

    @app.delete("/book-alchemy/projects/{project_id}")
    async def ba_delete_project(project_id: int, user_id: str) -> dict:
        import shutil

        uid = _ba_user_id(user_id)
        row = await ba_db.delete_project(project_id, uid)
        if row is None:
            raise HTTPException(status_code=404, detail="Project not found.")
        try:
            shutil.rmtree(ba_storage.project_dir(uid, project_id), ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass
        return {"ok": True, "deleted": project_id}


# ----------------------------
# Run directly (optional)
# ----------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
