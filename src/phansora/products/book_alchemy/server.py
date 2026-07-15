"""Book Alchemy API — turn a book / long-form document into a structured,
narrated audio course.

Voice generation is delegated to SpokenVerse over HTTP (see
``book_alchemy/audio.py`` -> ``POST /spokenverse/txt-to-audio``); this app owns
everything that is *not* voice generation: projects, curriculum, sessions and
downloads. Durable job state lives in Postgres; the standalone worker
(``book_alchemy/worker.py``) does the heavy processing.

Mounted at ``/book-alchemy`` by ``phansora.main``. Book Alchemy deps (asyncpg,
ebooklib, ...) are imported under a guard so the API still boots if they are
absent.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from phansora.shared.utils.uploads import (
    safe_ext as _safe_ext,
    safe_stem as _safe_stem,
    save_upload as _save_upload,
)


try:
    from phansora.products.book_alchemy import db as ba_db
    from phansora.products.book_alchemy import storage as ba_storage
    _BOOK_ALCHEMY_OK = True
except Exception as _ba_exc:  # noqa: BLE001
    _BOOK_ALCHEMY_OK = False
    import logging as _logging

    _logging.getLogger("book_alchemy").warning(
        "Book Alchemy routes disabled (import failed): %s", _ba_exc
    )


app = FastAPI(title="Book Alchemy", version="0.1.0")

# Mirror the SpokenVerse sub-app's CORS config (the two share the api.phansora.com
# origin and the same CORS_ALLOW_ORIGINS env). The browser calls this API
# cross-origin without cookies — the user is identified by the ?user_id= query
# param — so credentials stay off. allow_credentials=True with a wildcard origin
# is invalid CORS (no usable Access-Control-Allow-Origin is sent), which is what
# blocked generation from https://www.phansora.com.
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


@app.get("/")
def root() -> dict:
    return {"name": "book-alchemy", "status": "ok", "enabled": _BOOK_ALCHEMY_OK}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "enabled": _BOOK_ALCHEMY_OK}


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

    @app.post("/projects")
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

    @app.get("/projects")
    async def ba_list_projects(user_id: str) -> dict:
        uid = _ba_user_id(user_id)
        rows = await ba_db.list_projects(uid)
        return {"ok": True, "projects": [_ba_project_wire(r) for r in rows]}

    @app.get("/projects/{project_id}")
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

    @app.get("/projects/{project_id}/sessions/{session_id}/audio", response_model=None)
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

    @app.get("/projects/{project_id}/download", response_model=None)
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

    @app.post("/projects/{project_id}/sessions/{session_id}/regenerate")
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

    @app.delete("/projects/{project_id}")
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
