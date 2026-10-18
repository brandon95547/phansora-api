"""Shared conversion endpoints — turn a canonical timeline ``document`` into any
supported download format. Product-agnostic; included on the core app in
``phansora.main`` so any product can reuse it:

    GET  /convert/formats   -> list of {id, label, ext, mime}
    POST /convert           -> {format, document, filename?} -> file download
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from phansora.shared.conversions.converters import convert, list_formats
from phansora.shared.errors import fail

logger = logging.getLogger("phansora.conversions")

router = APIRouter(tags=["conversions"])


class ConvertRequest(BaseModel):
    format: str = Field(..., min_length=1)
    document: Dict[str, Any] = Field(default_factory=dict)
    filename: Optional[str] = None


def _safe_name(name: str, ext: str) -> str:
    base = "".join(ch.lower() if ch.isalnum() else "-" for ch in (name or "timeline"))
    base = "-".join(p for p in base.split("-") if p)[:80] or "timeline"
    return f"{base}.{ext}"


@router.get("/convert/formats")
def formats() -> dict:
    return {"formats": list_formats()}


@router.post("/convert")
def do_convert(req: ConvertRequest) -> Response:
    try:
        content, media, ext = convert(req.format, req.document)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {req.format}") from None
    except Exception as e:  # noqa: BLE001 — a document the converter cannot handle
        raise fail(422, "The timeline could not be converted.", e, logger=logger,
                   context=f"Conversion to {req.format} failed") from e
    title = req.document.get("title") if isinstance(req.document, dict) else None
    name = _safe_name(req.filename or title or "timeline", ext)
    return Response(
        content=content.encode("utf-8"),
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )
