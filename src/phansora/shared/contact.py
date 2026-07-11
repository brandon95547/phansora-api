"""Contact endpoint — submit a message from the marketing site.

This is a cross-cutting concern (not a SpokenVerse feature), so it lives in
``shared/`` alongside its delivery helper (``shared/utils/email.py``) and is
included on the core app in ``main.py`` rather than under any product prefix.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from phansora.shared.utils.email import send_email

router = APIRouter(tags=["contact"])


@router.post("/contact")
async def submit_contact(request: Request) -> dict:
    """Accept a contact-form submission and deliver it via email."""
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
