"""Admin email — the site owner writing to one of their users.

    POST /admin/email   {to, subject, message, reply_to?}

Guarded by ``X-Admin-Key`` like the rest of ``/admin`` (``require_admin`` in
``router.py``); the AuthGate lets ``/admin/`` through precisely because each route
here checks that key itself. Only phansora.com's Node server holds it, and it calls
this from ``POST /api/admin/users/:id/message``, behind the ADMIN_EMAIL session check.

This is the one route that lets a caller choose the recipient. ``/contact`` is public
and must never be able to, so it keeps its own path to ``send_email`` (fixed to
EMAIL_TO) and this one goes through ``send_message``.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request

from phansora.shared.admin.router import require_admin
from phansora.shared.utils.email import send_message

logger = logging.getLogger("phansora.admin")

router = APIRouter(prefix="/admin", tags=["admin"])

MAX_SUBJECT = 200
MAX_MESSAGE = 8000
MAX_ADDRESS = 254

# One plain address: no display name, no list, nothing that could end a header line.
_ADDRESS = re.compile(r"^[^\s@,;<>\"]+@[^\s@,;<>\"]+\.[^\s@,;<>\"]+$")


def _address(value: object) -> str:
    text = str(value or "").strip()
    return text if len(text) <= MAX_ADDRESS and _ADDRESS.match(text) else ""


@router.post("/email")
async def email_user(
    request: Request,
    x_admin_key: Optional[str] = Header(None, alias="X-Admin-Key"),
) -> dict:
    require_admin(x_admin_key)
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    to = _address(data.get("to"))
    if not to:
        raise HTTPException(status_code=400, detail="A single valid recipient address is required.")
    # Newlines in the subject are the header-injection vector; the body keeps its own.
    subject = " ".join(str(data.get("subject") or "").split())[:MAX_SUBJECT]
    message = str(data.get("message") or "").strip()[:MAX_MESSAGE]
    if not subject:
        raise HTTPException(status_code=400, detail="Subject is required.")
    if not message:
        raise HTTPException(status_code=400, detail="Message is required.")

    try:
        await send_message(to, subject, message, _address(data.get("reply_to")))
    except Exception as e:
        logger.exception("Admin email to %s failed", to)
        raise HTTPException(status_code=502, detail=f"Failed to send email: {e}")

    logger.info("Admin emailed %s: %s", to, subject)
    return {"ok": True}
