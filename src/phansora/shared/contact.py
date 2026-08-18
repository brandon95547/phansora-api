"""Contact endpoint — submit a message from the marketing sites.

This is a cross-cutting concern (not a SpokenVerse feature), so it lives in
``shared/`` alongside its delivery helper (``shared/utils/email.py``) and is
included on the core app in ``main.py`` rather than under any product prefix.

Two callers reach it:

* **phansora.com** — its Node server proxies ``/api/contact`` here and presents
  the internal key, so those requests arrive authenticated.
* **skylanex.com** — a static site with no server of its own. Its nginx vhost
  proxies ``/api/contact`` straight here with no key, so this route is public
  (see ``_PUBLIC_EXACT`` in ``shared/auth``). The recipient is fixed by
  ``EMAIL_TO`` in the environment — a submission can never choose where the mail
  goes — and unauthenticated callers are rate limited per IP below so an open
  form cannot be turned into a flood into that one mailbox.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from typing import Deque, Dict

from fastapi import APIRouter, HTTPException, Request

from phansora.shared.auth import INTERNAL_KEY_HEADER, verify_internal_key
from phansora.shared.utils.email import send_email

logger = logging.getLogger(__name__)

router = APIRouter(tags=["contact"])

# Caps on what a submission may contain. Generous for a real message, small
# enough that a bot cannot mail a novel (or a header-injection payload) through.
MAX_SUBJECT = 200
MAX_MESSAGE = 8000
MAX_REPLY_TO = 254

# Field names no human ever fills: they are hidden in the form, so anything in
# them is a bot that filled every input it found.
HONEYPOT_FIELDS = ("website", "url", "company_website")

# Per-IP sliding window for unauthenticated (public form) submissions.
RATE_LIMIT_MAX = 5
RATE_LIMIT_WINDOW = 3600  # seconds
_hits: Dict[str, Deque[float]] = {}


def _client_ip(request: Request) -> str:
    """Best-effort caller identity for rate limiting.

    Both marketing sites reach this through an nginx proxy on the same host, so
    the socket peer is always loopback and the real caller is in the forwarded
    header. Only trust that header for a loopback peer — from anywhere else it
    is caller-controlled text and would make the limit trivially bypassable.
    """
    peer = (request.client.host if request.client else "") or "unknown"
    if peer in ("127.0.0.1", "::1", "localhost"):
        forwarded = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
        if forwarded:
            return forwarded
    return peer


def _rate_limited(ip: str) -> bool:
    """Record a hit for ``ip`` and report whether it has now exceeded the window."""
    now = time.monotonic()
    window = _hits.setdefault(ip, deque())
    while window and now - window[0] > RATE_LIMIT_WINDOW:
        window.popleft()
    if not window:
        # Drop entries that have fully aged out, so the map cannot grow forever.
        for stale_ip in [k for k, v in _hits.items() if not v and k != ip]:
            _hits.pop(stale_ip, None)
    if len(window) >= RATE_LIMIT_MAX:
        return True
    window.append(now)
    return False


def _clean(value: object, limit: int) -> str:
    """Trim a submitted field to a single-line-safe string of at most ``limit``."""
    text = str(value or "").strip()
    # Newlines in a header field are the header-injection vector; the body keeps its own.
    return text[:limit]


@router.post("/contact")
async def submit_contact(request: Request) -> dict:
    """Accept a contact-form submission and deliver it via email."""
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # A filled honeypot is a bot. Answer exactly like a success so it learns nothing.
    if any(str(data.get(field) or "").strip() for field in HONEYPOT_FIELDS):
        logger.info("Contact submission dropped: honeypot filled")
        return {"status": "Email sent"}

    # The gate marks a PUBLIC path as non-internal without ever looking at the key,
    # so re-check it here: otherwise phansora.com's Node relay — one host funnelling
    # the whole site's submissions — shares a single bucket and locks out after five.
    auth = getattr(request.state, "auth", None) or {}
    is_relay = auth.get("internal") or verify_internal_key(request.headers.get(INTERNAL_KEY_HEADER))
    if not is_relay:
        ip = _client_ip(request)
        if _rate_limited(ip):
            logger.warning("Contact submission rate limited for %s", ip)
            raise HTTPException(
                status_code=429,
                detail="Too many messages from this address. Please try again later.",
            )

    payload = {
        "subject": _clean(data.get("subject"), MAX_SUBJECT).replace("\n", " ").replace("\r", " "),
        "message": _clean(data.get("message"), MAX_MESSAGE),
        "reply_to": _clean(data.get("reply_to"), MAX_REPLY_TO),
    }

    try:
        result = await send_email(payload)
    except ValueError as e:
        # Validation problem with the request payload.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # SMTP / delivery failure — must not report success.
        logger.exception("Contact email delivery failed")
        raise HTTPException(status_code=502, detail=f"Failed to send email: {str(e)}")

    return {"status": result}
