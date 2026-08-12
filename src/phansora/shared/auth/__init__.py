"""Who is allowed to talk to this API, decided in one place.

Two kinds of caller, two credentials:

  * The Node app, server-to-server. It has already authenticated the browser
    session and meters credits, so it vouches for whatever ``user_id`` it sends.
    It proves itself with the shared secret in the ``X-Phansora-Internal-Key``
    header (``PHANSORA_INTERNAL_KEY`` in both apps' environments).

  * A browser, directly. Some dashboards call the API from the page (uploads
    and audio would otherwise stream through Node twice). The browser can hold
    no secret, so Node mints it a short-lived token binding the SESSION's user
    id: ``v1.<uid>.<exp>.<sig>``, HMAC-SHA256 over the first three fields with
    the same shared key. The token authenticates the request and pins which
    ``user_id`` it may act as — ``enforce_user_scope`` below rejects any
    request whose user_id (path, query or form) disagrees with its token.

Before this module, there was no authentication at all: every endpoint trusted
a client-supplied ``user_id`` form field, so any internet client could read or
delete any user's data by enumerating small integers. That is the failure this
exists to close, which is why the gate FAILS CLOSED: no configured key means
503 for everything but the public paths, not open season. Set
``PHANSORA_AUTH_DISABLED=1`` to run without auth in local development only.

Tokens ride in ``Authorization: Bearer <token>`` normally, or in a ``pt=``
query parameter for the URLs that end up in ``<audio src>`` — media elements
cannot send custom headers.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
import time

from fastapi import HTTPException, Request
from starlette.responses import JSONResponse

logger = logging.getLogger("phansora.auth")

INTERNAL_KEY_HEADER = "x-phansora-internal-key"
TOKEN_QUERY_PARAM = "pt"
TOKEN_TTL_SECONDS = 12 * 3600  # page loads mint a fresh one; long enough for a work session

# The uid charset mirrors SpokenVerse's _safe_user_id: ids are opaque strings here,
# numeric in practice. Bounded so a token can never smuggle something path-shaped.
_UID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# Paths that stay reachable without credentials, by design rather than omission:
#   /            service name + mounted products; what uptime tooling probes
#   /health      the load balancer's question, must never depend on a secret
#   /admin       carries its own key check (shared/admin/router.py); one gate, not two
#
# /contact and /convert are NOT here although they serve public-site features: the
# browser posts them to Node, and Node relays with the internal key. Verified against
# every caller (this repo, the Node app, the static sites) before tightening.
_PUBLIC_EXACT = {"/", "/health", "/admin"}
_PUBLIC_PREFIXES = ("/admin/",)

# Interactive docs enumerate every route and schema — exactly the map an attacker
# wants and nothing a production caller needs. Authenticated callers still get them.
_DOCS_SUFFIXES = ("/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect")


def _key() -> str | None:
    return (os.getenv("PHANSORA_INTERNAL_KEY") or "").strip() or None


def _disabled() -> bool:
    return (os.getenv("PHANSORA_AUTH_DISABLED") or "").strip() == "1"


def verify_internal_key(presented: str | None) -> bool:
    key = _key()
    if not key or not presented:
        return False
    return hmac.compare_digest(presented.strip(), key)


def _sign(payload: str, key: str) -> str:
    return hmac.new(key.encode(), payload.encode(), hashlib.sha256).hexdigest()


def mint_user_token(user_id: str | int, ttl_seconds: int = TOKEN_TTL_SECONDS, *, now: float | None = None) -> str:
    """Node's job in production (src/utils/api-auth.js) — here for tests and tooling.

    The two implementations must agree byte-for-byte on the signed payload.
    """
    key = _key()
    if not key:
        raise RuntimeError("PHANSORA_INTERNAL_KEY is not configured")
    uid = str(user_id)
    if not _UID_RE.match(uid):
        raise ValueError("user id is not token-safe")
    exp = int((now if now is not None else time.time()) + ttl_seconds)
    payload = f"v1.{uid}.{exp}"
    return f"{payload}.{_sign(payload, key)}"


def verify_user_token(token: str | None, *, now: float | None = None) -> str | None:
    """The user id a token vouches for, or None for anything else.

    None rather than an exception: the middleware treats a bad token exactly like
    a missing one, so probing with mangled tokens learns nothing an anonymous
    request would not.
    """
    key = _key()
    if not key or not token:
        return None
    parts = token.strip().split(".")
    if len(parts) != 4 or parts[0] != "v1":
        return None
    _, uid, exp_s, sig = parts
    if not _UID_RE.match(uid):
        return None
    payload = f"v1.{uid}.{exp_s}"
    if not hmac.compare_digest(sig, _sign(payload, key)):
        return None
    try:
        if int(exp_s) < (now if now is not None else time.time()):
            return None
    except ValueError:
        return None
    return uid


class AuthGate:
    """Pure ASGI gate on the parent app, so every mounted product is behind it.

    Sets ``scope["state"]["auth"]`` to ``{"internal": True}`` for the Node app or
    ``{"internal": False, "user_id": uid}`` for a browser token; sub-app request
    objects read it as ``request.state.auth``. Kept as raw ASGI rather than
    BaseHTTPMiddleware so streamed uploads pass through untouched.
    """

    def __init__(self, app):
        self.app = app
        # One loud line per boot, not one per request.
        if _disabled():
            logger.critical("PHANSORA_AUTH_DISABLED=1 — the API is running WITHOUT authentication")
        elif not _key():
            logger.critical(
                "PHANSORA_INTERNAL_KEY is not set: every non-public route will answer 503 "
                "until it is configured (or PHANSORA_AUTH_DISABLED=1 for local dev)"
            )

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)

        path = scope.get("path") or "/"
        method = scope.get("method", "GET")

        if _disabled():
            scope.setdefault("state", {})["auth"] = {"internal": True}
            return await self.app(scope, receive, send)

        # Preflight carries no credentials by spec; letting it through leaks nothing
        # (the outer CORSMiddleware usually answers it before this runs anyway).
        if method == "OPTIONS":
            return await self.app(scope, receive, send)

        is_public = path in _PUBLIC_EXACT or path.startswith(_PUBLIC_PREFIXES)
        if is_public and not path.endswith(_DOCS_SUFFIXES):
            scope.setdefault("state", {})["auth"] = {"internal": False, "user_id": None}
            return await self.app(scope, receive, send)

        if not _key():
            response = JSONResponse(
                {"detail": "API authentication is not configured on this server."}, status_code=503
            )
            return await response(scope, receive, send)

        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}

        if verify_internal_key(headers.get(INTERNAL_KEY_HEADER)):
            scope.setdefault("state", {})["auth"] = {"internal": True}
            return await self.app(scope, receive, send)

        token = None
        authz = headers.get("authorization", "")
        if authz.lower().startswith("bearer "):
            token = authz[7:]
        if not token:
            # <audio>/<a download> URLs cannot carry headers; they carry ?pt= instead.
            from urllib.parse import parse_qs

            token = (parse_qs(scope.get("query_string", b"").decode("latin-1")).get(TOKEN_QUERY_PARAM) or [None])[0]

        uid = verify_user_token(token)
        if uid is not None:
            scope.setdefault("state", {})["auth"] = {"internal": False, "user_id": uid}
            return await self.app(scope, receive, send)

        response = JSONResponse({"detail": "Not authenticated."}, status_code=401)
        return await response(scope, receive, send)


_FORM_CONTENT_TYPES = ("multipart/form-data", "application/x-www-form-urlencoded")


async def enforce_user_scope(request: Request) -> None:
    """Product-level dependency: a browser token may only act as its own user.

    The gate above answered WHO is calling; this answers whether they may touch
    the ``user_id`` the request names. Internal callers pass — Node already
    authenticated the session it is relaying. Token callers must match on every
    place a user id can travel: path params, query string, and form fields.

    Reading the form here is safe and free: Starlette caches the parsed form on
    the request, so handlers that declare ``Form(...)``/``UploadFile`` params
    get the same parse, not a second one.
    """
    auth = getattr(request.state, "auth", None)
    if auth is None or auth.get("internal"):
        return
    uid = auth.get("user_id")
    if uid is None:
        # A public-path request that fell through to a product route; the gate
        # never vouched for anyone, so a user-scoped route cannot serve it.
        raise HTTPException(status_code=401, detail="Not authenticated.")

    def _check(value) -> None:
        if value is None:
            return
        presented = str(value).strip()
        if presented and presented != uid:
            raise HTTPException(status_code=403, detail="That user id does not match your session.")

    _check(request.path_params.get("user_id"))
    _check(request.query_params.get("user_id"))

    content_type = (request.headers.get("content-type") or "").lower()
    if request.method in ("POST", "PUT", "PATCH", "DELETE") and content_type.startswith(_FORM_CONTENT_TYPES):
        try:
            form = await request.form()
        except Exception:  # noqa: BLE001 — a malformed body is the handler's error to report
            return
        _check(form.get("user_id"))
