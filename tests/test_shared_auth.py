"""The auth gate and user-scope enforcement (phansora.shared.auth).

The topology under test is a miniature of main.py's: products are MOUNTED
sub-apps behind one AuthGate on the parent, and each product app carries the
enforce_user_scope dependency. That mount indirection is the part worth
pinning — the gate writes its verdict into the ASGI scope, and the sub-app's
request object has to read the same scope for any of this to work.
"""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed on this host")

from fastapi import Depends, FastAPI, Form  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from phansora.shared.auth import (  # noqa: E402
    AuthGate,
    enforce_user_scope,
    mint_user_token,
    verify_user_token,
)

KEY = "test-key-123"


@pytest.fixture(autouse=True)
def _keyed_env(monkeypatch):
    monkeypatch.setenv("PHANSORA_INTERNAL_KEY", KEY)
    monkeypatch.delenv("PHANSORA_AUTH_DISABLED", raising=False)


@pytest.fixture()
def client():
    product = FastAPI(dependencies=[Depends(enforce_user_scope)])

    @product.get("/things")
    def list_things(user_id: str = ""):
        return {"user_id": user_id}

    @product.post("/make")
    def make_thing(user_id: str = Form(...)):
        return {"user_id": user_id}

    @product.get("/users/{user_id}/history")
    def history(user_id: str):
        return {"user_id": user_id}

    parent = FastAPI()
    parent.add_middleware(AuthGate)

    @parent.get("/health")
    def health():
        return {"ok": True}

    parent.mount("/spokenverse", product)
    return TestClient(parent, raise_server_exceptions=False)


def bearer(uid=42, **kw):
    return {"Authorization": f"Bearer {mint_user_token(uid, **kw)}"}


INTERNAL = {"X-Phansora-Internal-Key": KEY}


# ── token primitives ────────────────────────────────────────────────────────

def test_mint_verify_roundtrip():
    assert verify_user_token(mint_user_token(42)) == "42"


def test_expired_token_is_rejected():
    assert verify_user_token(mint_user_token(42, ttl_seconds=-5)) is None


def test_tampering_is_rejected():
    tok = mint_user_token(42)
    flipped = tok[:-1] + ("0" if tok[-1] != "0" else "1")
    assert verify_user_token(flipped) is None
    # Re-owning a valid signature by swapping the uid must fail too.
    assert verify_user_token(tok.replace(".42.", ".43.", 1)) is None


def test_garbage_is_rejected_not_raised():
    assert verify_user_token("v1.a.b") is None
    assert verify_user_token(None) is None
    assert verify_user_token("") is None


# ── the gate ────────────────────────────────────────────────────────────────

def test_health_is_public(client):
    assert client.get("/health").status_code == 200


def test_products_reject_anonymous_and_wrong_key(client):
    assert client.get("/spokenverse/things").status_code == 401
    assert client.get("/spokenverse/things", headers={"X-Phansora-Internal-Key": "nope"}).status_code == 401


def test_internal_key_may_act_as_anyone(client):
    # Node authenticated the session itself; it vouches for the user_id it relays.
    assert client.get("/spokenverse/things?user_id=7", headers=INTERNAL).status_code == 200


def test_token_is_pinned_to_its_user(client):
    assert client.get("/spokenverse/things?user_id=42", headers=bearer()).status_code == 200
    assert client.get("/spokenverse/things?user_id=7", headers=bearer()).status_code == 403
    assert client.get("/spokenverse/users/42/history", headers=bearer()).status_code == 200
    assert client.get("/spokenverse/users/7/history", headers=bearer()).status_code == 403
    assert client.post("/spokenverse/make", data={"user_id": "42"}, headers=bearer()).status_code == 200
    assert client.post("/spokenverse/make", data={"user_id": "9"}, headers=bearer()).status_code == 403


def test_query_param_token_for_media_elements(client):
    # <audio src> cannot send headers; the token rides as ?pt= there.
    tok = mint_user_token(42)
    assert client.get(f"/spokenverse/things?pt={tok}").status_code == 200


def test_docs_require_auth(client):
    assert client.get("/docs").status_code == 401
    assert client.get("/spokenverse/docs").status_code == 401
    assert client.get("/docs", headers=INTERNAL).status_code == 200


# ── failure posture ─────────────────────────────────────────────────────────

def test_missing_key_fails_closed(client, monkeypatch):
    monkeypatch.setenv("PHANSORA_INTERNAL_KEY", "")
    assert client.get("/spokenverse/things").status_code == 503
    assert client.get("/health").status_code == 200


def test_dev_escape_hatch(client, monkeypatch):
    monkeypatch.setenv("PHANSORA_INTERNAL_KEY", "")
    monkeypatch.setenv("PHANSORA_AUTH_DISABLED", "1")
    assert client.get("/spokenverse/things").status_code == 200
