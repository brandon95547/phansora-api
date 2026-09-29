"""The admin email endpoint (phansora.shared.admin.email).

Unlike /contact, this route lets the caller choose the recipient, so the tests pin
what keeps that from being a relay: nothing is sent without the admin key, the
recipient must be exactly one plain address, and the header fields cannot carry a
line break.
"""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed on this host")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from phansora.shared.admin import email as email_mod  # noqa: E402
from phansora.shared.auth import AuthGate  # noqa: E402

ADMIN_KEY = "admin-key-456"


@pytest.fixture(autouse=True)
def _keyed_env(monkeypatch):
    monkeypatch.setenv("PHANSORA_ADMIN_KEY", ADMIN_KEY)
    monkeypatch.setenv("PHANSORA_INTERNAL_KEY", "internal-key-123")
    monkeypatch.delenv("PHANSORA_AUTH_DISABLED", raising=False)


@pytest.fixture()
def sent(monkeypatch):
    """Capture what would have gone to SMTP instead of sending it."""
    calls = []

    async def fake_send_message(to, subject, message, reply_to=""):
        calls.append({"to": to, "subject": subject, "message": message, "reply_to": reply_to})
        return "Email sent"

    monkeypatch.setattr(email_mod, "send_message", fake_send_message)
    return calls


@pytest.fixture()
def client():
    app = FastAPI()
    app.add_middleware(AuthGate)
    app.include_router(email_mod.router)
    return TestClient(app, raise_server_exceptions=False)


def _body(**over):
    payload = {
        "to": "ada@example.com",
        "subject": "About your book",
        "message": "Hi Ada,\n\nIt finished overnight.",
        "reply_to": "owner@example.com",
    }
    payload.update(over)
    return payload


def _post(client, key=ADMIN_KEY, **over):
    headers = {"X-Admin-Key": key} if key else {}
    return client.post("/admin/email", json=_body(**over), headers=headers)


def test_sends_to_the_chosen_user(client, sent):
    res = _post(client)
    assert res.status_code == 200
    assert res.json() == {"ok": True}
    assert sent == [{
        "to": "ada@example.com",
        "subject": "About your book",
        "message": "Hi Ada,\n\nIt finished overnight.",
        "reply_to": "owner@example.com",
    }]


@pytest.mark.parametrize("key", [None, "wrong", "internal-key-123"])
def test_nothing_is_sent_without_the_admin_key(client, sent, key):
    res = _post(client, key=key)
    assert res.status_code == 403
    assert sent == []


def test_locked_when_no_admin_key_is_configured(client, sent, monkeypatch):
    monkeypatch.delenv("PHANSORA_ADMIN_KEY", raising=False)
    assert _post(client).status_code == 403
    assert sent == []


@pytest.mark.parametrize("to", [
    "",
    "not-an-address",
    "a@example.com, b@example.com",
    "a@example.com\nBcc: b@example.com",
    "Ada <ada@example.com>",
])
def test_recipient_must_be_one_plain_address(client, sent, to):
    assert _post(client, to=to).status_code == 400
    assert sent == []


def test_subject_cannot_carry_a_header(client, sent):
    assert _post(client, subject="Hello\r\nBcc: someone@example.com").status_code == 200
    assert "\n" not in sent[0]["subject"] and "\r" not in sent[0]["subject"]


def test_subject_and_message_are_required(client, sent):
    assert _post(client, subject="  ").status_code == 400
    assert _post(client, message="").status_code == 400
    assert sent == []


def test_a_bad_reply_to_is_dropped_not_fatal(client, sent):
    assert _post(client, reply_to="nope\nBcc: x@y.z").status_code == 200
    assert sent[0]["reply_to"] == ""


def test_delivery_failure_is_not_reported_as_success(client, monkeypatch):
    async def boom(*args, **kwargs):
        raise OSError("SMTP down")

    monkeypatch.setattr(email_mod, "send_message", boom)
    assert _post(client).status_code == 502
