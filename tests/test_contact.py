"""The contact endpoint (phansora.shared.contact).

The point of interest is that this route is PUBLIC — skylanex.com is a static
site whose nginx vhost proxies straight here with no key — so the tests pin the
things that keep an open mail endpoint safe: the auth gate lets it through, the
recipient is never caller-chosen, bots are absorbed, and an unauthenticated
flood is cut off while the keyed Node relay is not.
"""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed on this host")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from phansora.shared import contact as contact_mod  # noqa: E402
from phansora.shared.auth import AuthGate  # noqa: E402

KEY = "test-key-123"


@pytest.fixture(autouse=True)
def _keyed_env(monkeypatch):
    monkeypatch.setenv("PHANSORA_INTERNAL_KEY", KEY)
    monkeypatch.delenv("PHANSORA_AUTH_DISABLED", raising=False)


@pytest.fixture()
def sent(monkeypatch):
    """Capture what would have gone to SMTP instead of sending it."""
    calls = []

    async def fake_send_email(data):
        calls.append(data)
        return "Email sent"

    monkeypatch.setattr(contact_mod, "send_email", fake_send_email)
    return calls


@pytest.fixture(autouse=True)
def _fresh_limiter():
    contact_mod._hits.clear()
    yield
    contact_mod._hits.clear()


def _from_peer(app, host):
    """``app`` as a socket peer at ``host`` would reach it.

    TestClient's own ``client=`` argument arrived after the Starlette this repo pins,
    so the peer is set on the ASGI scope instead.
    """

    async def asgi(scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            scope = {**scope, "client": (host, 54321)}
        await app(scope, receive, send)

    return asgi


def _app():
    app = FastAPI()
    app.add_middleware(AuthGate)
    app.include_router(contact_mod.router)
    return app


@pytest.fixture()
def client():
    # Pose as nginx on the same host: that is the only shape these requests ever
    # arrive in, and it is what makes the forwarded-header handling live code.
    return TestClient(_from_peer(_app(), "127.0.0.1"), raise_server_exceptions=False)


def _body(**over):
    payload = {"subject": "Skylanex inquiry", "message": "Hello there", "reply_to": "a@b.com"}
    payload.update(over)
    return payload


def test_public_no_key_needed(client, sent):
    """The static site posts with no credentials at all — that must reach SMTP."""
    res = client.post("/contact", json=_body())
    assert res.status_code == 200
    assert len(sent) == 1
    assert sent[0]["message"] == "Hello there"


def test_missing_message_is_rejected(client, monkeypatch):
    async def raiser(data):
        raise ValueError("Missing 'message'")

    monkeypatch.setattr(contact_mod, "send_email", raiser)
    res = client.post("/contact", json=_body(message=""))
    assert res.status_code == 400


def test_smtp_failure_is_not_reported_as_success(client, monkeypatch):
    async def boom(data):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(contact_mod, "send_email", boom)
    res = client.post("/contact", json=_body())
    assert res.status_code == 502


def test_honeypot_is_dropped_but_looks_like_success(client, sent):
    res = client.post("/contact", json=_body(website="http://spam.example"))
    assert res.status_code == 200
    assert res.json()["status"] == "Email sent"
    assert sent == []  # nothing actually mailed


def test_oversize_fields_are_truncated(client, sent):
    client.post("/contact", json=_body(message="x" * 50_000, subject="y" * 5_000))
    assert len(sent[0]["message"]) == contact_mod.MAX_MESSAGE
    assert len(sent[0]["subject"]) == contact_mod.MAX_SUBJECT


def test_newlines_cannot_be_smuggled_into_the_subject(client, sent):
    client.post("/contact", json=_body(subject="Hi\nBcc: victim@example.com"))
    assert "\n" not in sent[0]["subject"]
    assert "\r" not in sent[0]["subject"]


def test_unauthenticated_flood_is_cut_off(client, sent):
    for _ in range(contact_mod.RATE_LIMIT_MAX):
        assert client.post("/contact", json=_body()).status_code == 200
    res = client.post("/contact", json=_body())
    assert res.status_code == 429
    assert len(sent) == contact_mod.RATE_LIMIT_MAX


def test_forwarded_ips_are_limited_separately(client, sent):
    """Every skylanex submission shares one socket peer (nginx on loopback), so
    the limit has to key on the forwarded client or one visitor would lock out all."""
    for _ in range(contact_mod.RATE_LIMIT_MAX):
        client.post("/contact", json=_body(), headers={"X-Forwarded-For": "203.0.113.7"})
    blocked = client.post("/contact", json=_body(), headers={"X-Forwarded-For": "203.0.113.7"})
    other = client.post("/contact", json=_body(), headers={"X-Forwarded-For": "198.51.100.9"})
    assert blocked.status_code == 429
    assert other.status_code == 200


def test_a_forged_forwarded_entry_does_not_dodge_the_limit(client, sent):
    """nginx APPENDS the address it saw to whatever X-Forwarded-For the caller sent, so
    only the last entry is nginx's. A bot that sends a fresh first entry every time is
    still one address."""
    for n in range(contact_mod.RATE_LIMIT_MAX):
        forged = {"X-Forwarded-For": f"10.0.0.{n}, 203.0.113.7"}
        assert client.post("/contact", json=_body(), headers=forged).status_code == 200
    forged = {"X-Forwarded-For": "10.0.0.99, 203.0.113.7"}
    assert client.post("/contact", json=_body(), headers=forged).status_code == 429


def test_the_forwarded_header_is_ignored_from_anywhere_but_loopback(sent):
    """Off loopback the header is the caller's own text; the socket peer is the identity."""
    remote = TestClient(_from_peer(_app(), "198.51.100.20"), raise_server_exceptions=False)
    for n in range(contact_mod.RATE_LIMIT_MAX):
        remote.post("/contact", json=_body(), headers={"X-Forwarded-For": f"10.0.0.{n}"})
    res = remote.post("/contact", json=_body(), headers={"X-Forwarded-For": "10.0.0.99"})
    assert res.status_code == 429


def test_a_delivery_failure_does_not_echo_the_smtp_error(client, monkeypatch):
    async def boom(data):
        raise RuntimeError("530 relay.internal.example refused: auth secret wrong")

    monkeypatch.setattr(contact_mod, "send_email", boom)
    res = client.post("/contact", json=_body())
    assert res.status_code == 502
    assert "relay.internal" not in res.text and "secret" not in res.text


def test_keyed_relay_is_exempt_from_the_limit(client, sent):
    """phansora.com funnels every submission through one server; limiting that
    shared identity would throttle the whole site after five messages."""
    headers = {"x-phansora-internal-key": KEY}
    for _ in range(contact_mod.RATE_LIMIT_MAX + 3):
        assert client.post("/contact", json=_body(), headers=headers).status_code == 200
    assert len(sent) == contact_mod.RATE_LIMIT_MAX + 3
