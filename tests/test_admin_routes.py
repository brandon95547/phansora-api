"""Every /admin route needs the admin key (phansora.shared.admin).

The AuthGate lets every /admin path through without credentials, because the admin
routers check X-Admin-Key themselves. So the check has to hold for every route those
routers will ever have, not only the ones that remembered to call it.
"""
from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed on this host")

from fastapi import FastAPI  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from phansora.shared.admin import email as email_mod  # noqa: E402
from phansora.shared.admin import router as router_mod  # noqa: E402
from phansora.shared.auth import AuthGate  # noqa: E402

ADMIN_KEY = "admin-key-456"


@pytest.fixture(autouse=True)
def _keyed_env(monkeypatch):
    monkeypatch.setenv("PHANSORA_ADMIN_KEY", ADMIN_KEY)
    monkeypatch.setenv("PHANSORA_INTERNAL_KEY", "internal-key-123")
    monkeypatch.delenv("PHANSORA_AUTH_DISABLED", raising=False)


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(AuthGate)
    app.include_router(router_mod.router)
    app.include_router(email_mod.router)
    return app


def _admin_routes(app: FastAPI):
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path.startswith("/admin"):
            for method in route.methods:
                yield method, route.path


def test_there_are_admin_routes_to_check():
    assert len(list(_admin_routes(_app()))) >= 4


@pytest.mark.parametrize("key", [None, "wrong", "internal-key-123"])
def test_every_admin_route_refuses_without_the_admin_key(key):
    app = _app()
    client = TestClient(app, raise_server_exceptions=False)
    headers = {"X-Admin-Key": key} if key else {}
    for method, path in _admin_routes(app):
        res = client.request(method, path, headers=headers, json={})
        assert res.status_code == 403, f"{method} {path} answered {res.status_code} without the admin key"


def test_the_storage_report_answers_with_the_key(monkeypatch, tmp_path):
    monkeypatch.setattr(router_mod, "_dir_size", lambda p: 0)
    client = TestClient(_app(), raise_server_exceptions=False)
    res = client.get("/admin/storage/info", headers={"X-Admin-Key": ADMIN_KEY})
    assert res.status_code == 200
    assert res.json()["ok"] is True
