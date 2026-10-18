"""Settings shared by the whole suite."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_gate_in_front(monkeypatch):
    """Product apps are tested on their own, with no AuthGate in front of them — the way
    local development runs them — so the suite runs as local development does.

    Without this, ``enforce_user_scope`` refuses every request (no gate means no verdict,
    and no verdict is refused). The tests that exercise the gate itself unset it.
    """
    monkeypatch.setenv("PHANSORA_AUTH_DISABLED", "1")
