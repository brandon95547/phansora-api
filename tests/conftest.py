"""Settings shared by the whole suite."""
from __future__ import annotations

import os
import shutil
import tempfile

import pytest

# Runtime data goes to a scratch directory, never into the checkout. Set here, at import,
# because several products resolve their data directories when they are imported
# (Research Atlas makes its tmp/ folder then), which is before any fixture runs. Set
# outright rather than defaulted, so a PHANSORA_DATA_DIR in the shell or a .env cannot
# point the suite at real data.
_DATA_DIR = tempfile.mkdtemp(prefix="phansora-tests-")
os.environ["PHANSORA_DATA_DIR"] = _DATA_DIR


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_DATA_DIR, ignore_errors=True)


@pytest.fixture(autouse=True)
def _no_gate_in_front(monkeypatch):
    """Product apps are tested on their own, with no AuthGate in front of them — the way
    local development runs them — so the suite runs as local development does.

    Without this, ``enforce_user_scope`` refuses every request (no gate means no verdict,
    and no verdict is refused). The tests that exercise the gate itself unset it.
    """
    monkeypatch.setenv("PHANSORA_AUTH_DISABLED", "1")
