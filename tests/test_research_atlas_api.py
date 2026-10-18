"""The Research Atlas job API (phansora.products.research_atlas.api).

/run answers at once and runs the pipeline in the background, so the things worth
pinning are the ones a background task can get wrong without anyone seeing: the task
must not be collected mid-run, pipelines must not all run at once, one request must not
be able to hold unbounded text, and a failure must not hand the caller our internals.
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed on this host")

from fastapi.testclient import TestClient  # noqa: E402

from phansora.products.research_atlas import api  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_jobs():
    api._JOBS.clear()
    yield
    api._JOBS.clear()


def _wait_for(client, job_id, statuses, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in statuses:
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached {statuses}")


def test_a_failed_job_does_not_return_the_exception_text(monkeypatch):
    def explode(**kwargs):
        raise RuntimeError("DeepSeek HTTP 401: {'error': 'sk-live-secret'} at /var/www/phansora-api/.env")

    monkeypatch.setattr(api, "run_pipeline", explode)
    with TestClient(api.app) as client:
        job_id = client.post("/run", json={"text": "some research"}).json()["job_id"]
        body = _wait_for(client, job_id, {"failed"})
        assert "secret" not in body["error"] and "/var/www" not in body["error"]
        assert "reference" in body["error"]
        result = client.get(f"/jobs/{job_id}/result")
        assert result.status_code == 409 and "secret" not in result.text


def test_a_running_job_is_held_until_it_finishes(monkeypatch):
    def slow(**kwargs):
        time.sleep(0.2)
        return {"ok": True}

    monkeypatch.setattr(api, "run_pipeline", slow)
    with TestClient(api.app) as client:
        job_id = client.post("/run", json={"text": "some research"}).json()["job_id"]
        assert len(api._TASKS) == 1
        _wait_for(client, job_id, {"completed"})
        deadline = time.monotonic() + 2
        while api._TASKS and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not api._TASKS, "a finished task is still referenced"


def test_pipelines_past_the_cap_wait_their_turn(monkeypatch):
    release = threading.Event()
    started = []

    def held(**kwargs):
        started.append(kwargs)
        release.wait(5)
        return {"ok": True}

    monkeypatch.setattr(api, "run_pipeline", held)
    monkeypatch.setattr(api, "_RUN_SLOTS", asyncio.Semaphore(1))
    with TestClient(api.app) as client:
        first = client.post("/run", json={"text": "one"}).json()["job_id"]
        _wait_for(client, first, {"processing"})
        second = client.post("/run", json={"text": "two"}).json()["job_id"]
        time.sleep(0.1)
        assert client.get(f"/jobs/{second}").json()["status"] == "queued"
        assert len(started) == 1
        release.set()
        _wait_for(client, second, {"completed"})
        assert len(started) == 2


def test_too_many_waiting_jobs_is_a_busy_answer(monkeypatch):
    monkeypatch.setattr(api, "_MAX_OUTSTANDING", 0)
    with TestClient(api.app) as client:
        res = client.post("/run", json={"text": "one"})
        assert res.status_code == 503
        assert res.headers.get("retry-after") == "60"


@pytest.mark.parametrize("payload", [
    {"text": "x" * (api.MAX_TEXT_CHARS + 1)},
    {"sources": [{"label": "a", "text": "x"}] * (api.MAX_SOURCES + 1)},
    {"sources": [{"label": "a" * (api.MAX_LABEL_CHARS + 1), "text": "x"}]},
])
def test_oversized_requests_are_refused(payload, monkeypatch):
    monkeypatch.setattr(api, "run_pipeline", lambda **kw: pytest.fail("ran an oversized request"))
    with TestClient(api.app) as client:
        assert client.post("/run", json=payload).status_code == 422


def test_the_total_across_sources_is_capped(monkeypatch):
    monkeypatch.setattr(api, "MAX_TOTAL_CHARS", 10)
    with pytest.raises(ValueError):
        api.RunPipelineRequest(sources=[{"label": "a", "text": "x" * 6}, {"label": "b", "text": "y" * 6}])
    assert api.RunPipelineRequest(sources=[{"label": "a", "text": "x" * 5}])
