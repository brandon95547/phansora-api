"""Talari's illustration engine, against a fake ComfyUI — no GPU and no network.

Carried over from the standalone Nebiru server's tests when its ComfyUI client moved into
this API, plus what is new here: pictures belong to a user, and a browser token can only
reach its own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from phansora.products.talari import comfy as comfy_module
from phansora.products.talari.comfy import dimensions, fill_workflow
from phansora.products.talari.config import Settings
from phansora.products.talari.server import create_app
from phansora.products.talari.styles import build_prompts

COMFY = "http://comfy.test"
PNG = b"\x89PNG\r\n\x1a\n fake image bytes"
REQUEST = {"prompt": "A lighthouse at dusk.", "styleId": "ink", "aspect": 0.8, "purpose": "interior", "seed": 42}


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = dict(
        comfyui_url=COMFY,
        comfyui_checkpoint="",
        comfyui_workflow="",
        comfyui_base_size=1024,
        comfyui_steps=20,
        comfyui_cfg=7.0,
        comfyui_sampler="euler",
        comfyui_scheduler="normal",
        comfyui_timeout_s=5.0,
        data_dir=tmp_path / "talari",
        comfyui_pipeline="checkpoint",
    )
    values.update(overrides)
    return Settings(**values)


class Recorder:
    """A fake network: routes requests to a handler and remembers what was sent."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.handler = lambda _request: httpx.Response(404)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)

    def bodies(self, path_part: str) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests if path_part in r.url.path and r.content]


def comfy(history: dict | None = None, checkpoints: list[str] | None = None, queue_status: int = 200, object_info: dict | None = None):
    """A fake ComfyUI that finishes a job on the second history poll."""
    polls = {"count": 0}
    names = ["sdxl_base.safetensors", "other.safetensors"] if checkpoints is None else checkpoints

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/system_stats":
            return httpx.Response(200, json={"system": {}})
        if path == "/object_info/CheckpointLoaderSimple":
            return httpx.Response(200, json={"CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [names, {}]}}}})
        if path.startswith("/object_info/") and object_info is not None:
            node = path.rsplit("/", 1)[1]
            return httpx.Response(200, json={node: object_info[node]} if node in object_info else {})
        if path == "/prompt":
            if queue_status != 200:
                return httpx.Response(queue_status, json={"error": {"message": "Prompt outputs failed validation"}})
            return httpx.Response(200, json={"prompt_id": "job-1"})
        if path == "/history/job-1":
            polls["count"] += 1
            if polls["count"] < 2:
                return httpx.Response(200, json={})
            done = history or {
                "status": {"status_str": "success", "completed": True},
                "outputs": {"7": {"images": [{"filename": "talari_0001.png", "subfolder": "", "type": "output"}]}},
            }
            return httpx.Response(200, json={"job-1": done})
        if path == "/view":
            return httpx.Response(200, content=PNG)
        return httpx.Response(404)

    return handler


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch) -> None:
    monkeypatch.setattr(comfy_module, "POLL_INTERVAL_S", 0)


@pytest.fixture
def network() -> Recorder:
    return Recorder()


@pytest.fixture
def make_client(tmp_path: Path, network: Recorder):
    clients: list[TestClient] = []

    def build(auth: dict | None = None, **overrides: Any) -> TestClient:
        app = create_app(make_settings(tmp_path, **overrides), transport=httpx.MockTransport(network))
        if auth is not None:
            # What the parent app's AuthGate would have put on the request.
            @app.middleware("http")
            async def as_caller(request, call_next):
                request.scope.setdefault("state", {})["auth"] = auth
                return await call_next(request)
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield build
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()


def test_a_picture_is_drawn_stored_for_its_user_and_served_back(client: TestClient, network: Recorder, tmp_path: Path) -> None:
    network.handler = comfy()
    response = client.post("/users/7/illustrations", json=REQUEST)
    assert response.status_code == 200
    body = response.json()
    assert body["seed"] == 42
    assert body["src"].startswith("/talari/users/7/images/") and body["src"].endswith(".png")
    name = body["src"].rsplit("/", 1)[1]
    assert (tmp_path / "talari" / "images" / "7" / name).read_bytes() == PNG
    assert client.get(f"/users/7/images/{name}").content == PNG

    workflow = network.bodies("/prompt")[0]["prompt"]
    assert workflow["1"]["inputs"]["ckpt_name"] == "sdxl_base.safetensors"
    assert workflow["2"]["inputs"]["text"].startswith("A lighthouse at dusk, pen and ink illustration")
    assert "lettering" in workflow["3"]["inputs"]["text"]
    assert (workflow["4"]["inputs"]["width"], workflow["4"]["inputs"]["height"]) == (896, 1152)
    assert workflow["5"]["inputs"]["seed"] == 42 and workflow["5"]["inputs"]["steps"] == 20


def test_a_seed_is_chosen_when_none_is_given(client: TestClient, network: Recorder) -> None:
    network.handler = comfy()
    response = client.post("/users/7/illustrations", json={**REQUEST, "seed": None})
    assert response.status_code == 200
    assert isinstance(response.json()["seed"], int)


def test_flux_draws_with_the_configured_files(make_client, network: Recorder) -> None:
    client = make_client(comfyui_pipeline="flux", flux_unet="flux1-dev-Q4_K_S.gguf")
    network.handler = comfy()
    assert client.post("/users/7/illustrations", json=REQUEST).status_code == 200
    workflow = json.dumps(network.bodies("/prompt")[0]["prompt"])
    for name in ("flux1-dev-Q4_K_S.gguf", "t5xxl_fp8_e4m3fn.safetensors", "clip_l.safetensors", "ae.safetensors"):
        assert name in workflow


def test_a_failed_job_reports_comfys_own_error(client: TestClient, network: Recorder) -> None:
    network.handler = comfy(history={
        "status": {"status_str": "error", "completed": False, "messages": [
            ["execution_error", {"node_type": "KSampler", "exception_message": "CUDA out of memory"}],
        ]},
        "outputs": {},
    })
    response = client.post("/users/7/illustrations", json=REQUEST)
    assert response.status_code == 502
    assert "out of memory" in response.json()["detail"]


def test_a_rejected_workflow_reports_why(client: TestClient, network: Recorder) -> None:
    network.handler = comfy(queue_status=400)
    response = client.post("/users/7/illustrations", json=REQUEST)
    assert response.status_code == 502
    assert "failed validation" in response.json()["detail"]


def test_an_unreachable_comfyui_is_a_503_not_a_crash(client: TestClient, network: Recorder) -> None:
    def down(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    network.handler = down
    response = client.post("/users/7/illustrations", json=REQUEST)
    assert response.status_code == 503
    assert "not reachable" in response.json()["detail"]


def test_status_is_available_when_comfyui_is_ready(client: TestClient, network: Recorder) -> None:
    network.handler = comfy()
    assert client.get("/status").json() == {"illustration": {"available": True, "detail": ""}}


def test_status_reports_the_model_download_while_it_runs(make_client, network: Recorder, tmp_path: Path) -> None:
    import time

    progress = tmp_path / "status.json"
    progress.write_text(json.dumps({"state": "downloading", "doneBytes": 4.3e9, "totalBytes": 1e10, "updatedAt": time.time()}))
    client = make_client(comfyui_pipeline="flux", flux_unet="flux1-dev-Q4_K_S.gguf", model_status_file=progress)

    def down(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    network.handler = down
    state = client.get("/status").json()["illustration"]
    assert state["available"] is False
    assert "43%" in state["detail"]


def test_a_user_id_or_picture_name_cannot_walk_out_of_the_folder(client: TestClient, network: Recorder) -> None:
    network.handler = comfy()
    assert client.post("/users/..%2F..%2Fetc/illustrations", json=REQUEST).status_code in (400, 404)
    assert client.get("/users/7/images/..%2F..%2Fsecret.png").status_code == 404
    assert client.get("/users/7/images/not-a-picture.txt").status_code == 404
    assert client.get("/users/bad id/images/" + "a" * 32 + ".png").status_code == 400


def test_a_browser_token_only_reaches_its_own_pictures(make_client, network: Recorder) -> None:
    network.handler = comfy()
    client = make_client(auth={"internal": False, "user_id": "7"})
    assert client.post("/users/7/illustrations", json=REQUEST).status_code == 200
    assert client.post("/users/8/illustrations", json=REQUEST).status_code == 403
    assert client.get("/users/8/images/" + "a" * 32 + ".png").status_code == 403


def test_dimensions_keep_the_pixel_budget_and_snap_to_the_grid() -> None:
    width, height = dimensions(0.8, 1024)
    assert width % 64 == 0 and height % 64 == 0
    assert abs(width * height - 1024 * 1024) / (1024 * 1024) < 0.1


def test_fill_workflow_only_turns_whole_numeric_tokens_into_numbers() -> None:
    values = {token: "" for token in comfy_module.TEXT_TOKENS} | {"%seed%": 7, "%prompt%": "a cat"}
    filled = fill_workflow({"a": "%seed%", "b": "seed %seed%", "c": ["%prompt%"]}, values)
    assert filled == {"a": 7, "b": "seed %seed%", "c": ["a cat"]}


def test_pictures_for_dark_pages_and_covers_ask_for_it() -> None:
    dark, _ = build_prompts("A moonlit tower", "ink", "interior", "dark")
    cover, _ = build_prompts("A moonlit tower", "ink", "cover")
    assert "black background" in dark
    assert "book cover" in cover
