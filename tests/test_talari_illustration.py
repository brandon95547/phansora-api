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
        if path == "/upload/image":
            return httpx.Response(200, json={"name": "talari-ref.png", "subfolder": "", "type": "input"})
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


# ── Redrawing from a reference picture (image-to-image) ──────────────────────
# A page built from an imported picture: the picture is the starting point, and the book's
# style is what it is redrawn in. The engine gets the picture uploaded, and a workflow that
# starts the sampler from it rather than from noise.

import base64 as _b64

REFERENCE = _b64.b64encode(b"\x89PNG\r\n\x1a\n a reference picture").decode()


def test_a_reference_is_uploaded_and_the_sampler_starts_from_it(client: TestClient, network: Recorder) -> None:
    network.handler = comfy()
    response = client.post("/users/7/illustrations", json={**REQUEST, "reference": REFERENCE, "strength": 0.45})
    assert response.status_code == 200

    upload = next(r for r in network.requests if r.url.path == "/upload/image")
    assert b"a reference picture" in upload.content and b'name="type"' in upload.content

    workflow = network.bodies("/prompt")[0]["prompt"]
    assert workflow["20"] == {"class_type": "LoadImage", "inputs": {"image": "talari-ref.png"}}
    # Scaled and cropped to the picture's own shape, never stretched to it.
    assert workflow["21"]["class_type"] == "ImageScale"
    assert (workflow["21"]["inputs"]["width"], workflow["21"]["inputs"]["height"]) == (896, 1152)
    assert workflow["21"]["inputs"]["crop"] == "center"
    # The empty latent is gone: the sampler starts from the encoded reference, at the strength asked.
    assert workflow["4"] == {"class_type": "VAEEncode", "inputs": {"pixels": ["21", 0], "vae": ["1", 2]}}
    assert workflow["5"]["inputs"]["latent_image"] == ["4", 0]
    assert workflow["5"]["inputs"]["denoise"] == 0.45


def test_flux_redraws_with_its_own_vae(make_client, network: Recorder) -> None:
    client = make_client(comfyui_pipeline="flux", flux_unet="flux1-dev-Q4_K_S.gguf")
    network.handler = comfy()
    assert client.post("/users/7/illustrations", json={**REQUEST, "reference": REFERENCE}).status_code == 200
    workflow = network.bodies("/prompt")[0]["prompt"]
    assert workflow["7"] == {"class_type": "VAEEncode", "inputs": {"pixels": ["21", 0], "vae": ["3", 0]}}
    assert workflow["8"]["inputs"]["denoise"] == 0.6  # the default
    assert workflow["8"]["inputs"]["cfg"] == 1.0


def test_without_a_reference_nothing_is_uploaded_and_it_draws_from_noise(client: TestClient, network: Recorder) -> None:
    network.handler = comfy()
    assert client.post("/users/7/illustrations", json=REQUEST).status_code == 200
    assert not any(r.url.path == "/upload/image" for r in network.requests)
    workflow = network.bodies("/prompt")[0]["prompt"]
    assert workflow["4"]["class_type"] == "EmptyLatentImage" and workflow["5"]["inputs"]["denoise"] == 1.0


def test_a_reference_alone_is_enough_but_nothing_at_all_is_refused(client: TestClient, network: Recorder) -> None:
    network.handler = comfy()
    assert client.post("/users/7/illustrations", json={**REQUEST, "prompt": "", "reference": REFERENCE}).status_code == 200
    assert client.post("/users/7/illustrations", json={**REQUEST, "prompt": ""}).status_code == 422


def test_a_reference_that_is_not_a_picture_is_refused_before_comfyui_sees_it(client: TestClient, network: Recorder) -> None:
    network.handler = comfy()
    not_a_picture = _b64.b64encode(b"%PDF-1.7 not a picture").decode()
    response = client.post("/users/7/illustrations", json={**REQUEST, "reference": not_a_picture})
    assert response.status_code == 502
    assert "PNG, JPEG or WebP" in response.json()["detail"]
    assert not any(r.url.path in ("/upload/image", "/prompt") for r in network.requests)


def test_a_custom_workflow_gets_the_reference_and_the_denoise() -> None:
    template = {"load": {"inputs": {"image": "%image%"}}, "sample": {"inputs": {"denoise": "%denoise%", "seed": "%seed%"}}}
    values = {token: "" for token in comfy_module.TEXT_TOKENS} | {token: 0 for token in comfy_module.NUMERIC_TOKENS}
    values.update({"%image%": "talari-ref.png", "%denoise%": 0.45, "%seed%": 7})
    filled = fill_workflow(template, values)
    assert filled == {"load": {"inputs": {"image": "talari-ref.png"}}, "sample": {"inputs": {"denoise": 0.45, "seed": 7}}}


# --- the ComfyUI lab ----------------------------------------------------------------------
# The styleguide's ComfyUI test page: every workflow input passed through as given.

LAB_INFO = {
    "UnetLoaderGGUF": {"input": {"required": {"unet_name": [["flux1-dev-Q4_K_S.gguf"], {}]}}},
    "UNETLoader": {"input": {"required": {"unet_name": [[], {}]}}},
    "KSampler": {"input": {"required": {"sampler_name": [["euler", "dpmpp_2m"], {}], "scheduler": [["simple", "karras"], {}]}}},
    "StyleModelLoader": {"input": {"required": {"style_model_name": [["flux1-redux-dev.safetensors"], {}]}}},
    "CLIPVisionLoader": {"input": {"required": {"clip_name": [["sigclip_vision_patch14_384.safetensors"], {}]}}},
    "StyleModelApply": {"input": {"required": {"strength_type": ["COMBO", {"options": ["multiply", "attn_bias"]}]}}},
}
LAB = {"prompt": "A lighthouse at dusk", "model": "flux1-dev-Q4_K_S.gguf"}


def lab_client(make_client, **overrides: Any) -> TestClient:
    return make_client(comfyui_pipeline="flux", flux_unet="flux1-dev-Q4_K_S.gguf", **overrides)


def test_the_lab_is_open_only_to_the_named_accounts_and_the_app(make_client, network: Recorder) -> None:
    network.handler = comfy(object_info=LAB_INFO)
    named = lab_client(make_client, auth={"internal": False, "user_id": "7"}, lab_users=frozenset({"7"}))
    assert named.post("/users/7/lab/illustrations", json=LAB).status_code == 200
    assert named.get("/users/7/lab/options").status_code == 200
    other = lab_client(make_client, auth={"internal": False, "user_id": "8"}, lab_users=frozenset({"7"}))
    refused = other.post("/users/8/lab/illustrations", json=LAB)
    assert refused.status_code == 403 and "lab" in refused.json()["detail"]
    assert other.get("/users/8/lab/options").status_code == 403
    app = lab_client(make_client, auth={"internal": True})
    assert app.post("/users/9/lab/illustrations", json=LAB).status_code == 200


def test_the_lab_passes_every_setting_through_untouched(make_client, network: Recorder) -> None:
    network.handler = comfy(object_info=LAB_INFO)
    client = lab_client(make_client)
    settings = {
        **LAB, "negative_prompt": "fog", "width": 1010, "height": 640, "seed": 2**40, "steps": 9, "cfg": 2.5,
        "guidance": 5.0, "denoise": 0.9, "sampler_name": "dpmpp_2m", "scheduler": "karras",
    }
    response = client.post("/users/7/lab/illustrations", json=settings)
    assert response.status_code == 200
    workflow = network.bodies("/prompt")[0]["prompt"]
    assert response.json()["workflow"] == workflow
    assert (response.json()["width"], response.json()["height"]) == (1008, 640)  # 16-pixel steps
    assert workflow["1"] == {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "flux1-dev-Q4_K_S.gguf"}}
    # The prompt exactly as written: no style's words, no cover or page wording.
    assert workflow["4"]["inputs"]["text"] == "A lighthouse at dusk"
    assert workflow["5"] == {"class_type": "FluxGuidance", "inputs": {"conditioning": ["4", 0], "guidance": 5.0}}
    assert workflow["6"]["inputs"]["text"] == "fog"
    assert workflow["7"] == {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1008, "height": 640, "batch_size": 1}}
    assert workflow["8"]["inputs"] | {"model": None} == {
        "model": None, "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0], "seed": 2**40,
        "steps": 9, "cfg": 2.5, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.9,
    }
    # The camelCase spellings mean the same.
    assert client.post("/users/7/lab/illustrations", json={**LAB, "negativePrompt": "rain", "samplerName": "euler"}).status_code == 200
    again = network.bodies("/prompt")[1]["prompt"]
    assert again["6"]["inputs"]["text"] == "rain" and again["8"]["inputs"]["sampler_name"] == "euler"


def test_the_lab_redraws_a_picture_inside_its_mask(make_client, network: Recorder) -> None:
    network.handler = comfy(object_info=LAB_INFO)
    client = lab_client(make_client)
    response = client.post("/users/7/lab/illustrations", json={**LAB, "image": REFERENCE, "mask": REFERENCE, "denoise": 0.7, "width": 832, "height": 1216})
    assert response.status_code == 200
    uploads = [r.content for r in network.requests if r.url.path == "/upload/image"]
    assert any(b'filename="lab-image-' in u for u in uploads) and any(b'filename="lab-mask-' in u for u in uploads)
    workflow = network.bodies("/prompt")[0]["prompt"]
    assert workflow["21"]["inputs"] | {"image": None} == {"image": None, "upscale_method": "lanczos", "width": 832, "height": 1216, "crop": "center"}
    assert workflow["7"] == {"class_type": "VAEEncode", "inputs": {"pixels": ["21", 0], "vae": ["3", 0]}}
    # The mask is cut to the same size and shape as the picture, then limits the noise.
    assert workflow["22"]["class_type"] == "LoadImageMask" and workflow["22"]["inputs"]["channel"] == "red"
    assert (workflow["24"]["inputs"]["width"], workflow["24"]["inputs"]["height"], workflow["24"]["inputs"]["crop"]) == (832, 1216, "center")
    assert workflow["26"] == {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["7", 0], "mask": ["25", 0]}}
    assert workflow["8"]["inputs"]["latent_image"] == ["26", 0] and workflow["8"]["inputs"]["denoise"] == 0.7


def test_the_lab_applies_a_style_picture_with_its_weight(make_client, network: Recorder) -> None:
    network.handler = comfy(object_info=LAB_INFO)
    client = lab_client(make_client)
    body = {**LAB, "style_image": REFERENCE, "style_image_weight": 0.4, "style_strength_type": "attn_bias"}
    assert client.post("/users/7/lab/illustrations", json=body).status_code == 200
    workflow = network.bodies("/prompt")[0]["prompt"]
    assert workflow["31"]["inputs"]["clip_name"] == "sigclip_vision_patch14_384.safetensors"
    assert workflow["33"]["inputs"]["style_model_name"] == "flux1-redux-dev.safetensors"
    assert workflow["34"]["inputs"] | {"style_model": None, "clip_vision_output": None} == {
        "conditioning": ["5", 0], "style_model": None, "clip_vision_output": None, "strength": 0.4, "strength_type": "attn_bias",
    }
    assert workflow["8"]["inputs"]["positive"] == ["34", 0]


def test_a_style_picture_without_the_style_models_says_what_is_missing(make_client, network: Recorder) -> None:
    info = {**LAB_INFO, "StyleModelLoader": {"input": {"required": {"style_model_name": [[], {}]}}}}
    network.handler = comfy(object_info=info)
    response = lab_client(make_client).post("/users/7/lab/illustrations", json={**LAB, "style_image": REFERENCE})
    assert response.status_code == 503 and "style model" in response.json()["detail"]
    assert not network.bodies("/prompt")


def test_the_lab_refuses_a_mask_without_a_picture_and_an_empty_request(make_client, network: Recorder) -> None:
    network.handler = comfy(object_info=LAB_INFO)
    client = lab_client(make_client)
    assert client.post("/users/7/lab/illustrations", json={**LAB, "mask": REFERENCE}).status_code == 422
    assert client.post("/users/7/lab/illustrations", json={"model": "flux1-dev-Q4_K_S.gguf"}).status_code == 422
    assert client.post("/users/7/lab/illustrations", json={**LAB, "steps": 500}).status_code == 422


def test_a_checkpoint_model_loads_from_one_file_without_flux_guidance(make_client, network: Recorder) -> None:
    network.handler = comfy(object_info=LAB_INFO)
    client = lab_client(make_client)
    assert client.post("/users/7/lab/illustrations", json={**LAB, "model": "sdxl_base.safetensors", "cfg": 6}).status_code == 200
    workflow = network.bodies("/prompt")[0]["prompt"]
    assert workflow["1"] == {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sdxl_base.safetensors"}}
    assert "5" not in workflow and workflow["8"]["inputs"]["positive"] == ["4", 0]
    assert workflow["7"]["class_type"] == "EmptyLatentImage"
    assert workflow["4"]["inputs"]["clip"] == ["1", 1] and workflow["9"]["inputs"]["vae"] == ["1", 2]


def test_the_lab_lists_what_this_comfyui_has(make_client, network: Recorder) -> None:
    network.handler = comfy(object_info=LAB_INFO)
    options = lab_client(make_client).get("/users/7/lab/options").json()
    assert options["models"] == ["flux1-dev-Q4_K_S.gguf", "sdxl_base.safetensors", "other.safetensors"]
    assert options["defaultModel"] == "flux1-dev-Q4_K_S.gguf"
    assert options["samplers"] == ["euler", "dpmpp_2m"] and options["schedulers"] == ["simple", "karras"]
    assert options["styleModels"] == ["flux1-redux-dev.safetensors"]
    assert options["clipVision"] == ["sigclip_vision_patch14_384.safetensors"]
    assert options["strengthTypes"] == ["multiply", "attn_bias"]
    assert options["defaults"]["cfg"] == 1.0 and options["defaults"]["guidance"] == 3.5


def test_flux_reads_clip_l_and_t5xxl_separately_with_its_guidance_inside(make_client, network: Recorder) -> None:
    network.handler = comfy(object_info=LAB_INFO)
    client = lab_client(make_client)
    body = {
        "model": "flux1-dev-Q4_K_S.gguf", "clip_l": "watercolor, soft light", "t5xxl": "A lighthouse on a rocky coast at dusk.",
        "guidance": 4.5, "style_image": REFERENCE,
    }
    assert client.post("/users/7/lab/illustrations", json=body).status_code == 200
    workflow = network.bodies("/prompt")[0]["prompt"]
    assert workflow["4"] == {
        "class_type": "CLIPTextEncodeFlux",
        "inputs": {"clip": ["2", 0], "clip_l": "watercolor, soft light", "t5xxl": "A lighthouse on a rocky coast at dusk.", "guidance": 4.5},
    }
    assert "5" not in workflow  # no separate FluxGuidance: it is inside
    assert workflow["34"]["inputs"]["conditioning"] == ["4", 0] and workflow["8"]["inputs"]["positive"] == ["34", 0]
    # Either one alone is enough to draw; neither, nor a prompt, nor a picture, is not.
    assert client.post("/users/7/lab/illustrations", json={**LAB, "prompt": "", "t5xxl": "A fox."}).status_code == 200
    assert client.post("/users/7/lab/illustrations", json={"model": "flux1-dev-Q4_K_S.gguf", "clip_l": " ", "t5xxl": ""}).status_code == 422
    # A checkpoint model has one text encoder: it reads the T5 text.
    assert client.post("/users/7/lab/illustrations", json={"model": "sdxl_base.safetensors", "t5xxl": "A fox."}).status_code == 200
    assert network.bodies("/prompt")[-1]["prompt"]["4"] == {"class_type": "CLIPTextEncode", "inputs": {"text": "A fox.", "clip": ["1", 1]}}
