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


def test_flux_draws_from_nothing_at_its_denoise_and_a_remix_at_its_strength_both_in_its_steps(make_client, network: Recorder) -> None:
    client = make_client(comfyui_pipeline="flux", flux_unet="flux1-dev-Q4_K_S.gguf", comfyui_steps=16, comfyui_denoise=0.85)
    network.handler = comfy()
    first = client.post("/users/7/illustrations", json=REQUEST)
    second = client.post("/users/7/illustrations", json={**REQUEST, "reference": REFERENCE, "strength": 0.4})
    assert first.status_code == 200 and second.status_code == 200
    fresh, remix = (b["prompt"]["8"]["inputs"] for b in network.bodies("/prompt"))
    assert (fresh["denoise"], fresh["steps"]) == (0.85, 16)
    assert (remix["denoise"], remix["steps"]) == (0.4, 16)
    # What ComfyUI was given comes back with the picture, as it went.
    workflow = network.bodies("/prompt")[0]["prompt"]
    sent = first.json()["sent"]
    assert sent["prompt"] == workflow["4"]["inputs"]["text"] and sent["prompt"].startswith("A lighthouse at dusk, pen and ink")
    assert sent["negative"] == "" and sent["guidance"] == 3.5 and sent["seed"] == 42
    assert (sent["steps"], sent["denoise"], sent["reference"]) == (16, 0.85, False)
    assert (sent["width"], sent["height"]) == (workflow["7"]["inputs"]["width"], workflow["7"]["inputs"]["height"])
    assert second.json()["sent"]["reference"] is True and second.json()["sent"]["denoise"] == 0.4


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
    # What a style picture (USO) needs: its LoRA, its projector, and the vision model.
    "LoraLoaderModelOnly": {"input": {"required": {"lora_name": [["uso-flux1-dit-lora-v1.safetensors"], {}]}}},
    "ModelPatchLoader": {"input": {"required": {"name": ["COMBO", {"options": ["uso-flux1-projector-v1.safetensors"]}]}}},
    "CLIPVisionLoader": {"input": {"required": {"clip_name": [["sigclip_vision_patch14_384.safetensors"], {}]}}},
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
    # Drawn at the 16-pixel step above (1024), then cut to exactly the size asked for.
    assert (response.json()["width"], response.json()["height"]) == (1010, 640)
    assert workflow["1"] == {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "flux1-dev-Q4_K_S.gguf"}}
    # The prompt exactly as written: no style's words, no cover or page wording.
    assert workflow["4"]["inputs"]["text"] == "A lighthouse at dusk"
    assert workflow["5"] == {"class_type": "FluxGuidance", "inputs": {"conditioning": ["4", 0], "guidance": 5.0}}
    assert workflow["6"]["inputs"]["text"] == "fog"
    assert workflow["7"] == {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 640, "batch_size": 1}}
    assert workflow["13"] == {
        "class_type": "ImageScale",
        "inputs": {"image": ["9", 0], "upscale_method": "lanczos", "width": 1010, "height": 640, "crop": "center"},
    }
    assert workflow["10"]["inputs"]["images"] == ["13", 0]
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


def test_the_lab_applies_a_style_picture_with_uso_as_comfyuis_template_does(make_client, network: Recorder) -> None:
    network.handler = comfy(object_info=LAB_INFO)
    client = lab_client(make_client)
    body = {**LAB, "style_image": REFERENCE, "style_image_weight": 0.8}
    assert client.post("/users/7/lab/illustrations", json=body).status_code == 200
    upload = [r.content for r in network.requests if r.url.path == "/upload/image"]
    assert any(b'filename="lab-style-' in u for u in upload)
    workflow = network.bodies("/prompt")[0]["prompt"]
    # The vision model reads the picture; USO's LoRA goes on the model, then the style.
    assert workflow["31"] == {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "sigclip_vision_patch14_384.safetensors"}}
    assert workflow["32"]["inputs"] | {"image": None} == {"clip_vision": ["31", 0], "image": None, "crop": "center"}
    assert workflow["33"] == {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {"model": ["1", 0], "lora_name": "uso-flux1-dit-lora-v1.safetensors", "strength_model": 0.8},
    }
    assert workflow["34"] == {"class_type": "ModelPatchLoader", "inputs": {"name": "uso-flux1-projector-v1.safetensors"}}
    assert workflow["35"] == {
        "class_type": "USOStyleReference",
        "inputs": {"model": ["33", 0], "model_patch": ["34", 0], "clip_vision_output": ["32", 0]},
    }
    # The style acts on the model the sampler draws with; the prompt stays the prompt.
    assert workflow["8"]["inputs"]["model"] == ["35", 0] and workflow["8"]["inputs"]["positive"] == ["5", 0]
    # Nothing of Redux is left.
    assert not {"StyleModelLoader", "StyleModelApply"} & {n["class_type"] for n in workflow.values()}


def test_a_style_picture_says_which_uso_files_are_missing(make_client, network: Recorder) -> None:
    info = {**LAB_INFO, "ModelPatchLoader": {"input": {"required": {"name": ["COMBO", {"options": []}]}}}}
    network.handler = comfy(object_info=info)
    client = lab_client(make_client)
    response = client.post("/users/7/lab/illustrations", json={**LAB, "style_image": REFERENCE})
    assert response.status_code == 503 and "uso-flux1-projector-v1.safetensors" in response.json()["detail"]
    assert not network.bodies("/prompt")
    options = client.get("/users/7/lab/options").json()
    assert options["styleReady"] is False and options["styleMissing"] == ["uso-flux1-projector-v1.safetensors"]
    # A checkpoint model cannot take USO, which is made for FLUX.
    network.handler = comfy(object_info=LAB_INFO)
    refused = client.post("/users/7/lab/illustrations", json={**LAB, "model": "sdxl_base.safetensors", "style_image": REFERENCE})
    assert refused.status_code == 422 and "FLUX" in refused.json()["detail"]


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
    assert options["styleReady"] is True and options["styleMissing"] == []
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
    assert workflow["8"]["inputs"]["positive"] == ["4", 0] and workflow["8"]["inputs"]["model"] == ["35", 0]
    # Either one alone is enough to draw; neither, nor a prompt, nor a picture, is not.
    assert client.post("/users/7/lab/illustrations", json={**LAB, "prompt": "", "t5xxl": "A fox."}).status_code == 200
    assert client.post("/users/7/lab/illustrations", json={"model": "flux1-dev-Q4_K_S.gguf", "clip_l": " ", "t5xxl": ""}).status_code == 422
    # A checkpoint model has one text encoder: it reads the T5 text.
    assert client.post("/users/7/lab/illustrations", json={"model": "sdxl_base.safetensors", "t5xxl": "A fox."}).status_code == 200
    assert network.bodies("/prompt")[-1]["prompt"]["4"] == {"class_type": "CLIPTextEncode", "inputs": {"text": "A fox.", "clip": ["1", 1]}}


def test_a_picture_is_drawn_in_its_art_style_and_medium(client: TestClient, network: Recorder) -> None:
    network.handler = comfy()
    body = {**REQUEST, "artStyle": "art-nouveau", "medium": "woodcut"}
    assert client.post("/users/7/illustrations", json=body).status_code == 200
    text = network.bodies("/prompt")[0]["prompt"]["2"]["inputs"]["text"]
    assert text.startswith("A lighthouse at dusk, art nouveau illustration") and "woodcut print" in text
    assert "pen and ink" not in text  # the older styleId ("ink") gives way


def test_flux_skips_unchanging_steps_with_easycache_unless_it_is_off(make_client, network: Recorder) -> None:
    network.handler = comfy()
    client = make_client(comfyui_pipeline="flux", flux_unet="flux1-dev-Q4_K_S.gguf", flux_easycache=0.2)
    assert client.post("/users/7/illustrations", json={**REQUEST, "reference": REFERENCE}).status_code == 200
    workflow = network.bodies("/prompt")[0]["prompt"]
    assert workflow["12"] == {
        "class_type": "EasyCache",
        "inputs": {"model": ["1", 0], "reuse_threshold": 0.2, "start_percent": 0.15, "end_percent": 0.95, "verbose": False},
    }
    assert workflow["8"]["inputs"]["model"] == ["12", 0]
    assert workflow["7"]["class_type"] == "VAEEncode"  # a redraw keeps it too
    off = make_client(comfyui_pipeline="flux", flux_unet="flux1-dev-Q4_K_S.gguf", flux_easycache=0.0)
    assert off.post("/users/7/illustrations", json=REQUEST).status_code == 200
    plain = network.bodies("/prompt")[-1]["prompt"]
    assert "12" not in plain and plain["8"]["inputs"]["model"] == ["1", 0]


def test_the_lab_adds_easycache_after_a_style_when_asked(make_client, network: Recorder) -> None:
    network.handler = comfy(object_info=LAB_INFO)
    client = lab_client(make_client)
    assert client.post("/users/7/lab/illustrations", json={**LAB, "style_image": REFERENCE, "easycache": 0.3}).status_code == 200
    workflow = network.bodies("/prompt")[0]["prompt"]
    assert workflow["36"]["inputs"]["model"] == ["35", 0] and workflow["36"]["inputs"]["reuse_threshold"] == 0.3
    assert workflow["8"]["inputs"]["model"] == ["36", 0]
    assert client.post("/users/7/lab/illustrations", json=LAB).status_code == 200
    assert "36" not in network.bodies("/prompt")[-1]["prompt"]


def test_the_gpu_is_handed_back_after_every_picture_drawn_or_failed(client: TestClient, network: Recorder) -> None:
    network.handler = comfy()
    assert client.post("/users/7/illustrations", json=REQUEST).status_code == 200
    assert network.bodies("/free") == [{"unload_models": True}]
    network.handler = comfy(queue_status=400)
    assert client.post("/users/7/illustrations", json=REQUEST).status_code == 502
    assert network.bodies("/free") == [{"unload_models": True}, {"unload_models": True}]


def test_the_lab_gives_exactly_the_size_asked_for(make_client, network: Recorder) -> None:
    network.handler = comfy(object_info=LAB_INFO)
    client = lab_client(make_client)
    # 900 is between FLUX's 16-pixel steps: drawn at 912, trimmed back to 900. It used to come
    # back as 896.
    response = client.post("/users/7/lab/illustrations", json={**LAB, "width": 900, "height": 1200, "image": REFERENCE})
    assert (response.json()["width"], response.json()["height"]) == (900, 1200)
    workflow = network.bodies("/prompt")[0]["prompt"]
    assert (workflow["21"]["inputs"]["width"], workflow["21"]["inputs"]["height"]) == (912, 1200)
    assert (workflow["13"]["inputs"]["width"], workflow["13"]["inputs"]["height"]) == (900, 1200)
    # A size on the steps already is drawn as it is, with nothing to cut.
    client.post("/users/7/lab/illustrations", json={**LAB, "width": 912, "height": 1216})
    exact = network.bodies("/prompt")[-1]["prompt"]
    assert "13" not in exact and exact["10"]["inputs"]["images"] == ["9", 0]


# --- after drawing: enlarged ---------------------------------------------------------


def real_png(width: int = 64, height: int = 48) -> bytes:
    from io import BytesIO

    from PIL import Image

    out = BytesIO()
    Image.new("RGB", (width, height), (200, 120, 40)).save(out, format="PNG")
    return out.getvalue()


def saved_picture(tmp_path: Path, src: str):
    from PIL import Image

    return Image.open(tmp_path / "talari" / "images" / "7" / src.rsplit("/", 1)[1])


def test_a_page_filling_picture_is_enlarged(client: TestClient, network: Recorder, tmp_path: Path) -> None:
    drawn = real_png()
    base = comfy()
    network.handler = lambda r: httpx.Response(200, content=drawn) if r.url.path == "/view" else base(r)

    response = client.post("/users/7/illustrations", json={**REQUEST, "upscale": 2})
    assert response.status_code == 200, response.text
    assert saved_picture(tmp_path, response.json()["src"]).size == (128, 96)
    # Without asking, it is kept as drawn.
    response = client.post("/users/7/illustrations", json=REQUEST)
    assert saved_picture(tmp_path, response.json()["src"]).size == (64, 48)


def test_an_enlargement_is_capped_at_twice(client: TestClient, network: Recorder) -> None:
    network.handler = comfy()
    assert client.post("/users/7/illustrations", json={**REQUEST, "upscale": 3}).status_code == 422


def test_a_substyle_is_drawn_in_its_own_words_and_only_with_its_art_style() -> None:
    from phansora.products.talari.styles import look_text

    assert look_text("art-nouveau", "watercolor", "glasgow-style").startswith("Glasgow style, elongated vertical lines")
    assert "watercolor" in look_text("art-nouveau", "watercolor", "glasgow-style")
    # Another art style's substyle, or none: the art style as it is.
    assert look_text("art-deco", "auto", "glasgow-style") == look_text("art-deco", "auto")
    assert look_text("art-nouveau", "auto", None).startswith("art nouveau illustration")


def test_a_picture_is_drawn_in_its_substyle(client: TestClient, network: Recorder) -> None:
    network.handler = comfy()
    response = client.post("/users/7/illustrations", json={**REQUEST, "artStyle": "comic-book", "substyle": "golden-age", "medium": "auto"})
    assert response.status_code == 200, response.text
    text = network.bodies("/prompt")[0]["prompt"]["2"]["inputs"]["text"]
    assert "golden age comic book art" in text and "bold inked outlines" not in text


# --- background removal --------------------------------------------------------------

BG_INFO = {"LoadBackgroundRemovalModel": {"input": {"required": {"bg_removal_name": [["birefnet.safetensors"], {}]}}}}


def a_mask(width: int = 64, height: int = 48) -> bytes:
    from io import BytesIO

    from PIL import Image

    mask = Image.new("L", (width, height), 0)
    mask.paste(255, (width // 2, 0, width, height))
    out = BytesIO()
    mask.save(out, format="PNG")
    return out.getvalue()


def test_a_picture_asked_without_its_background_is_cut_out_by_birefnet(client: TestClient, network: Recorder, tmp_path: Path) -> None:
    views = iter([real_png(64, 48), a_mask()])
    base = comfy(object_info=BG_INFO)
    network.handler = lambda r: httpx.Response(200, content=next(views)) if r.url.path == "/view" else base(r)
    response = client.post("/users/7/illustrations", json={**REQUEST, "cutout": True})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["sent"]["cutout"] is True
    # The drawing, then its mask: ComfyUI's own nodes, the drawn picture sent back in.
    draw, cut = (b["prompt"] for b in network.bodies("/prompt"))
    assert cut["2"]["inputs"] == {"bg_removal_name": "birefnet.safetensors"}
    assert cut["3"]["class_type"] == "RemoveBackground" and cut["4"]["class_type"] == "MaskToImage"
    assert any(b'filename="talari-cutout-' in r.content for r in network.requests if r.url.path == "/upload/image")
    kept = saved_picture(tmp_path, body["src"])
    assert kept.mode == "RGBA" and kept.size == (64, 48)
    assert kept.getpixel((5, 20))[3] == 0 and kept.getpixel((60, 20))[3] == 255


def test_a_picture_that_cannot_be_cut_out_is_kept_as_drawn_and_says_why(client: TestClient, network: Recorder, tmp_path: Path) -> None:
    network.handler = comfy(object_info={"LoadBackgroundRemovalModel": {"input": {"required": {"bg_removal_name": [[], {}]}}}})
    response = client.post("/users/7/illustrations", json={**REQUEST, "cutout": True})
    assert response.status_code == 200, response.text
    sent = response.json()["sent"]
    assert sent["cutout"] is False and "models/background_removal" in sent["cutout_error"]
    assert len(network.bodies("/prompt")) == 1


# --- upscaling, for print -------------------------------------------------------------

UPSCALE_INFO = {"UpscaleModelLoader": {"input": {"required": {"model_name": [["4x_NMKD-Siax_200k.pth", "4x-UltraSharp.pth"], {}]}}}}


def a_picture_of_theirs(tmp_path: Path, width: int = 64, height: int = 48) -> str:
    name = "ab" * 16 + ".png"
    folder = tmp_path / "talari" / "images" / "7"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(real_png(width, height))
    return name


def test_a_picture_is_enlarged_by_its_model_and_scaled_to_what_was_asked(client: TestClient, network: Recorder, tmp_path: Path) -> None:
    name = a_picture_of_theirs(tmp_path)
    enlarged = real_png(128, 96)
    base = comfy(object_info=UPSCALE_INFO)
    network.handler = lambda r: httpx.Response(200, content=enlarged) if r.url.path == "/view" else base(r)
    response = client.post("/users/7/upscale", json={"model": "nmkd-siax", "scale": 2, "name": name})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["src"].startswith("/talari/users/7/images/") and (body["width"], body["height"]) == (128, 96) and body["model"] == "nmkd-siax"
    workflow = network.bodies("/prompt")[0]["prompt"]
    assert workflow["2"]["inputs"]["model_name"] == "4x_NMKD-Siax_200k.pth"
    assert workflow["3"]["class_type"] == "ImageUpscaleWithModel"
    assert workflow["4"]["inputs"] == {"image": ["3", 0], "upscale_method": "lanczos", "scale_by": 0.5}
    assert workflow["9"]["inputs"]["images"] == ["4", 0]
    # The picture went to ComfyUI as an upload of its own.
    assert any(b'filename="upscale-' in r.content for r in network.requests if r.url.path == "/upload/image")
    assert saved_picture(tmp_path, body["src"]).size == (128, 96)


def test_at_four_times_the_model_alone_enlarges_it_and_a_picture_can_be_sent(client: TestClient, network: Recorder) -> None:
    import base64

    network.handler = comfy(object_info=UPSCALE_INFO)
    sent = base64.b64encode(real_png(32, 32)).decode()
    response = client.post("/users/7/upscale", json={"model": "ultrasharp", "image": sent})
    assert response.status_code == 200, response.text
    workflow = network.bodies("/prompt")[0]["prompt"]
    assert "4" not in workflow and workflow["9"]["inputs"]["images"] == ["3", 0]


def test_a_print_copy_is_a_jpeg_and_a_transparent_picture_keeps_its_transparency(client: TestClient, network: Recorder, tmp_path: Path) -> None:
    from io import BytesIO

    from PIL import Image

    enlarged = real_png(128, 96)
    base = comfy(object_info=UPSCALE_INFO)
    network.handler = lambda r: httpx.Response(200, content=enlarged) if r.url.path == "/view" else base(r)
    response = client.post("/users/7/upscale", json={"model": "ultrasharp", "scale": 2, "name": a_picture_of_theirs(tmp_path), "format": "auto"})
    assert response.status_code == 200, response.text
    src = response.json()["src"]
    assert src.endswith(".jpg") and saved_picture(tmp_path, src).format == "JPEG"
    assert client.get(src.removeprefix("/talari")).headers["content-type"] == "image/jpeg"

    # ComfyUI hands back colors only: a cut-out's clear part is enlarged and put back.
    cutout = Image.new("RGBA", (64, 48), (200, 120, 40, 255))
    cutout.paste((0, 0, 0, 0), (0, 0, 32, 48))
    sent = BytesIO()
    cutout.save(sent, format="PNG")
    name = "ef" * 16 + ".png"
    (tmp_path / "talari" / "images" / "7" / name).write_bytes(sent.getvalue())
    response = client.post("/users/7/upscale", json={"model": "ultrasharp", "scale": 2, "name": name, "format": "auto"})
    assert response.status_code == 200, response.text
    kept = saved_picture(tmp_path, response.json()["src"])
    assert kept.format == "PNG" and kept.mode == "RGBA" and kept.size == (128, 96)
    assert kept.getpixel((10, 50))[3] == 0 and kept.getpixel((110, 50))[3] == 255


def test_pictures_are_drawn_at_768_in_16_steps_at_0_85_unless_set_otherwise(monkeypatch: pytest.MonkeyPatch) -> None:
    from phansora.products.talari.config import load_settings

    for name in ("FLUX_SIZE", "COMFYUI_PIPELINE", "FLUX_STEPS", "FLUX_DENOISE"):
        monkeypatch.delenv(name, raising=False)
    settings = load_settings()
    assert (settings.comfyui_base_size, settings.comfyui_steps, settings.comfyui_denoise) == (768, 16, 0.85)
    assert dimensions(6.25 / 9.25, 768) == (640, 960)
    monkeypatch.setenv("FLUX_SIZE", "1024")
    assert load_settings().comfyui_base_size == 1024


def test_an_upscaler_comfyui_lacks_says_so(client: TestClient, network: Recorder, tmp_path: Path) -> None:
    network.handler = comfy(object_info=UPSCALE_INFO)
    response = client.post("/users/7/upscale", json={"model": "animesharp", "name": a_picture_of_theirs(tmp_path)})
    assert response.status_code == 503 and "4x-AnimeSharp.pth" in response.json()["detail"]


def test_upscaling_refuses_what_it_cannot_do(client: TestClient, network: Recorder, tmp_path: Path) -> None:
    network.handler = comfy(object_info=UPSCALE_INFO)
    assert client.post("/users/7/upscale", json={"model": "nmkd-siax", "name": "../../etc/passwd"}).status_code == 404
    assert client.post("/users/7/upscale", json={"model": "nmkd-siax", "name": "cd" * 16 + ".png"}).status_code == 404
    assert client.post("/users/7/upscale", json={"model": "nmkd-siax"}).status_code == 422
    assert client.post("/users/7/upscale", json={"model": "waifu", "name": a_picture_of_theirs(tmp_path)}).status_code == 422
    big = a_picture_of_theirs(tmp_path, 2100, 100)
    response = client.post("/users/7/upscale", json={"model": "nmkd-siax", "scale": 4, "name": big})
    assert response.status_code == 422 and "enlarge it less" in response.json()["detail"]
    assert network.bodies("/prompt") == []


def test_the_lab_says_which_upscalers_comfyui_has(make_client, network: Recorder) -> None:
    network.handler = comfy(object_info={**LAB_INFO, **UPSCALE_INFO})
    upscalers = lab_client(make_client).get("/users/7/lab/options").json()["upscalers"]
    assert [(u["id"], u["installed"]) for u in upscalers] == [("ultrasharp", True), ("nmkd-siax", True), ("animesharp", False)]


# --- the ComfyUI gateway ------------------------------------------------------------------
# ComfyUI lent to Telios, which builds its own workflows: a workflow and its pictures in,
# the picture out, nothing kept, and only the nodes the gateway runs.

FLUX_INFO = {
    **LAB_INFO,
    "DualCLIPLoader": {"input": {"required": {"clip_name1": [["t5xxl_fp8_e4m3fn.safetensors", "clip_l.safetensors"], {}]}}},
    "VAELoader": {"input": {"required": {"vae_name": [["ae.safetensors"], {}]}}},
}
RESTYLE = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "source.png"}},
    "2": {"class_type": "ImageScale", "inputs": {"image": ["1", 0], "upscale_method": "lanczos", "width": 768, "height": 768, "crop": "center"}},
    "9": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": "../elsewhere"}},
}
SOURCE = _b64.b64encode(b"\x89PNG\r\n\x1a\n a source picture").decode()


def test_the_gateway_runs_a_workflow_with_its_pictures_and_keeps_nothing(make_client, network: Recorder, tmp_path: Path) -> None:
    network.handler = comfy(object_info=FLUX_INFO)
    client = lab_client(make_client, auth={"internal": False, "user_id": "telios"}, lab_users=frozenset({"telios"}))
    response = client.post("/users/telios/comfyui/run", json={"workflow": RESTYLE, "images": {"source.png": SOURCE}})
    assert response.status_code == 200 and response.content == PNG and response.headers["content-type"] == "image/png"
    [sent] = network.bodies("/prompt")
    assert sent["prompt"]["1"]["inputs"]["image"] == "talari-ref.png"   # what ComfyUI named the upload
    assert sent["prompt"]["9"]["inputs"]["filename_prefix"] == "gateway"
    assert not (tmp_path / "talari").exists()
    assert any(r.url.path == "/free" for r in network.requests)        # the GPU handed back


def test_the_gateway_says_what_to_draw_with(make_client, network: Recorder) -> None:
    network.handler = comfy(object_info={**FLUX_INFO, **UPSCALE_INFO})
    info = lab_client(make_client).get("/users/telios/comfyui").json()
    assert info["ready"] is True and info["flux"]["unet"] == "flux1-dev-Q4_K_S.gguf"
    assert info["upscalers"] == ["4x-UltraSharp.pth", "4x_NMKD-Siax_200k.pth"] and info["backgroundRemoval"] is None
    assert "KSampler" in info["nodes"]


def test_the_gateway_is_open_only_to_the_named_accounts(make_client, network: Recorder) -> None:
    network.handler = comfy(object_info=FLUX_INFO)
    other = lab_client(make_client, auth={"internal": False, "user_id": "8"}, lab_users=frozenset({"telios"}))
    assert other.post("/users/8/comfyui/run", json={"workflow": RESTYLE, "images": {"source.png": SOURCE}}).status_code == 403
    assert other.get("/users/8/comfyui").status_code == 403
    assert network.bodies("/prompt") == []


@pytest.mark.parametrize("workflow, images, reason", [
    ({"1": {"class_type": "LoadImage", "inputs": {"image": "/etc/passwd"}}, "9": RESTYLE["9"]}, {}, "not sent"),
    ({**RESTYLE, "5": {"class_type": "SaveText", "inputs": {"text": "x"}}}, {"source.png": SOURCE}, "SaveText"),
    ({"1": RESTYLE["1"]}, {"source.png": SOURCE}, "saves no picture"),
    ({}, {}, "empty"),
    (RESTYLE, {"../source.png": SOURCE}, "plain file name"),
])
def test_the_gateway_refuses_a_workflow_before_comfyui_sees_it(make_client, network: Recorder, workflow, images, reason) -> None:
    network.handler = comfy(object_info=FLUX_INFO)
    response = lab_client(make_client).post("/users/telios/comfyui/run", json={"workflow": workflow, "images": images})
    assert response.status_code == 422 and reason in response.json()["detail"]
    assert network.bodies("/prompt") == [] and not any(r.url.path == "/upload/image" for r in network.requests)
