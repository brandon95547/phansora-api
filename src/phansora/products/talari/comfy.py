"""ComfyUI client: queue a workflow, wait for it, fetch the picture.

ComfyUI runs in its own container (Phansora's docker-compose.prod.yml, service
"comfyui"), so it has its own CUDA context. That is why nothing here goes through
``phansora.shared.gpu``: that gate protects CUDA work inside THIS process while the
voice engine captures graphs, and ComfyUI's work never happens in this process.
Sharing the card's memory is handled on ComfyUI's side (``--cache-none --reserve-vram``).
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import math
import random
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

from .config import Settings
from .errors import NotConfigured, ServiceError
from .schemas import IllustrationRequest, LabOptions, LabRequest
from .styles import build_prompts

POLL_INTERVAL_S = 1.0

# The files a style picture needs (USO, ByteDance's style reference for FLUX.1 Dev), as
# ComfyUI's template names them and Phansora's ComfyUI container fetches them.
USO_LORA = "uso-flux1-dit-lora-v1.safetensors"
USO_PROJECTOR = "uso-flux1-projector-v1.safetensors"
USO_VISION = "sigclip_vision_patch14_384.safetensors"
SIZE_STEP = 64
MIN_SIZE = 256

# Values a custom workflow file may ask for. A string that is exactly one of
# the numeric tokens becomes a number; the text tokens are replaced in place.
# %image% is the uploaded reference picture's name (empty without one) and %denoise% how
# far to move from it (1.0 without one), for a custom workflow that redraws.
NUMERIC_TOKENS = ("%seed%", "%width%", "%height%", "%steps%", "%cfg%", "%denoise%")
TEXT_TOKENS = ("%prompt%", "%negative%", "%checkpoint%", "%sampler%", "%scheduler%", "%image%")

# What a reference picture's bytes have to start with: PNG, JPEG, WebP.
_IMAGE_MAGIC = ((b"\x89PNG\r\n\x1a\n", "png"), (b"\xff\xd8", "jpg"), (b"RIFF", "webp"))


def dimensions(aspect: float, base: int) -> tuple[int, int]:
    """Width and height with the model's native pixel area, snapped to the latent grid."""
    width = math.sqrt(base * base * aspect)
    height = width / aspect

    def snap(value: float) -> int:
        return max(MIN_SIZE, round(value / SIZE_STEP) * SIZE_STEP)

    return snap(width), snap(height)


def checkpoint_workflow(values: dict[str, Any]) -> dict[str, Any]:
    """Text-to-image with a single checkpoint: works for SD 1.5 and SDXL models."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": values["%checkpoint%"]}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": values["%prompt%"], "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": values["%negative%"], "clip": ["1", 1]}},
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": values["%width%"], "height": values["%height%"], "batch_size": 1},
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": values["%seed%"],
                "steps": values["%steps%"],
                "cfg": values["%cfg%"],
                "sampler_name": values["%sampler%"],
                "scheduler": values["%scheduler%"],
                "denoise": 1.0,
            },
        },
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "talari"}},
    }


def with_reference(
    workflow: dict[str, Any],
    *,
    latent: str,
    sampler: str,
    vae: list[Any],
    image: str,
    width: int,
    height: int,
    denoise: float,
) -> dict[str, Any]:
    """A text-to-image workflow turned into image-to-image.

    The empty latent the sampler starts from is replaced by the reference picture: loaded,
    scaled and center-cropped to the picture's own shape (so a landscape photo becomes a
    portrait page picture without stretching), encoded with the workflow's own VAE. The
    sampler then starts from it at `denoise` — how far it may move — instead of from noise.
    Core ComfyUI nodes only; nothing to install.
    """
    out = {key: {**node, "inputs": dict(node["inputs"])} for key, node in workflow.items()}
    out["20"] = {"class_type": "LoadImage", "inputs": {"image": image}}
    out["21"] = {
        "class_type": "ImageScale",
        "inputs": {"image": ["20", 0], "upscale_method": "lanczos", "width": width, "height": height, "crop": "center"},
    }
    out[latent] = {"class_type": "VAEEncode", "inputs": {"pixels": ["21", 0], "vae": vae}}
    out[sampler]["inputs"]["denoise"] = denoise
    return out


def with_easycache(workflow: dict[str, Any], *, sampler: str, threshold: float) -> dict[str, Any]:
    """The sampler's model through EasyCache, which skips steps that would barely change the
    picture (built into ComfyUI). `threshold` is its reuse threshold; 0 leaves it out."""
    if threshold <= 0:
        return workflow
    out = {key: {**node, "inputs": dict(node["inputs"])} for key, node in workflow.items()}
    out["12"] = {
        "class_type": "EasyCache",
        "inputs": {"model": out[sampler]["inputs"]["model"], "reuse_threshold": threshold,
                   "start_percent": 0.15, "end_percent": 0.95, "verbose": False},
    }
    out[sampler]["inputs"]["model"] = ["12", 0]
    return out


def fill_workflow(node: Any, values: dict[str, Any]) -> Any:
    """Substitute tokens throughout a workflow loaded from a file."""
    if isinstance(node, dict):
        return {key: fill_workflow(value, values) for key, value in node.items()}
    if isinstance(node, list):
        return [fill_workflow(value, values) for value in node]
    if isinstance(node, str):
        if node in NUMERIC_TOKENS:
            return values[node]
        for token in TEXT_TOKENS:
            node = node.replace(token, str(values[token]))
    return node


def flux_workflow(values: dict[str, Any]) -> dict[str, Any]:
    """Text-to-image with FLUX.1 Dev loaded from a quantized GGUF file.

    Dev is guidance-distilled: the sampler runs at cfg 1 with an empty negative
    prompt, and the strength of the prompt is set by the FluxGuidance node.
    """
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": values["%unet%"]}},
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {"clip_name1": values["%t5%"], "clip_name2": values["%clip_l%"], "type": "flux"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": values["%vae%"]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": values["%prompt%"], "clip": ["2", 0]}},
        "5": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["4", 0], "guidance": values["%guidance%"]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
        "7": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": values["%width%"], "height": values["%height%"], "batch_size": 1},
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["7", 0],
                "seed": values["%seed%"],
                "steps": values["%steps%"],
                "cfg": 1.0,
                "sampler_name": values["%sampler%"],
                "scheduler": values["%scheduler%"],
                "denoise": 1.0,
            },
        },
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "talari"}},
    }


def _options(info: dict[str, Any], node: str, field: str) -> list[str]:
    """The choices ComfyUI offers for one input of a node, e.g. the model files it can see."""
    try:
        spec = info[node]["input"]["required"][field]
    except (KeyError, TypeError):
        return []
    if spec and isinstance(spec[0], list):
        return [name for name in spec[0] if isinstance(name, str)]
    # Newer ComfyUI describes a choice as ["COMBO", {"options": [...]}].
    if len(spec) > 1 and isinstance(spec[1], dict):
        return [name for name in spec[1].get("options", []) if isinstance(name, str)]
    return []


def _first_image(outputs: dict[str, Any]) -> dict[str, str] | None:
    for output in outputs.values():
        for image in output.get("images", []) if isinstance(output, dict) else []:
            if image.get("type") != "temp":
                return image
    return None


def _failure_message(entry: dict[str, Any]) -> str:
    for kind, detail in (entry.get("status") or {}).get("messages", []):
        if kind == "execution_error" and isinstance(detail, dict):
            node = detail.get("node_type") or "a node"
            return f"ComfyUI failed in {node}: {detail.get('exception_message', 'unknown error')}".strip()
    return "ComfyUI reported an error while drawing."


class ComfyUI:
    def __init__(self, settings: Settings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client

    @property
    def _custom_workflow(self) -> Path | None:
        configured = self._settings.comfyui_workflow
        if not configured:
            return None
        path = Path(configured)
        return path if path.is_absolute() else self._settings.data_dir / path

    async def _get(self, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return await self._client.get(f"{self._settings.comfyui_url}{path}", **kwargs)
        except httpx.HTTPError:
            raise NotConfigured(f"ComfyUI is not reachable at {self._settings.comfyui_url}.") from None

    async def _choices(self, node: str, field: str) -> list[str] | None:
        """Choices for a node input, or None when ComfyUI does not have that node at all."""
        response = await self._get(f"/object_info/{node}", timeout=10.0)
        info = response.json() if response.status_code == 200 else {}
        return _options(info, node, field) if node in info else None

    async def checkpoints(self) -> list[str]:
        return await self._choices("CheckpointLoaderSimple", "ckpt_name") or []

    async def _flux_problem(self) -> str | None:
        """Check that ComfyUI has the GGUF loader and every file FLUX needs."""
        settings = self._settings
        if not settings.flux_unet:
            return "FLUX_UNET_FILE is not set."
        unets = await self._choices("UnetLoaderGGUF", "unet_name")
        if unets is None:
            return "ComfyUI is missing the GGUF loader (the ComfyUI-GGUF custom node)."
        encoders = await self._choices("DualCLIPLoader", "clip_name1") or []
        vaes = await self._choices("VAELoader", "vae_name") or []
        wanted = [
            (settings.flux_unet, unets),
            (settings.flux_t5, encoders),
            (settings.flux_clip_l, encoders),
            (settings.flux_vae, vaes),
        ]
        missing = [name for name, available in wanted if name not in available]
        if missing:
            return f"ComfyUI does not have {', '.join(missing)} yet."
        return None

    async def _checkpoint(self) -> str:
        """The configured checkpoint, or the first one ComfyUI already has. Nothing is downloaded."""
        if self._settings.comfyui_checkpoint:
            return self._settings.comfyui_checkpoint
        names = await self.checkpoints()
        if not names:
            raise NotConfigured("ComfyUI has no checkpoint models installed.")
        return names[0]

    async def problem(self) -> str | None:
        """Why illustrations cannot be drawn, or None when ComfyUI is ready."""
        try:
            response = await self._get("/system_stats", timeout=3.0)
            if response.status_code != 200:
                return f"ComfyUI at {self._settings.comfyui_url} answered HTTP {response.status_code}."
            workflow = self._custom_workflow
            if workflow:
                return None if workflow.is_file() else f"The workflow file {workflow} does not exist."
            if self._settings.comfyui_pipeline == "flux":
                return await self._flux_problem()
            await self._checkpoint()
        except NotConfigured as error:
            return error.message
        return None

    async def _upload_reference(self, data: str) -> str:
        """Send a reference picture to ComfyUI's input folder; returns the name to load it by."""
        return await self._upload_picture(data, "reference picture", "talari-ref")

    async def _upload_picture(self, data: str, what: str, prefix: str) -> str:
        """Send a picture (base64 PNG, JPEG or WebP) to ComfyUI's input folder, under a fresh
        name starting with `prefix`; returns the name to load it by. `what` names it in errors."""
        try:
            raw = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError):
            raise ServiceError(f"The {what} could not be read.") from None
        ext = next((e for magic, e in _IMAGE_MAGIC if raw.startswith(magic)), None)
        if ext is None or (ext == "webp" and raw[8:12] != b"WEBP"):
            raise ServiceError(f"The {what} is not a PNG, JPEG or WebP.")
        try:
            uploaded = await self._client.post(
                f"{self._settings.comfyui_url}/upload/image",
                files={"image": (f"{prefix}-{uuid.uuid4().hex}.{ext}", raw, f"image/{'jpeg' if ext == 'jpg' else ext}")},
                data={"type": "input", "overwrite": "true"},
                timeout=60.0,
            )
        except httpx.HTTPError:
            raise NotConfigured(f"ComfyUI is not reachable at {self._settings.comfyui_url}.") from None
        if uploaded.status_code != 200:
            raise ServiceError(f"ComfyUI would not take the {what} (HTTP {uploaded.status_code}).")
        info = uploaded.json()
        name = info.get("name")
        if not name:
            raise ServiceError(f"ComfyUI did not say where it put the {what}.")
        return f"{info['subfolder']}/{name}" if info.get("subfolder") else name

    async def _workflow(self, request: IllustrationRequest, seed: int, reference: str | None = None) -> dict[str, Any]:
        settings = self._settings
        positive, negative = build_prompts(
            request.prompt,
            request.style_id,
            request.purpose,
            request.theme,
            custom_style=request.custom_style,
            palette=request.palette,
            detail=request.detail,
            art_style=request.art_style,
            medium=request.medium,
        )
        width, height = dimensions(request.aspect, settings.comfyui_base_size)
        custom = self._custom_workflow
        values: dict[str, Any] = {
            "%prompt%": positive,
            "%negative%": negative,
            "%seed%": seed,
            "%width%": width,
            "%height%": height,
            "%steps%": settings.comfyui_steps,
            "%cfg%": settings.comfyui_cfg,
            "%sampler%": settings.comfyui_sampler,
            "%scheduler%": settings.comfyui_scheduler,
            # A custom workflow names its own models, so only look one up when it asks.
            "%checkpoint%": settings.comfyui_checkpoint,
            "%image%": reference or "",
            "%denoise%": request.strength if reference else 1.0,
        }
        redraw = (
            {"image": reference, "width": width, "height": height, "denoise": request.strength}
            if reference else None
        )
        if custom is None and settings.comfyui_pipeline == "flux":
            values.update(
                {
                    "%unet%": settings.flux_unet,
                    "%t5%": settings.flux_t5,
                    "%clip_l%": settings.flux_clip_l,
                    "%vae%": settings.flux_vae,
                    "%guidance%": settings.flux_guidance,
                }
            )
            flow = with_easycache(flux_workflow(values), sampler="8", threshold=settings.flux_easycache)
            return with_reference(flow, latent="7", sampler="8", vae=["3", 0], **redraw) if redraw else flow
        if custom is None:
            values["%checkpoint%"] = await self._checkpoint()
            flow = checkpoint_workflow(values)
            return with_reference(flow, latent="4", sampler="5", vae=["1", 2], **redraw) if redraw else flow
        try:
            raw = custom.read_text()
            template = json.loads(raw)
        except (OSError, json.JSONDecodeError) as error:
            raise NotConfigured(f"The workflow file {custom} could not be read: {error}") from None
        if "%checkpoint%" in raw and not values["%checkpoint%"]:
            values["%checkpoint%"] = await self._checkpoint()
        return fill_workflow(template, values)

    async def generate(self, request: IllustrationRequest) -> tuple[bytes, int]:
        """Draw one picture. Returns the image bytes and the seed used."""
        seed = request.seed if request.seed is not None else random.randrange(2**32)
        reference = await self._upload_reference(request.reference) if request.reference else None
        workflow = await self._workflow(request, seed, reference)
        return await self.run(workflow), seed

    async def run(self, workflow: dict[str, Any]) -> bytes:
        """Queue a workflow, wait for it, and return the first picture it saved. Whatever
        happens, the GPU is handed back afterwards (_release)."""
        try:
            return await self._run(workflow)
        finally:
            await self._release()

    async def _release(self) -> None:
        """Ask ComfyUI to move its models off the GPU, into RAM, now the picture is done.

        The card is shared with the voice models in this API's own process. ComfyUI keeps
        the models it loaded cached in RAM (its RAM-pressure cache, which drops them when the
        machine runs short), so the next picture copies them back in seconds rather than
        reading them from disk; but on the card they would sit idle where a voice is about
        to be synthesized. Best effort: a ComfyUI that does not answer has nothing to free.
        """
        try:
            await self._client.post(f"{self._settings.comfyui_url}/free", json={"unload_models": True}, timeout=5.0)
        except httpx.HTTPError:
            pass

    async def _run(self, workflow: dict[str, Any]) -> bytes:
        settings = self._settings
        try:
            queued = await self._client.post(
                f"{settings.comfyui_url}/prompt",
                json={"prompt": workflow, "client_id": uuid.uuid4().hex},
                timeout=30.0,
            )
        except httpx.HTTPError:
            raise NotConfigured(f"ComfyUI is not reachable at {settings.comfyui_url}.") from None
        if queued.status_code != 200:
            try:
                reason = queued.json().get("error", {}).get("message") or queued.text[:300]
            except ValueError:
                reason = queued.text[:300]
            raise ServiceError(f"ComfyUI rejected the workflow: {reason}")
        prompt_id = queued.json().get("prompt_id")
        if not prompt_id:
            raise ServiceError("ComfyUI did not accept the job.")

        deadline = time.monotonic() + settings.comfyui_timeout_s
        while True:
            history = await self._get(f"/history/{prompt_id}", timeout=15.0)
            entry = history.json().get(prompt_id) if history.status_code == 200 else None
            if entry:
                if (entry.get("status") or {}).get("status_str") == "error":
                    raise ServiceError(_failure_message(entry))
                image = _first_image(entry.get("outputs") or {})
                if image:
                    break
                if (entry.get("status") or {}).get("completed"):
                    raise ServiceError("ComfyUI finished without producing an image.")
            if time.monotonic() > deadline:
                raise ServiceError(f"ComfyUI did not finish within {settings.comfyui_timeout_s:.0f} seconds.")
            await asyncio.sleep(POLL_INTERVAL_S)

        picture = await self._get(
            "/view",
            params={
                "filename": image["filename"],
                "subfolder": image.get("subfolder", ""),
                "type": image.get("type", "output"),
            },
            timeout=60.0,
        )
        if picture.status_code != 200 or not picture.content:
            raise ServiceError("ComfyUI drew the image but it could not be downloaded.")
        return picture.content

    # --- the lab ------------------------------------------------------------------------
    # The styleguide's ComfyUI test page: every input passed through as given (LabRequest).

    async def lab_options(self) -> LabOptions:
        """The model files, samplers, schedulers and style files this ComfyUI has."""
        settings = self._settings
        models = [
            *(await self._choices("UnetLoaderGGUF", "unet_name") or []),
            *(await self._choices("UNETLoader", "unet_name") or []),
            *(await self.checkpoints()),
        ]
        default = settings.flux_unet if settings.comfyui_pipeline == "flux" else (settings.comfyui_checkpoint or "")
        missing = await self._uso_missing()
        return LabOptions(
            models=models,
            default_model=default if default in models else (models[0] if models else ""),
            samplers=await self._choices("KSampler", "sampler_name") or [],
            schedulers=await self._choices("KSampler", "scheduler") or [],
            style_ready=not missing,
            style_missing=missing,
            defaults={
                "steps": settings.comfyui_steps,
                "cfg": 1.0 if settings.comfyui_pipeline == "flux" else settings.comfyui_cfg,
                "guidance": settings.flux_guidance,
                "sampler_name": settings.comfyui_sampler,
                "scheduler": settings.comfyui_scheduler,
                "width": 1024,
                "height": 1024,
                "denoise": 1.0,
                "style_image_weight": 1.0,
                "easycache": settings.flux_easycache,
            },
        )

    async def lab(self, request: LabRequest) -> tuple[bytes, int, int, int, dict[str, Any]]:
        """Draw one picture from a lab request. Returns the picture, the seed, the size it
        was drawn at, and the workflow ComfyUI was given."""
        settings = self._settings
        seed = request.seed if request.seed is not None else random.randrange(2**32)
        model = request.model or settings.flux_unet
        checkpoints = await self.checkpoints()
        if request.style_image:
            missing = await self._uso_missing()
            if missing:
                raise NotConfigured(f"A style picture needs {' and '.join(missing)} in ComfyUI, which it does not have yet.")
            if model in checkpoints:
                raise ServiceError("A style picture works with FLUX (USO); pick a FLUX model for it.", status=422)
        image = await self._upload_picture(request.image, "picture", "lab-image") if request.image else ""
        mask = await self._upload_picture(request.mask, "mask", "lab-mask") if request.mask else ""
        style = await self._upload_picture(request.style_image, "style picture", "lab-style") if request.style_image else ""
        width, height = _snap16(request.width), _snap16(request.height)
        workflow = lab_workflow(
            request,
            seed=seed,
            width=width,
            height=height,
            loader="checkpoint" if model in checkpoints else "gguf" if model.endswith(".gguf") else "unet",
            model=model,
            settings=settings,
            image=image,
            mask=mask,
            style_image=style,
        )
        return await self.run(workflow), seed, width, height, workflow

    async def _uso_missing(self) -> list[str]:
        """What a style picture needs and ComfyUI does not have: the files below, in the folders
        Phansora's ComfyUI container fetches them into (docker/comfyui/fetch_models.py)."""
        have = {
            USO_LORA: await self._choices("LoraLoaderModelOnly", "lora_name") or [],
            USO_PROJECTOR: await self._choices("ModelPatchLoader", "name") or [],
            USO_VISION: await self._choices("CLIPVisionLoader", "clip_name") or [],
        }
        return [name for name, available in have.items() if name not in available]


def _snap16(value: int) -> int:
    """FLUX and SD3 latents come in 16-pixel steps."""
    return max(256, round(value / 16) * 16)


def lab_workflow(
    request: LabRequest,
    *,
    seed: int,
    width: int,
    height: int,
    loader: str,
    model: str,
    settings: Settings,
    image: str = "",
    mask: str = "",
    style_image: str = "",
) -> dict[str, Any]:
    """The lab's workflow: one node per input of LabRequest, core ComfyUI nodes only (plus the
    GGUF loader for a .gguf model).

    `loader` is how the model is loaded: "gguf" or "unet" (a diffusion model, with FLUX's text
    encoders, VAE and FluxGuidance) or "checkpoint" (model, text encoder and VAE in one file,
    as SD 1.5 and SDXL come). `image`, `mask` and `style_image` are upload names, or empty. A
    style picture is ComfyUI's own USO style reference (its "Flux.1 Dev USO Reference Image
    Generation" template, style only): USO's LoRA on the model, then USOStyleReference, which
    adds the picture as the vision model reads it — its style, not its subject.
    """
    wf: dict[str, Any] = {}
    if loader == "checkpoint":
        wf["1"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model}}
        model_out, clip, vae = ["1", 0], ["1", 1], ["1", 2]
    else:
        wf["1"] = (
            {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": model}}
            if loader == "gguf"
            else {"class_type": "UNETLoader", "inputs": {"unet_name": model, "weight_dtype": "default"}}
        )
        wf["2"] = {
            "class_type": "DualCLIPLoader",
            "inputs": {"clip_name1": settings.flux_t5, "clip_name2": settings.flux_clip_l, "type": "flux"},
        }
        wf["3"] = {"class_type": "VAELoader", "inputs": {"vae_name": settings.flux_vae}}
        model_out, clip, vae = ["1", 0], ["2", 0], ["3", 0]

    # The positive conditioning; the style picture then acts on it. FLUX with clip_l / t5xxl:
    # one text per encoder, guidance included (CLIPTextEncodeFlux). Otherwise one text for
    # both (CLIPTextEncode), and for FLUX its guidance after (FluxGuidance).
    if loader != "checkpoint" and (request.clip_l.strip() or request.t5xxl.strip()):
        wf["4"] = {
            "class_type": "CLIPTextEncodeFlux",
            "inputs": {"clip": clip, "clip_l": request.clip_l, "t5xxl": request.t5xxl, "guidance": request.guidance},
        }
        positive = ["4", 0]
    else:
        text = request.prompt or request.t5xxl or request.clip_l
        wf["4"] = {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": clip}}
        positive = ["4", 0]
        if loader != "checkpoint":
            wf["5"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": positive, "guidance": request.guidance}}
            positive = ["5", 0]
    if style_image:
        wf["30"] = {"class_type": "LoadImage", "inputs": {"image": style_image}}
        wf["31"] = {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": USO_VISION}}
        wf["32"] = {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["31", 0], "image": ["30", 0], "crop": "center"}}
        wf["33"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": model_out, "lora_name": USO_LORA, "strength_model": request.style_image_weight},
        }
        wf["34"] = {"class_type": "ModelPatchLoader", "inputs": {"name": USO_PROJECTOR}}
        wf["35"] = {
            "class_type": "USOStyleReference",
            "inputs": {"model": ["33", 0], "model_patch": ["34", 0], "clip_vision_output": ["32", 0]},
        }
        model_out = ["35", 0]
    # EasyCache, last on the model, so it caches the steps of whatever the model has become.
    if request.easycache > 0:
        wf["36"] = {
            "class_type": "EasyCache",
            "inputs": {"model": model_out, "reuse_threshold": request.easycache, "start_percent": 0.15, "end_percent": 0.95, "verbose": False},
        }
        model_out = ["36", 0]
    # negative_prompt → the negative conditioning. It only acts when cfg is above 1.
    wf["6"] = {"class_type": "CLIPTextEncode", "inputs": {"text": request.negative_prompt, "clip": clip}}

    # The canvas: the source picture, scaled and center-cropped to the size, or an empty one.
    if image:
        wf["20"] = {"class_type": "LoadImage", "inputs": {"image": image}}
        wf["21"] = {
            "class_type": "ImageScale",
            "inputs": {"image": ["20", 0], "upscale_method": "lanczos", "width": width, "height": height, "crop": "center"},
        }
        wf["7"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["21", 0], "vae": vae}}
        latent = ["7", 0]
        if mask:
            # White is where the picture may change. Scaled and cropped exactly as the picture.
            wf["22"] = {"class_type": "LoadImageMask", "inputs": {"image": mask, "channel": "red"}}
            wf["23"] = {"class_type": "MaskToImage", "inputs": {"mask": ["22", 0]}}
            wf["24"] = {
                "class_type": "ImageScale",
                "inputs": {"image": ["23", 0], "upscale_method": "bilinear", "width": width, "height": height, "crop": "center"},
            }
            wf["25"] = {"class_type": "ImageToMask", "inputs": {"image": ["24", 0], "channel": "red"}}
            wf["26"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": latent, "mask": ["25", 0]}}
            latent = ["26", 0]
    else:
        empty = "EmptyLatentImage" if loader == "checkpoint" else "EmptySD3LatentImage"
        wf["7"] = {"class_type": empty, "inputs": {"width": width, "height": height, "batch_size": 1}}
        latent = ["7", 0]

    wf["8"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_out,
            "positive": positive,
            "negative": ["6", 0],
            "latent_image": latent,
            "seed": seed,
            "steps": request.steps,
            "cfg": request.cfg,
            "sampler_name": request.sampler_name,
            "scheduler": request.scheduler,
            "denoise": request.denoise,
        },
    }
    wf["9"] = {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": vae}}
    wf["10"] = {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "lab"}}
    return wf
