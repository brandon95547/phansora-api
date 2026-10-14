"""ComfyUI client: queue a workflow, wait for it, fetch the picture.

ComfyUI runs in its own container (Phansora's docker-compose.prod.yml, service
"comfyui"), so it has its own CUDA context. That is why nothing here goes through
``phansora.shared.gpu``: that gate protects CUDA work inside THIS process while the
voice engine captures graphs, and ComfyUI's work never happens in this process.
Sharing the card's memory is handled on ComfyUI's side (``--cache-none --reserve-vram``).
"""

from __future__ import annotations

import asyncio
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
from .schemas import IllustrationRequest
from .styles import build_prompts

POLL_INTERVAL_S = 1.0
SIZE_STEP = 64
MIN_SIZE = 256

# Values a custom workflow file may ask for. A string that is exactly one of
# the numeric tokens becomes a number; the text tokens are replaced in place.
NUMERIC_TOKENS = ("%seed%", "%width%", "%height%", "%steps%", "%cfg%")
TEXT_TOKENS = ("%prompt%", "%negative%", "%checkpoint%", "%sampler%", "%scheduler%")


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

    async def _workflow(self, request: IllustrationRequest, seed: int) -> dict[str, Any]:
        settings = self._settings
        positive, negative = build_prompts(request.prompt, request.style_id, request.purpose, request.theme)
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
        }
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
            return flux_workflow(values)
        if custom is None:
            values["%checkpoint%"] = await self._checkpoint()
            return checkpoint_workflow(values)
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
        settings = self._settings
        seed = request.seed if request.seed is not None else random.randrange(2**32)
        workflow = await self._workflow(request, seed)

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
        return picture.content, seed
