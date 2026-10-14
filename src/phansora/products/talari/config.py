"""Talari's settings, from the environment.

The ComfyUI half is the same set of names the stack's Compose file uses, so one .env
line (FLUX_QUANT=Q4_K_S, say) means the same thing to the container that holds the
model and to the API that asks for it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from phansora.shared.paths import runtime_dir


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    return float(raw) if raw else default


def _path(name: str) -> Path | None:
    raw = os.getenv(name, "").strip()
    return Path(raw) if raw else None


@dataclass(frozen=True)
class Settings:
    comfyui_url: str
    comfyui_checkpoint: str
    comfyui_workflow: str
    comfyui_base_size: int
    comfyui_steps: int
    comfyui_cfg: float
    comfyui_sampler: str
    comfyui_scheduler: str
    comfyui_timeout_s: float
    # Talari's own folder under the API's runtime root: pictures, and the place a
    # relative COMFYUI_WORKFLOW path is read from.
    data_dir: Path
    # "flux" (FLUX.1 Dev as a quantized GGUF — what the bundled container runs) or
    # "checkpoint" (any single checkpoint file a ComfyUI of your own has).
    comfyui_pipeline: str = "flux"
    flux_unet: str = ""
    flux_clip_l: str = "clip_l.safetensors"
    flux_t5: str = "t5xxl_fp8_e4m3fn.safetensors"
    flux_vae: str = "ae.safetensors"
    flux_guidance: float = 3.5
    # Progress file the ComfyUI container writes while it downloads its models; read so
    # the status can say "downloading, 43%" rather than "not reachable".
    model_status_file: Path | None = None

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"


def load_settings() -> Settings:
    pipeline = os.getenv("COMFYUI_PIPELINE", "").strip().lower() or "flux"
    # FLUX.1 Dev is guidance-distilled: it wants fewer steps and its own sampler pairing.
    flux = pipeline == "flux"
    quant = os.getenv("FLUX_QUANT", "").strip() or "Q4_K_S"
    return Settings(
        comfyui_url=(os.getenv("COMFYUI_URL", "").strip() or "http://127.0.0.1:8188").rstrip("/"),
        comfyui_checkpoint=os.getenv("COMFYUI_CHECKPOINT", "").strip(),
        comfyui_workflow=os.getenv("COMFYUI_WORKFLOW", "").strip(),
        comfyui_base_size=_int("FLUX_SIZE", 1024),
        comfyui_steps=_int("FLUX_STEPS", 24 if flux else 28),
        comfyui_cfg=_float("COMFYUI_CFG", 6.5),
        comfyui_sampler=os.getenv("COMFYUI_SAMPLER", "").strip() or ("euler" if flux else "dpmpp_2m"),
        comfyui_scheduler=os.getenv("COMFYUI_SCHEDULER", "").strip() or ("simple" if flux else "karras"),
        comfyui_timeout_s=_float("FLUX_TIMEOUT_SECONDS", 600.0 if flux else 300.0),
        data_dir=runtime_dir("talari"),
        comfyui_pipeline=pipeline,
        flux_unet=os.getenv("FLUX_UNET_FILE", "").strip() or f"flux1-dev-{quant}.gguf",
        flux_clip_l=os.getenv("FLUX_CLIP_L_FILE", "").strip() or "clip_l.safetensors",
        flux_t5=os.getenv("FLUX_T5_FILE", "").strip() or "t5xxl_fp8_e4m3fn.safetensors",
        flux_vae=os.getenv("FLUX_VAE_FILE", "").strip() or "ae.safetensors",
        flux_guidance=_float("FLUX_GUIDANCE", 3.5),
        model_status_file=_path("COMFYUI_MODEL_STATUS_FILE"),
    )
