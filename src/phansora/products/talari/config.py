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
    # How far the sampler goes when a picture is drawn from nothing (no reference): its
    # denoise. A redraw from a reference uses the strength it is asked for instead.
    comfyui_denoise: float = 1.0
    # EasyCache (built into ComfyUI): skips sampling steps that would barely change the picture.
    # Its reuse threshold — higher skips more; 0 switches it off. 0.2 is ComfyUI's default:
    # about 40% faster at 1024px on the A4000, with slightly grainier flat areas.
    flux_easycache: float = 0.0
    # Progress file the ComfyUI container writes while it downloads its models; read so
    # the status can say "downloading, 43%" rather than "not reachable".
    model_status_file: Path | None = None
    # User ids the ComfyUI lab (/users/{id}/lab/...) answers for, besides the app itself.
    # The lab runs whatever settings it is given on the GPU and is not metered, so it is
    # open only to the accounts named here (TALARI_LAB_USERS=1,5); empty, to nobody.
    lab_users: frozenset[str] = frozenset()

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
        # FLUX draws at about this size squared, in the picture's shape: 768 (0.6 megapixels)
        # draws in about half the time of 1024 and still well; the app enlarges a picture for
        # print when it is exported (up to 4×, 4x-UltraSharp), which covers any page at 200 DPI.
        comfyui_base_size=_int("FLUX_SIZE", 768 if flux else 1024),
        # 16 for every picture, redraws included (the user's choice, 2026-10-19): a fifth
        # faster than ComfyUI's own FLUX templates' 20.
        comfyui_steps=_int("FLUX_STEPS", 16 if flux else 28),
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
        # 0.85 for a picture drawn from nothing (the user's choice, 2026-10-19); 1.0 draws from
        # pure noise. A remix keeps its own strength.
        comfyui_denoise=min(1.0, max(0.05, _float("FLUX_DENOISE", 0.85 if flux else 1.0))),
        flux_easycache=_float("FLUX_EASYCACHE", 0.2),
        model_status_file=_path("COMFYUI_MODEL_STATUS_FILE"),
        lab_users=frozenset(u.strip() for u in os.getenv("TALARI_LAB_USERS", "").split(",") if u.strip()),
    )
