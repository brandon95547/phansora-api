#!/usr/bin/env bash
#
# ⚠ LEGACY — NO LONGER REQUIRED by the CosyVoice2 TTS engine. CosyVoice2 runs on vLLM +
# TensorRT, whose CUDA ops ship precompiled in their wheels — the box needs only the NVIDIA
# driver, not the toolkit/nvcc. This script (and deploy/.../cuda-env.conf) existed for the
# old IndexTTS2 + DeepSpeed path. Kept for reference; safe to delete if no other engine
# needs nvcc.
#
# Install the CUDA 12.6 TOOLKIT ONLY (nvcc + headers) to /usr/local/cuda-12.6 on the
# RHEL-family prod GPU box. Needed so IndexTTS2's DeepSpeed kernel-inject
# (INDEXTTS2_USE_DEEPSPEED=1) and the BigVGAN CUDA kernel (INDEXTTS2_USE_CUDA_KERNEL=1)
# can JIT-compile their fused CUDA ops at model-load time.
#
# This NEVER installs `cuda-drivers` — the working NVIDIA driver is left untouched. Only
# the `cuda-toolkit-12-6` meta-package is used (the `cuda` / `cuda-12-6` meta-packages
# would pull the driver — do not use those).
#
# Idempotent: safe to re-run. Run as root on prod:
#     bash scripts/install-cuda-toolkit.sh
#
# After it succeeds, wire CUDA_HOME into the phansora-api service:
#     cp deploy/phansora-api.service.d/cuda-env.conf /etc/systemd/system/phansora-api.service.d/
#     systemctl daemon-reload && systemctl restart phansora-api
#
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "ERROR: run as root (dnf needs it)." >&2
  exit 1
fi

# Detect the RHEL major version -> rhel8 / rhel9 repo.
. /etc/os-release
rel="rhel${VERSION_ID%%.*}"
repo_url="https://developer.download.nvidia.com/compute/cuda/repos/${rel}/x86_64/cuda-${rel}.repo"
echo "==> Detected ${PRETTY_NAME:-$rel}; using CUDA repo: ${repo_url}"

# `dnf config-manager` lives in dnf-plugins-core.
dnf install -y dnf-plugins-core

# Add the NVIDIA CUDA repo (config-manager is a no-op if it's already present).
dnf config-manager --add-repo "${repo_url}"

# Toolkit only — NOT the driver.
dnf install -y cuda-toolkit-12-6

# torch 2.8 needs GCC >= 9 to compile its C++/CUDA extensions; RHEL8's default is 8.5.
# Install gcc-toolset-12 (used ONLY to build the DeepSpeed / BigVGAN kernels — the CUDA_HOME
# drop-in points CC/CXX at it). Skipped automatically when the system gcc is already >= 9.
if gcc -dumpversion 2>/dev/null | awk -F. '{exit ($1>=9)?0:1}'; then
  echo "==> System GCC $(gcc -dumpversion) is >= 9; skipping gcc-toolset."
else
  echo "==> System GCC $(gcc -dumpversion 2>/dev/null || echo '?') < 9; installing gcc-toolset-12."
  dnf install -y gcc-toolset-12
  /opt/rh/gcc-toolset-12/root/usr/bin/gcc --version | head -1
fi

echo "==> Installed. nvcc:"
/usr/local/cuda-12.6/bin/nvcc --version

cat <<'EOF'

==> Next steps:
    cp deploy/phansora-api.service.d/cuda-env.conf /etc/systemd/system/phansora-api.service.d/
    systemctl daemon-reload
    # then set in prod .env:  INDEXTTS2_USE_DEEPSPEED=1  INDEXTTS2_USE_CUDA_KERNEL=1
    #                         INDEXTTS2_DEFAULT_REF=/path/to/ref.wav
    systemctl restart phansora-api
    journalctl -u phansora-api -f   # watch for a clean load (no "Falling back" warnings)
EOF
