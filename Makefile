# Phansora API — developer tasks
.PHONY: help install install-dev install-tts install-mac dev run worker test compile clean

VENV   ?= .venv
# Where the CosyVoice2 TTS engine checkout lives (git clone, not a pip package).
COSYVOICE_REPO ?= /var/www/CosyVoice
# Prod needs Python 3.10 — the torch wheels in requirements.txt are cp310, and
# CentOS Stream 8's default `python3` is 3.6 (which fails with "not a supported
# wheel"). `install` builds the venv with 3.10 via uv (see below).
PYTHON ?= python3.10
# Local dev (Linux + macOS) runs 3.11 — and does NOT have to match prod's 3.10. The cp310
# pins in requirements.txt apply only to `install` (prod/CUDA); `install-dev` and
# `install-mac` fetch CPU torch by index, which resolves a wheel per interpreter, so dev is
# free to sit on 3.11. Homebrew often ships 3.11 as plain `python3`, so both names are tried.
DEV_PYTHON ?= python3.11
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip
HOST   ?= 0.0.0.0
PORT   ?= 8000

# All runtime data (voices, generated audio/text, Book Alchemy, caches) lives under
# ./data in dev, mirroring prod's PHANSORA_DATA_DIR=<repo>/data. Exported to every
# recipe so the API and workers share one root instead of scattering dirs at the repo root.
PHANSORA_DATA_DIR ?= $(CURDIR)/data
export PHANSORA_DATA_DIR

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Create Python 3.10 venv and install deps (CUDA torch — prod / Linux GPU)
	@# The interpreter is validated BEFORE anything is deleted. This target used to
	@# `rm -rf` first and only then discover it had the wrong Python, which cost you a
	@# working venv to learn that. It also used to fall back to $(PYTHON) without
	@# checking its version, so a 3.11 venv got built and pip then rejected the cp310
	@# torch wheels with "not a supported wheel on this platform" — an error that blames
	@# the wheel for what is really the wrong interpreter.
	@set -e; \
	if command -v uv >/dev/null 2>&1; then \
		CREATE="uv venv --python 3.10 --seed $(VENV)"; WITH="uv (Python 3.10)"; \
	elif command -v $(PYTHON) >/dev/null 2>&1 \
	     && $(PYTHON) -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 10) else 1)' 2>/dev/null; then \
		CREATE="$(PYTHON) -m venv $(VENV)"; WITH="$(PYTHON)"; \
	else \
		echo "ERROR: this target needs Python 3.10."; \
		echo "       requirements.txt pins torch/torchvision/torchaudio as cp310 wheels."; \
		echo "       pip rejects those on any other version, reporting it as an"; \
		echo "       unsupported *wheel* rather than an unsupported *interpreter*."; \
		printf "       found: "; \
		if command -v $(PYTHON) >/dev/null 2>&1; then $(PYTHON) -V; else echo "no $(PYTHON) on PATH"; fi; \
		echo ""; \
		echo "  Fix, pick one:"; \
		echo "    curl -LsSf https://astral.sh/uv/install.sh | sh   # then re-run; uv supplies 3.10"; \
		echo "    install a system python3.10, then: make install PYTHON=python3.10"; \
		echo "    make install-dev                                  # local dev, no GPU/TTS needed"; \
		exit 1; \
	fi; \
	echo "==> creating $(VENV) with $$WITH"; \
	rm -rf $(VENV); \
	$$CREATE
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo ""
	@echo "API installed. Now install the TTS engine:  make install-tts"

install-tts: ## Clone CosyVoice2, install its reqs (torch-stripped) + download the model
	@test -d $(COSYVOICE_REPO)/.git || \
		git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git $(COSYVOICE_REPO)
	cd $(COSYVOICE_REPO) && git submodule update --init --recursive
	# CosyVoice's requirements pin torch==2.3.1 / torchaudio==2.3.1 / pydantic==2.7.0, which
	# would fight vLLM 0.9.0 (torch 2.7, pydantic>=2.9). A constraints file can't override an
	# explicit '==' pin, so STRIP those lines and re-assert our versions on the command line.
	# (If pip then errors on deepspeed/lightning capping torch, add them to the sed list —
	# they are training-only and unused at inference.)
	#
	# ALSO strip openai-whisper==20231117: it caps triton<3, but torch 2.7 AND vLLM both need
	# triton==3.3.0 — an unsolvable clash in one resolution. CosyVoice only calls
	# whisper.log_mel_spectrogram (no triton kernels), so we install whisper separately with
	# --no-deps (keeps triton 3.3.0). Its legacy setup.py imports pkg_resources at build, so
	# pin setuptools<81 + --no-build-isolation. more-itertools is a whisper runtime dep that
	# --no-deps skips (numba/tiktoken already come from librosa / the base env).
	$(PIP) install "setuptools<81" wheel
	sed -E '/^(torch|torchaudio|pydantic|openai-whisper)==/d' $(COSYVOICE_REPO)/requirements.txt > $(VENV)/cosy-reqs.txt
	$(PIP) install torch==2.7.0 torchaudio==2.7.0 "pydantic>=2.9" -r $(VENV)/cosy-reqs.txt
	$(PIP) install --no-deps --no-build-isolation openai-whisper==20231117
	$(PIP) install more-itertools
	$(PY) -c "from modelscope import snapshot_download; snapshot_download('iic/CosyVoice2-0.5B', local_dir='$(COSYVOICE_REPO)/pretrained_models/CosyVoice2-0.5B')"
	$(PY) -c "import torch, torchaudio, vllm; print('OK torch', torch.__version__, 'vllm', vllm.__version__, 'cuda', torch.cuda.is_available())"
	@echo ""
	@echo "Set COSYVOICE2_REPO=$(COSYVOICE_REPO) in .env (+ COSYVOICE2_DEFAULT_REF / _REF_TEXT)."

install-dev: ## Linux local dev: venv + CPU torch + deps (no CUDA, no vLLM — so no TTS)
	@# Everything except the GPU stack. Nothing installed here is pinned to a specific
	@# CPython, so it works on whatever python3 the machine has (3.11 / 3.12) instead of
	@# demanding 3.10 like `install` does.
	@#
	@# What you lose: vLLM and therefore CosyVoice2 synthesis. main.py mounts products
	@# defensively (_load_products skips any that fail to import), so the API still comes
	@# up with /studio, /chrono, /dossier and /book-alchemy — it just logs a warning and
	@# omits /spokenverse. Narration authoring works; voicing a script does not.
	@# Interpreter resolved and validated BEFORE the venv is destroyed.
	@set -e; \
	for c in $(DEV_PYTHON) python3; do \
		if command -v $$c >/dev/null 2>&1 \
		   && $$c -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)' 2>/dev/null; then \
			echo "==> creating $(VENV) with $$c ($$($$c -V 2>&1))"; \
			rm -rf $(VENV); $$c -m venv $(VENV); exit 0; \
		fi; \
	done; \
	echo "ERROR: local dev needs Python 3.11 (prod's 3.10 is only for \`make install\`)."; \
	printf "       tried: "; for c in $(DEV_PYTHON) python3; do \
		printf "%s=%s " "$$c" "$$(command -v $$c >/dev/null 2>&1 && $$c -V 2>&1 | cut -d' ' -f2 || echo absent)"; \
	done; echo ""; \
	echo "  Fix: install python3.11, or override:  make install-dev DEV_PYTHON=python3.x"; \
	exit 1
	$(PIP) install --upgrade pip
	@# CPU wheels by INDEX, not by pinned URL: pip resolves the right build for whichever
	@# interpreter is present, which is exactly the brittleness that breaks `install`.
	$(PIP) install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
	grep -vE '^(torch|torchvision|torchaudio|vllm)[[:space:]]*[@=]' requirements.txt > $(VENV)/requirements-dev.txt
	$(PIP) install -r $(VENV)/requirements-dev.txt
	$(PIP) install -e .
	@echo ""
	@echo "Dev deps installed (CPU torch, no vLLM). SpokenVerse/TTS will be skipped at boot."
	@echo "Run: make dev"

install-mac: ## macOS local dev: venv + CPU/MPS torch + deps (no CUDA)
	@# Same 3.11 rule as install-dev — Homebrew commonly exposes it as plain `python3`,
	@# so both names are tried. Validated before anything is removed.
	@set -e; \
	for c in $(DEV_PYTHON) python3; do \
		if command -v $$c >/dev/null 2>&1 \
		   && $$c -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)' 2>/dev/null; then \
			echo "==> creating $(VENV) with $$c ($$($$c -V 2>&1))"; \
			rm -rf $(VENV); $$c -m venv $(VENV); exit 0; \
		fi; \
	done; \
	echo "ERROR: macOS dev needs Python 3.11 (brew install python@3.11)."; \
	printf "       tried: "; for c in $(DEV_PYTHON) python3; do \
		printf "%s=%s " "$$c" "$$(command -v $$c >/dev/null 2>&1 && $$c -V 2>&1 | cut -d' ' -f2 || echo absent)"; \
	done; echo ""; \
	echo "  Or override:  make install-mac DEV_PYTHON=python3.x"; \
	exit 1
	$(PIP) install --upgrade pip
	$(PIP) install torch torchvision torchaudio
	grep -vE '^(--extra-index-url|torch|torchvision|torchaudio)' requirements.txt > $(VENV)/requirements-mac.txt
	$(PIP) install -r $(VENV)/requirements-mac.txt
	$(PIP) install -e .
	@echo ""
	@echo "API deps installed. Note: CosyVoice2 (vLLM) is CUDA-only — Mac dev runs the API"
	@echo "without TTS, or set COSYVOICE2_USE_VLLM=0/_USE_TRT=0 for a slow CPU path."

dev: ## Run the unified API with autoreload
	$(PY) -m uvicorn phansora.main:app --host $(HOST) --port $(PORT) --reload

run: ## Run the unified API (no reload)
	# --workers 1: CosyVoice2 is a per-process singleton (weights + a resident vLLM engine).
	# Each extra worker would load its OWN copy — doubling VRAM and the ~80s startup warmup —
	# so the GPU model must live in a single warm process. Scale non-TTS load via a proxy/
	# replicas, not in-process workers, if ever needed.
	$(PY) -m uvicorn phansora.main:app --host $(HOST) --port $(PORT) --workers 1

worker: ## Run the SpokenVerse / Book Alchemy durable worker
	$(PY) -m phansora.products.spokenverse.worker

test: ## Run the test suite
	$(PY) -m pytest -q

compile: ## Byte-compile every module (fast syntax check)
	$(PY) -m py_compile $$(find src -name '*.py')

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache
