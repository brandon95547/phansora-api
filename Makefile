# Phansora API — developer tasks
.PHONY: help install install-tts install-mac dev run worker test compile docker-build docker-up docker-down clean

VENV   ?= .venv
# Where the CosyVoice2 TTS engine checkout lives (git clone, not a pip package).
COSYVOICE_REPO ?= /var/www/CosyVoice
# Prod needs Python 3.10 — the torch wheels in requirements.txt are cp310, and
# CentOS Stream 8's default `python3` is 3.6 (which fails with "not a supported
# wheel"). `install` builds the venv with 3.10 via uv (see below).
PYTHON ?= python3.10
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip
HOST   ?= 0.0.0.0
PORT   ?= 8000

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Create Python 3.10 venv and install deps (CUDA torch — prod / Linux GPU)
	rm -rf $(VENV)
	@if command -v uv >/dev/null 2>&1; then \
		echo "==> creating $(VENV) with uv (Python 3.10)"; \
		uv venv --python 3.10 --seed $(VENV); \
	elif command -v $(PYTHON) >/dev/null 2>&1; then \
		echo "==> creating $(VENV) with $(PYTHON)"; \
		$(PYTHON) -m venv $(VENV); \
	else \
		echo "ERROR: need Python 3.10. Install uv (curl -LsSf https://astral.sh/uv/install.sh | sh)"; \
		echo "       then 'uv python install 3.10', or install a system python3.10 and re-run."; \
		exit 1; \
	fi
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

install-mac: ## macOS local dev: venv + CPU/MPS torch + deps (no CUDA)
	python3 -m venv $(VENV)
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

docker-build: ## Build the production image
	docker compose build

docker-up: ## Start the stack
	docker compose up -d

docker-down: ## Stop the stack
	docker compose down

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache
