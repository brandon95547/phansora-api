# Phansora API — developer tasks
.PHONY: help install install-mac dev run worker test compile docker-build docker-up docker-down clean

VENV   ?= .venv
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
		uv venv --python 3.10 $(VENV); \
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

install-mac: ## macOS local dev: venv + CPU/MPS torch + deps (no CUDA)
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install torch torchvision torchaudio
	grep -vE '^(--extra-index-url|torch|torchvision|torchaudio)' requirements.txt > $(VENV)/requirements-mac.txt
	$(PIP) install -r $(VENV)/requirements-mac.txt
	$(PIP) install -e .
	@echo ""
	@echo "API deps installed. For actual TTS also install the GPT-SoVITS engine"
	@echo "(brew install ffmpeg; clone + checkpoints) — see README 'Local dev on Mac'."

dev: ## Run the unified API with autoreload
	$(PY) -m uvicorn phansora.main:app --host $(HOST) --port $(PORT) --reload

run: ## Run the unified API (no reload)
	$(PY) -m uvicorn phansora.main:app --host $(HOST) --port $(PORT) --workers 2

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
