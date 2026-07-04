# Phansora API

The unified Python backend for [phansora.com](https://phansora.com) — a platform
that hosts multiple AI products behind a single FastAPI application. It
consolidates three formerly separate services:

| Product | Path prefix | What it does |
|---|---|---|
| **SpokenVerse** | `/spokenverse` | PDF→text (render + OCR + AI cleanup), text→audio (GPT-SoVITS voice cloning), audio→text (faster-whisper), and **Book Alchemy** (durable, Postgres-backed book→audio-course pipeline) |
| **Chrono-Origin** | `/chrono` | Traces the earliest known origin of a story/myth/event and maps its evolution, using Claude's grounded web search |
| **Dossier Nova** | `/dossier` | AI research & dossier generation — multi-source ingest → cleanup → profiling → TOC → organized, source-attributed dossier (local embeddings + DeepSeek) |

## Layout

```
phansora-api/
├── src/phansora/
│   ├── main.py                 # unified FastAPI app — mounts each product
│   ├── config.py               # platform-level settings
│   ├── cli.py                  # `phansora serve|tts|dossier`
│   ├── shared/                 # cross-cutting infra (product -> shared only)
│   │   ├── ai/                 #   anthropic + deepseek clients
│   │   ├── auth/  billing/     #   platform scaffolds (Square credit system)
│   │   ├── database/           #   generic asyncpg pool
│   │   ├── storage/  queue/    #   scaffolds
│   │   └── utils/              #   chunking, ffmpeg, naming, email
│   └── products/
│       ├── spokenverse/        # services/, txt_to_voice/, book_alchemy/, worker.py
│       ├── chrono_origin/      # pipeline/, services/, models.py
│       ├── dossier_nova/       # embeddings, organizer, toc, validation, ...
│       ├── image_tools/        # planned product (scaffold)
│       └── future_products/    # namespace for what's next
├── tests/  docs/  scripts/  examples/  assets/  data/
├── requirements.txt  pyproject.toml  Makefile
├── Dockerfile  docker-compose.yml
├── .env.example  .gitignore  README.md  LICENSE
```

**Design rule:** each product owns its API routes, business logic, prompts,
models and config. Reusable infrastructure lives in `shared/`. Nothing under
`shared/` may import from `products/` — the dependency direction is always
product → shared. A smoke test enforces this.

## Quick start

```bash
cp .env.example .env          # fill in API keys / DB creds
make install                  # venv + pip install -r requirements.txt + pip install -e .
make dev                      # uvicorn phansora.main:app --reload
```

System packages required: `ffmpeg`, `tesseract-ocr`.

Then:

- `GET /` — platform info + which products mounted
- `GET /health` — health + mounted prefixes
- `…/spokenverse/*`, `…/chrono/*`, `…/dossier/*` — each product's own routes

### Resilient boot

Products are mounted only if they import cleanly. A host missing one product's
heavy optional deps (torch, faiss, gptsovits, asyncpg…) still serves the rest.
Restrict what's mounted with `PHANSORA_ENABLED_PRODUCTS=spokenverse,chrono_origin`.

### Book Alchemy worker

Book Alchemy runs long jobs in a separate durable process:

```bash
make worker        # python -m phansora.products.spokenverse.worker
```

### CLI

```bash
phansora serve                 # run the API
phansora tts   --help          # SpokenVerse batch TTS / PDF->TXT
phansora dossier --help        # Dossier Nova pipeline
```

## Dependencies note

SpokenVerse (GPT-SoVITS) and Dossier Nova (sentence-transformers) both need
PyTorch but historically pinned different builds. This repo standardizes on the
CUDA `torch==2.5.1+cu124` build. For a CPU-only host, edit the
`--extra-index-url` line in `requirements.txt` to the CPU wheel index and drop
the `+cu124` suffixes.

## Provenance

Merged from three repositories (each retains its own history as a backup):
`spokenverse` (base), `tomeweaver` → `dossier_nova`, `chrono-origin` → `chrono_origin`.
