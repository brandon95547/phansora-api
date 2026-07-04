# Phansora API

Unified Python (FastAPI) backend for [phansora.com](https://phansora.com). One app
serves all products, each under a path prefix:

| Product | Prefix | What it does |
|---|---|---|
| **SpokenVerse** | `/spokenverse` | text→audio (GPT-SoVITS voice cloning), PDF→text, audio→text, Book Alchemy |
| **Chrono-Origin** | `/chrono` | traces a story/myth's earliest origin (Claude web search) |
| **Dossier Nova** | `/dossier` | AI research → source-attributed dossier (local embeddings + DeepSeek) |

System packages: `ffmpeg`, `tesseract-ocr`.

## Install

```bash
cp .env.example .env      # fill in DB creds / API keys
make install              # venv + deps (CUDA torch 2.5.1+cu124) + editable install
```

CPU-only host: swap the `--extra-index-url` in `requirements.txt` to the CPU wheel
index and drop the `+cu124` suffixes.

### Local dev on Mac

`make install` pins CUDA torch, which won't install on macOS. Use:

```bash
make install-mac    # venv + arm64 CPU/MPS torch + deps (skips the +cu124 pins)
make dev            # API runs on http://localhost:8000
```

The API boots **without** the TTS engine — every non-TTS route works immediately;
only a voice-generation call needs GPT-SoVITS, and without it you get a clean
"engine not configured" error instead of a crash.

To generate audio locally too (runs on CPU — slower than the prod GPU, but works):

```bash
brew install ffmpeg
git clone https://github.com/RVC-Boss/GPT-SoVITS.git ~/GPT-SoVITS
.venv/bin/pip install -r ~/GPT-SoVITS/requirements.txt
# GPT-SoVITS's requirements re-pull torch — re-pin the Mac build:
.venv/bin/pip install torch torchvision torchaudio
```

Then download the v2 checkpoints + NLTK data (steps 3–4 under *TTS engine* below,
pointing `local_dir` at `~/GPT-SoVITS/GPT_SoVITS/pretrained_models`) and set in
`.env`:

```bash
TTS_ENGINE=gptsovits
GPTSOVITS_REPO=/Users/<you>/GPT-SoVITS
GPTSOVITS_USE_GPU=0        # no CUDA on Mac → runs on CPU
GPTSOVITS_IS_HALF=0
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

`_resolve_device()` auto-falls back to `cpu` on a Mac. Apple-Silicon GPU (`mps`) is
possible via `GPTSOVITS_DEVICE=mps` but GPT-SoVITS's MPS path is flaky — stick with
CPU locally.

### TTS engine — GPT-SoVITS (prod)

The engine is a git checkout, not a pip package. Install it into the **same venv**:

```bash
# 1. clone + its deps
git clone https://github.com/RVC-Boss/GPT-SoVITS.git /var/www/GPT-SoVITS
.venv/bin/pip install -r /var/www/GPT-SoVITS/requirements.txt

# 2. re-pin cu124 torch (step 1 pulls a cu13x build that breaks CUDA 12.4 drivers)
.venv/bin/pip install "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" \
  --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install "transformers>=4.43,<4.51" "sentence-transformers>=5.0"
.venv/bin/pip check       # must be clean

# 3. v2 model checkpoints (~1 GB)
.venv/bin/pip install "huggingface_hub[cli]"
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("lj1995/GPT-SoVITS", local_dir="/var/www/GPT-SoVITS/GPT_SoVITS/pretrained_models",
  allow_patterns=["chinese-hubert-base/*","chinese-roberta-wwm-ext-large/*",
                  "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
                  "gsv-v2final-pretrained/s2G2333k.pth"])
PY

# 4. NLTK data (English g2p)
.venv/bin/python -c "import nltk; [nltk.download(r) for r in ('averaged_perceptron_tagger_eng','cmudict','punkt','punkt_tab')]"
```

Then set in `.env` and restart the service:

```bash
TTS_ENGINE=gptsovits
GPTSOVITS_REPO=/var/www/GPT-SoVITS
GPTSOVITS_USE_GPU=1
GPTSOVITS_IS_HALF=0                 # fp16 can produce silent output on some GPUs
# "Default" voice needs a reference clip (GPT-SoVITS always clones from one):
# GPTSOVITS_DEFAULT_REF=/path/to/ref.wav
# GPTSOVITS_DEFAULT_REF_TEXT=what that clip says
```

The API boots without the engine; only TTS calls need it.

## Environment (`.env`)

Copy `.env.example` → `.env` and fill it in. `.env.example` documents **every** var;
below are only the ones that **differ between dev and prod** — everything else
(`TTS_ENGINE`, DB name/creds, `ANTHROPIC_*`, `DEEPSEEK_*`, `SMTP_*`) is identical in
both.

| Var | Dev (Mac / local) | Prod (Linux GPU) |
|---|---|---|
| `CORS_ALLOW_ORIGINS` | `http://localhost:3000` | your real site origin(s) |
| `PHANSORA_DATA_DIR` | *unset* → uses cwd | `/var/lib/phansora` (voices/audio/db live here) |
| `GPTSOVITS_REPO` | `~/GPT-SoVITS` | `/var/www/GPT-SoVITS` |
| `GPTSOVITS_USE_GPU` | `0` (no CUDA → CPU) | `1` (CUDA) |
| `GPTSOVITS_IS_HALF` | `0` | `0` (fp16 → silent output on some GPUs) |
| `WHISPER_DEVICE` | `cpu` | `cuda` if supported, else `cpu` |
| `WHISPER_COMPUTE_TYPE` | `int8` | `float16` (cuda) / `int8` (cpu) |
| `DB_HOST` / `DB_PORT` | your local Postgres | `127.0.0.1` / the shared Postgres port |

Notes:
- **Torch build differs at install time, not via `.env`:** `make install` pins CUDA
  `+cu124` (prod); `make install-mac` uses the arm64 CPU/MPS wheel (dev). See above.
- **DB is shared:** this API and the Node site talk to the **same** Postgres —
  `DB_PORT` must match that server (they are not two databases).
- **`.env` format — keep comments on their own lines.** Both python-dotenv and
  systemd's `EnvironmentFile` treat *everything* after `=` as the value, so
  `PORT=8000  # optional` becomes the literal `8000  # optional` and breaks int
  parsing. Inline `#` comments are only safe on lines you never read as a number —
  just don't.

## Run

**Dev (local / Mac):**

```bash
make dev      # uvicorn --reload on http://localhost:8000
make worker   # Book Alchemy job runner — SEPARATE process, only needed for Book Alchemy
```

**Prod (Linux GPU):** `phansora-api` and `phansora-worker` run under **systemd**
(not `make`), behind **nginx** with `proxy_read_timeout 300s` — the first TTS request
loads the model and otherwise times out. One uvicorn worker (the GPU model is loaded
once per process). After editing `.env`, `systemctl restart phansora-api`.

```bash
make run      # the prod-ish command systemd wraps: uvicorn --workers 1
```

## Endpoints

- `GET /health` — health + mounted products
- `GET /spokenverse/tts-options` — TTS backend + available settings
- `…/spokenverse/*`, `…/chrono/*`, `…/dossier/*` — each product's routes

## CLI

```bash
phansora tts --help        # batch TTS / PDF→TXT
phansora dossier --help    # Dossier Nova pipeline
```
