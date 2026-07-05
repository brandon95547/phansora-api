# Phansora API

Unified Python (FastAPI) backend for [phansora.com](https://phansora.com). One app
serves all products, each under a path prefix:

| Product | Prefix | What it does |
|---|---|---|
| **SpokenVerse** | `/spokenverse` | text→audio (CosyVoice 2 voice cloning), PDF→text, audio→text, Book Alchemy |
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
only a voice-generation call needs CosyVoice, and without it you get a clean
"engine not configured" error instead of a crash.

To generate audio locally too (runs on CPU — slower than the prod GPU, but works),
follow the *TTS engine* steps below but clone to `~/CosyVoice`, install the Mac torch
build (`.venv/bin/pip install torch torchaudio`), and set in `.env`:

```bash
TTS_ENGINE=cosyvoice
COSYVOICE_REPO=/Users/<you>/CosyVoice
COSYVOICE_USE_GPU=0        # no CUDA on Mac → runs on CPU
COSYVOICE_FP16=0
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

CosyVoice uses CUDA automatically when available and falls back to CPU otherwise.
The `pynini` dependency is the tricky part on macOS — install it via conda
(`conda install -c conda-forge pynini==2.1.6`) into the same env, or set
`COSYVOICE_TEXT_FRONTEND=0` to skip text normalization.

### TTS engine — CosyVoice 2 (prod)

CosyVoice 2 is Apache-2.0 (commercial-OK). The engine is a git checkout, not a pip
package. Install it into the **same venv**:

```bash
# 1. clone (with the vendored Matcha-TTS submodule) + its deps
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /var/www/CosyVoice
.venv/bin/pip install -r /var/www/CosyVoice/requirements.txt

# 2. pynini/WeTextProcessing need OpenFst — install via conda into this env
#    (pip install pynini usually fails to build on CentOS):
conda install -y -c conda-forge pynini==2.1.6
.venv/bin/pip install WeTextProcessing

# 3. re-pin cu124 torch (step 1 may pull a different build)
.venv/bin/pip install "torch==2.5.1" "torchvision==0.20.1" "torchaudio==2.5.1" \
  --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install "transformers>=4.43,<4.51" "sentence-transformers>=5.0"
.venv/bin/pip check       # must be clean

# 4. CosyVoice2-0.5B checkpoints (~2 GB) via ModelScope
.venv/bin/pip install modelscope
.venv/bin/python - <<'PY'
from modelscope import snapshot_download
snapshot_download('iic/CosyVoice2-0.5B',
  local_dir='/var/www/CosyVoice/pretrained_models/CosyVoice2-0.5B')
PY
```

Then set in `.env` and restart the service:

```bash
TTS_ENGINE=cosyvoice
COSYVOICE_REPO=/var/www/CosyVoice
COSYVOICE_USE_GPU=1
COSYVOICE_FP16=0                   # 1 for faster GPU inference once verified
# "Default" voice needs a reference clip (CosyVoice always clones from one):
# COSYVOICE_DEFAULT_REF=/path/to/ref.wav
# COSYVOICE_DEFAULT_REF_TEXT=what that clip says
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
| `COSYVOICE_REPO` | `~/CosyVoice` | `/var/www/CosyVoice` |
| `COSYVOICE_USE_GPU` | `0` (no CUDA → CPU) | `1` (CUDA) |
| `COSYVOICE_FP16` | `0` | `0` (set `1` for faster GPU once verified) |
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
(not `make`), behind **nginx**. Two nginx settings this API needs:
- `proxy_read_timeout 300s` — the first TTS request loads the model and otherwise times out.
- `client_max_body_size 25m` — create-voice uploads a reference clip; nginx's 1 MB
  default returns **413** for anything larger (the app trims the clip to 9s, but only
  after it arrives).

One uvicorn worker (the GPU model is loaded once per process). After editing `.env`,
`systemctl restart phansora-api`.

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
