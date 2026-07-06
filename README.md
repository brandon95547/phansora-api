# Phansora API

Unified Python (FastAPI) backend for [phansora.com](https://phansora.com). One app
serves all products, each under a path prefix:

| Product | Prefix | What it does |
|---|---|---|
| **SpokenVerse** | `/spokenverse` | text→audio (IndexTTS2 voice cloning + emotion control), PDF→text, audio→text, Book Alchemy |
| **Chrono-Origin** | `/chrono` | traces a story/myth's earliest origin (Claude web search) |
| **Dossier Nova** | `/dossier` | AI research → source-attributed dossier (local embeddings + DeepSeek) |

System packages: `ffmpeg`, `tesseract-ocr`.

## Install

```bash
cp .env.example .env      # fill in DB creds / API keys
make install              # venv + deps (CUDA torch 2.8.x) + editable install
```

CPU-only host: `requirements.txt` pins the three torch wheels as direct
`download.pytorch.org/whl/cu128/…` CloudFront URLs (the CUDA *index* links to
Cloudflare R2, which the prod network can't reach over TLS). Swap those filenames to
the `whl/cpu` build and drop the `+cu128` suffixes. The CUDA build (`cu126` vs `cu128`)
must match the prod GPU driver — check `nvidia-smi` and pin accordingly.

### Local dev on Mac

`make install` pins CUDA torch, which won't install on macOS. Use:

```bash
make install-mac    # venv + arm64 CPU/MPS torch + deps (skips the +cu124 pins)
make dev            # API runs on http://localhost:8000
```

The API boots **without** the TTS engine — every non-TTS route works immediately;
only a voice-generation call needs IndexTTS2, and without it you get a clean
"engine not configured" error instead of a crash.

To generate audio locally too (runs on CPU/MPS — slower than the prod GPU, but works),
follow the *TTS engine* steps below but clone to `~/index-tts`, install the Mac torch
build (`.venv/bin/pip install torch torchaudio`), and set in `.env`:

```bash
TTS_ENGINE=indextts2
INDEXTTS2_REPO=/Users/<you>/index-tts
INDEXTTS2_USE_GPU=0        # no CUDA on Mac → runs on CPU/MPS
INDEXTTS2_FP16=0
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

IndexTTS2 uses CUDA automatically when available and falls back to CPU otherwise.
The `pynini` dependency (pulled in by WeTextProcessing) is the tricky part on macOS —
install it via conda (`conda install -c conda-forge pynini==2.1.6`) into the same env.

### TTS engine — IndexTTS2 (prod)

> **License:** IndexTTS2 is released under a **non-commercial** license; commercial use
> requires a separate license from Bilibili (`indexspeech@bilibili.com`). Ensure the
> deployment is covered before shipping.

IndexTTS2 is a git checkout, not a published pip package, run in-process. It needs
**torch 2.8.x** and `transformers==4.52.1`, both already pinned in `requirements.txt`
and installed by `make install` — so run `make install` FIRST, then install IndexTTS2
into the **same venv**:

```bash
# 1. clone
git clone https://github.com/index-tts/index-tts.git /var/www/index-tts

# 2. install IndexTTS2's deps into this venv. It pins torch==2.8.* / transformers==4.52.1,
#    which make install already satisfied, so pip leaves the cu128 torch alone (no
#    re-download). If you install this BEFORE make install, pip would pull a plain-PyPI
#    torch 2.8 instead of our cu128 build — always run make install first.
.venv/bin/pip install -e /var/www/index-tts

# 3. pynini/WeTextProcessing need OpenFst — install via conda into this env
#    (pip install pynini usually fails to build on CentOS):
conda install -y -c conda-forge pynini==2.1.6
.venv/bin/pip install WeTextProcessing
.venv/bin/pip check       # must be clean — torch should still be 2.8.x

# 4. IndexTTS-2 checkpoints (several GB) via Hugging Face
.venv/bin/pip install "huggingface_hub[cli]"
.venv/bin/hf download IndexTeam/IndexTTS-2 --local-dir /var/www/index-tts/checkpoints
```

Then set in `.env` and restart the service:

```bash
TTS_ENGINE=indextts2
INDEXTTS2_REPO=/var/www/index-tts
INDEXTTS2_MODEL_DIR=/var/www/index-tts/checkpoints   # holds config.yaml + *.pth
INDEXTTS2_USE_GPU=1
INDEXTTS2_FP16=0                   # 1 for faster GPU inference once verified
# "Default" voice needs a reference clip (IndexTTS2 always clones from a speaker prompt):
# INDEXTTS2_DEFAULT_REF=/path/to/ref.wav
```

The API boots without the engine; only TTS calls need it. Emotion control (an
expressiveness weight + the 8-way emotion vector) and speed are per-request options —
see `GET /spokenverse/tts-options`.

## Environment (`.env`)

Copy `.env.example` → `.env` and fill it in. `.env.example` documents **every** var;
below are only the ones that **differ between dev and prod** — everything else
(`TTS_ENGINE`, DB name/creds, `ANTHROPIC_*`, `DEEPSEEK_*`, `SMTP_*`) is identical in
both.

| Var | Dev (Mac / local) | Prod (Linux GPU) |
|---|---|---|
| `CORS_ALLOW_ORIGINS` | `http://localhost:3000` | your real site origin(s) |
| `PHANSORA_DATA_DIR` | *unset* → uses cwd | `/var/lib/phansora` (voices/audio/db live here) |
| `INDEXTTS2_REPO` | `~/index-tts` | `/var/www/index-tts` |
| `INDEXTTS2_USE_GPU` | `0` (no CUDA → CPU) | `1` (CUDA) |
| `INDEXTTS2_FP16` | `0` | `0` (set `1` for faster GPU once verified) |
| `WHISPER_DEVICE` | `cpu` | `cuda` if supported, else `cpu` |
| `WHISPER_COMPUTE_TYPE` | `int8` | `float16` (cuda) / `int8` (cpu) |
| `DB_HOST` / `DB_PORT` | your local Postgres | `127.0.0.1` / the shared Postgres port |

Notes:
- **Torch build differs at install time, not via `.env`:** `make install` pins CUDA
  torch 2.8.x (prod, CUDA build matched to the driver); `make install-mac` uses the
  arm64 CPU/MPS wheel (dev). See above.
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
