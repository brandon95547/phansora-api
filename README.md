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

**Prerequisite — Python 3.10 via [uv](https://astral.sh/uv).** The torch wheels are
`cp310`, so the venv MUST be Python 3.10. CentOS Stream 8's system `python3` is 3.6
(and it ships no 3.10), so `make install` builds the venv with uv, which fetches a
real 3.10. Install uv first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"      # persist in ~/.bashrc
```

Then:

```bash
cp .env.example .env      # fill in DB creds / API keys
make install              # uv → Python 3.10 venv (seeded w/ pip) + deps (CUDA torch 2.8.x) + editable install
```

`make install` wipes any existing `.venv` and rebuilds it with uv (`uv venv --python
3.10 --seed`). If uv isn't on PATH it falls back to a system `python3.10`, and errors
with instructions if neither is found — so **don't** run a bare `python3 -m venv`
(that gives a 3.6 venv and the torch wheel fails with "not a supported wheel").

`requirements.txt` pins the `+cu126` torch 2.8.0 wheels as direct
`download.pytorch.org/whl/cu126/…` CloudFront URLs, cp310 / `manylinux_2_28_x86_64`
(the index links to Cloudflare R2, which the prod network can't reach over TLS). Prod
runs on an **RTX A4000 (16 GB, Ampere)** — plenty for IndexTTS2 (~5 GB); `+cu126` runs
on any recent driver via CUDA minor-version compatibility. Note: IndexTTS2 needs a real
GPU (~8 GB+); a 4 GB card OOMs, and CPU synthesis is minutes/generation. For a CPU-only
host, swap `whl/cu126` + `+cu126` for `whl/cpu` + `+cpu`.

### Local dev on Mac

`make install` pins the Linux CUDA torch wheel, which won't install on macOS. Use:

```bash
make install-mac    # venv + arm64 CPU/MPS torch + deps (skips the +cu126 pins)
make dev            # API runs on http://localhost:8000
```

The API boots **without** the TTS engine — every non-TTS route works immediately;
only a voice-generation call needs IndexTTS2, and without it you get a clean
"engine not configured" error instead of a crash.

To generate audio locally too, follow the *TTS engine* steps below but clone to
`~/index-tts`, install the Mac torch build (`.venv/bin/pip install torch torchaudio`),
and set in `.env`:

```bash
TTS_ENGINE=indextts2
INDEXTTS2_REPO=/Users/<you>/index-tts
INDEXTTS2_USE_GPU=0
INDEXTTS2_FP16=0
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

IndexTTS2 auto-selects CUDA when available and falls back to CPU otherwise (Mac has no
CUDA, so it runs on CPU). The `pynini` dependency (pulled in by WeTextProcessing) is the tricky
part on macOS — install it via conda (`conda install -c conda-forge pynini==2.1.6`).

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
#    which make install already satisfied, so pip leaves the cu126 torch alone (no
#    re-download). If you install this BEFORE make install, pip would pull a plain-PyPI
#    torch 2.8 instead of our +cu126 build — always run make install first.
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
INDEXTTS2_FP16=1                   # fp16 on GPU — faster, lower VRAM
# "Default" voice needs a reference clip (IndexTTS2 always clones from a speaker prompt):
# INDEXTTS2_DEFAULT_REF=/path/to/ref.wav
```

> **Runs on GPU (RTX A4000, 16 GB).** IndexTTS2 loads (~5 GB) with headroom and
> synthesizes in seconds. Requires a real GPU with ~8 GB+ VRAM — a 4 GB card OOMs on
> load, and CPU falls back to minutes/generation. First request after a restart pays a
> one-time model load (~10–30 s); keep nginx `proxy_read_timeout` at a few minutes.

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
| `INDEXTTS2_USE_GPU` | `0` (no CUDA → CPU) | `1` (RTX A4000, 16 GB) |
| `INDEXTTS2_FP16` | `0` | `1` (fp16 on GPU — faster, lower VRAM) |
| `WHISPER_DEVICE` | `cpu` | `cuda` |
| `WHISPER_COMPUTE_TYPE` | `int8` | `float16` |
| `DB_HOST` / `DB_PORT` | your local Postgres | `127.0.0.1` / the shared Postgres port |

Notes:
- **Torch build differs at install time, not via `.env`:** `make install` pins the
  CUDA torch 2.8.x wheels (`+cu126`, prod GPU); `make install-mac` uses the arm64
  CPU/MPS wheel (dev). See above.
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
- `proxy_read_timeout 300s` — the first TTS request after a restart loads the model
  (~10–30 s on GPU) and otherwise times out; generations themselves are quick.
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
