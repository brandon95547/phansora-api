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

## Run

```bash
make dev      # dev: uvicorn --reload
make run      # prod-ish: uvicorn --workers 1  (GPU model = 1 worker)
make worker   # Book Alchemy job runner — a SEPARATE process, required for Book Alchemy
```

Prod runs `phansora-api` (and `phansora-worker`) under systemd. Put the API behind
nginx with `proxy_read_timeout 300s` — the first TTS request loads the model and
otherwise times out.

## Endpoints

- `GET /health` — health + mounted products
- `GET /spokenverse/tts-options` — TTS backend + available settings
- `…/spokenverse/*`, `…/chrono/*`, `…/dossier/*` — each product's routes

## CLI

```bash
phansora tts --help        # batch TTS / PDF→TXT
phansora dossier --help    # Dossier Nova pipeline
```
