# Phansora API

Unified Python (FastAPI) backend for [phansora.com](https://phansora.com). One app
serves all products, each under a path prefix:

| Product | Prefix | What it does |
|---|---|---|
| **SpokenVerse** | `/spokenverse` | text→audio (CosyVoice2 voice cloning), PDF→text, audio→text, Book Alchemy |
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

`requirements.txt` pins the `+cu126` torch 2.7.0 wheels as direct
`download.pytorch.org/whl/cu126/…` CloudFront URLs, cp310 / `manylinux_2_28_x86_64`
(the index links to Cloudflare R2, which the prod network can't reach over TLS). Prod
runs on an **RTX A4000 (16 GB, Ampere)** — plenty for CosyVoice2 (~3 GB fp16); `+cu126`
runs on any recent driver via CUDA minor-version compatibility. Note: the vLLM/TensorRT
acceleration is CUDA-only; on CPU set `COSYVOICE2_USE_VLLM=0`/`_USE_TRT=0` (minutes/gen).
For a CPU-only host, swap `whl/cu126` + `+cu126` for `whl/cpu` + `+cpu`.

### Local dev on Mac

`make install` pins the Linux CUDA torch wheel, which won't install on macOS. Use:

```bash
make install-mac    # venv + arm64 CPU/MPS torch + deps (skips the +cu126 pins)
make dev            # API runs on http://localhost:8000
```

The API boots **without** the TTS engine — every non-TTS route works immediately;
only a voice-generation call needs CosyVoice2, and without it you get a clean
"engine not configured" error instead of a crash.

CosyVoice2's fast path (vLLM + TensorRT) is **CUDA-only**, so full-quality TTS is not
available on a Mac. If you must synthesize locally, clone CosyVoice to `~/CosyVoice`,
install the Mac torch build, and set in `.env`:

```bash
TTS_ENGINE=cosyvoice2
COSYVOICE2_REPO=/Users/<you>/CosyVoice
COSYVOICE2_FP16=0
COSYVOICE2_USE_VLLM=0            # CUDA-only — falls back to the (slow) in-process LLM loop
COSYVOICE2_USE_TRT=0            # CUDA-only
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

CosyVoice2 auto-selects CUDA when available and falls back to CPU otherwise (Mac has no
CUDA). The `pynini` dependency (pulled in by CosyVoice's WeTextProcessing) is the tricky
part on macOS — install it via conda (`conda install -c conda-forge pynini==2.1.6`).

### TTS engine — CosyVoice2 (prod)

> **License:** CosyVoice is released under **Apache-2.0** (commercial use permitted).

CosyVoice2 is a git checkout, not a published pip package, run in-process. Its vLLM
backend **hard-pins `torch==2.7.0`** and needs `transformers==4.51.3` + `pydantic>=2.9`,
all pinned in `requirements.txt` and installed by `make install` — so run `make install`
FIRST, then `make install-tts` (clones CosyVoice, installs its requirements with the
conflicting torch/pydantic pins stripped, and downloads the CosyVoice2-0.5B checkpoints):

```bash
make install        # API venv: torch 2.7 + vllm 0.9.0 + transformers 4.51.3
make install-tts    # clone CosyVoice + its deps + model  (COSYVOICE_REPO=/var/www/CosyVoice)
```

`make install-tts` does the equivalent of:

```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /var/www/CosyVoice
# strip torch/torchaudio/pydantic pins (they fight vLLM) and re-assert ours:
sed -E '/^(torch|torchaudio|pydantic)==/d' /var/www/CosyVoice/requirements.txt > /tmp/cosy-reqs.txt
.venv/bin/pip install torch==2.7.0 torchaudio==2.7.0 "pydantic>=2.9" -r /tmp/cosy-reqs.txt
.venv/bin/python -c "from modelscope import snapshot_download; \
  snapshot_download('iic/CosyVoice2-0.5B', local_dir='/var/www/CosyVoice/pretrained_models/CosyVoice2-0.5B')"
```

Then set in `.env` and restart the service:

```bash
TTS_ENGINE=cosyvoice2
COSYVOICE2_REPO=/var/www/CosyVoice
# COSYVOICE2_MODEL_DIR=/var/www/CosyVoice/pretrained_models/CosyVoice2-0.5B  # default
COSYVOICE2_FP16=1                  # half VRAM + bandwidth (CUDA-only)
COSYVOICE2_USE_VLLM=1              # vLLM LLM backend — the big speedup (CUDA-only)
COSYVOICE2_USE_TRT=1               # TensorRT flow estimator (CUDA-only; builds once)
# "Default" voice needs a reference clip AND its transcript (CosyVoice conditions on the
# transcript). Set both so warmup also kernel-warms and the FIRST request is fast:
COSYVOICE2_DEFAULT_REF=/path/to/ref.wav
COSYVOICE2_DEFAULT_REF_TEXT=the exact words spoken in ref.wav
```

> **Runs on GPU (RTX A4000, 16 GB).** CosyVoice2 loads at ~3 GB (fp16) with plenty of
> headroom. The engine loads **once at FastAPI startup** (the model is a per-process
> singleton) — this pays weights + vLLM CUDA-graph capture (~80 s) and, on the very
> first boot, a one-time TensorRT engine build (cached to disk after). The startup
> preload runs off-thread, so `/health` stays responsive; **subsequent requests only pay
> synthesis time** (sub-real-time). Run the service with a **single worker** (`make run`)
> — extra workers would each load their own copy, doubling VRAM and warmup.

> **No CUDA toolkit / nvcc needed.** Unlike the old DeepSpeed path, vLLM's and TensorRT's
> CUDA ops ship precompiled in their wheels — the box needs only the NVIDIA **driver**,
> not the CUDA toolkit. `scripts/install-cuda-toolkit.sh` and the `cuda-env.conf` drop-in
> are legacy (DeepSpeed) and no longer required for the TTS engine.

The API boots without the engine; only TTS calls need it. Speed (0.5–2.0×, native) is a
per-request option — see `GET /spokenverse/tts-options`. There is no emotion control
(CosyVoice2 has none; it clones from a reference clip + its transcript).

## Environment (`.env`)

Copy `.env.example` → `.env` and fill it in. `.env.example` documents **every** var;
below are only the ones that **differ between dev and prod** — everything else
(`TTS_ENGINE`, DB name/creds, `ANTHROPIC_*`, `DEEPSEEK_*`, `SMTP_*`) is identical in
both.

| Var | Dev (Mac / local) | Prod (Linux GPU) |
|---|---|---|
| `CORS_ALLOW_ORIGINS` | `http://localhost:3000` | your real site origin(s) |
| `PHANSORA_DATA_DIR` | *unset* → uses cwd | `/var/lib/phansora` (voices/audio/db live here) |
| `COSYVOICE2_REPO` | `~/CosyVoice` | `/var/www/CosyVoice` |
| `COSYVOICE2_FP16` | `0` (CPU) | `1` (fp16 on GPU — faster, lower VRAM) |
| `COSYVOICE2_USE_VLLM` | `0` (CUDA-only) | `1` (the LLM speedup) |
| `COSYVOICE2_USE_TRT` | `0` (CUDA-only) | `1` (flow TensorRT engine) |
| `COSYVOICE2_DEFAULT_REF` / `_REF_TEXT` | *unset* | ref clip + its transcript for the "default" voice |
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

## Known issues / what not to do (SpokenVerse + CosyVoice2 + vLLM)

Hard-won notes from the CosyVoice2 + vLLM GPU setup. **Read before touching the TTS engine
or its torch/vLLM pins** — several of these cost real time.

1. **vLLM 0.9.0 hard-pins `torch==2.7.0` — the whole API venv is on torch 2.7, not 2.8.**
   Installing CosyVoice's own `requirements.txt` unmodified will *downgrade* torch to 2.3.1
   and pydantic to 2.7.0 (its pins), breaking vLLM. A pip **constraints** file can't override
   an explicit `==` pin, so `make install-tts` **strips** `torch`/`torchaudio`/`pydantic` from
   CosyVoice's requirements and re-asserts `torch==2.7.0 torchaudio==2.7.0 pydantic>=2.9` on the
   command line. If pip then errors on `deepspeed`/`lightning` capping torch, add them to the
   strip list — they're training-only and unused at inference.

2. **Register `CosyVoice2ForCausalLM` with vLLM BEFORE constructing the engine.** CosyVoice2's
   LLM is a custom vLLM architecture; without `ModelRegistry.register_model(...)` (done in
   `cosyvoice2_client._load_cosy`, gated on `COSYVOICE2_USE_VLLM`) vLLM raises "Cannot find
   model module 'CosyVoice2ForCausalLM'".

3. **The model loads ONCE at FastAPI startup; don't construct it per request.** It's a
   per-process singleton (`_load_cosy`, lock-guarded, cached). Startup preload (off-thread)
   pays weights + vLLM CUDA-graph capture (~80 s) + first-run TensorRT build (cached to disk).
   Constructing `CosyVoice2` inside a request handler would re-pay all of that every call.

4. **Run a single uvicorn worker (`make run` → `--workers 1`).** vLLM's graph capture is
   per-process and not disk-cached, and each worker holds its own resident engine — extra
   workers multiply both VRAM and the ~80 s warmup. Scale non-TTS load via replicas/proxy.

5. **CosyVoice needs the reference clip's transcript (`prompt_text`).** Unlike the old engine,
   CosyVoice conditions on the transcript. Cloned voices store it as `ref_text` (auto-whisper
   at create-voice); the `default` voice needs `COSYVOICE2_DEFAULT_REF_TEXT`. No transcript →
   a clear "needs transcript" error, not silent garbage.

6. **No emotion control.** CosyVoice2 has no emotion vector/intensity knob (that was IndexTTS2).
   Only `speed` (0.5–2.0×, native) remains. Don't wire emotion sliders back into the UI.

7. **No CUDA toolkit / nvcc required.** vLLM and TensorRT ship precompiled CUDA ops in their
   wheels; the box needs only the NVIDIA driver. `scripts/install-cuda-toolkit.sh` and the
   `cuda-env.conf` systemd drop-in are legacy (DeepSpeed) and no longer needed for TTS.

8. **CosyVoice2 silently drops/skips words on long inference chunks — cap chunks at ~200 chars.**
   Given a long text in a single `inference_zero_shot` call, CosyVoice2 intermittently stops
   generating early and **omits whole clauses** — the dropped span clusters at the *tail* of the
   chunk. It is **far worse with cloned voices** than the `default` voice, and it is a *model*
   behavior, not a chunking-boundary or ffmpeg-concat bug (drops land mid-chunk, not at joins).
   Measured on prod with a cloned voice (synthesize via the warm API → whisper-transcribe →
   word-diff against the source): 550/400 chars dropped whole sentences, 300 dropped the tail,
   250 dropped words in 1/2 trials, **200 was clean in 8/8 trials** (and clean on a 5-paragraph,
   ~2 k-char / 351-word input). **Fix/workaround:** `MAX_CHARS_DEFAULT = 200` in
   `txt_to_voice/adapters/cosyvoice2_client.py` bounds every inference chunk; `_chunk_text` packs
   whole lines/sentences up to that cap (newline-aware so verse — line breaks, no periods — also
   splits), and only a single run longer than the cap is broken at a word boundary. Tune without
   redeploying via `COSYVOICE2_MAX_CHARS` in `.env` (then `systemctl restart phansora-api`). Do
   **not** raise it back toward 550 "for fewer joins" — 250 already dropped words. If drops ever
   reappear, lower it further and re-verify with the whisper word-diff method above.
   *Latency note:* smaller chunks ⇒ more serialized inference (one `_INFER_LOCK`), so a very
   large single upload to the synchronous `/txt-to-audio` can approach nginx's `proxy_read_timeout`
   (300 s). Book Alchemy long-form runs through the durable worker, not that endpoint, so it is
   unaffected; only watch the direct front-end upload path.
