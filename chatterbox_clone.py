#!/usr/bin/env python3
"""
Dead-simple Chatterbox (Resemble AI) voice clone — terminal only, no app.

Edit the three variables below, then run:  python chatterbox_clone.py

Setup (on a machine with torch 2.6 support — Linux/NVIDIA or Apple Silicon;
NOT an Intel Mac, which caps at torch 2.2):

    python3 -m venv .venv && source .venv/bin/activate
    pip install --upgrade pip
    pip install chatterbox-tts        # pulls torch 2.6 + deps; first run downloads weights (~1-2 GB)
"""
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# ---- edit these ----
REFERENCE = "strong.mp3"                                  # voice to clone (wav or mp3)
TEXT = "This is a sample of the cloned voice."            # what it should say
OUTPUT = "cloned.wav"                                     # output file
# --------------------

if torch.cuda.is_available():
    device = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"loading chatterbox on {device} (first run downloads weights)...")
model = ChatterboxTTS.from_pretrained(device=device)

print(f"cloning from {REFERENCE} ...")
wav = model.generate(TEXT, audio_prompt_path=REFERENCE)

ta.save(OUTPUT, wav, model.sr)
print(f"done -> {OUTPUT}  ({model.sr} Hz)")