# Phansora API — unified image (serves all products; also runs the worker)
FROM python:3.11-slim

# System dependencies:
#   ffmpeg       — audio concat/transcode (SpokenVerse)
#   tesseract    — OCR (SpokenVerse PDF->TXT)
#   espeak-ng    — phonemization for StyleTTS2
#   libgl/glib   — runtime libs for pillow / pymupdf / faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        tesseract-ocr \
        espeak-ng \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better layer caching.
# NOTE: requirements.txt pins the CUDA (cu124) torch build. For a CPU-only
# image, edit the --extra-index-url line in requirements.txt before building.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .
RUN pip install -e .

ENV HOST=0.0.0.0 PORT=8000
EXPOSE 8000

CMD ["uvicorn", "phansora.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
