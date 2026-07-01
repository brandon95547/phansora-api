"""Dossier Nova — AI research & dossier generation (formerly tomeweaver).

IMPORTANT — OpenMP guard.
faiss and PyTorch (pulled in by sentence-transformers) each bundle their own
OpenMP runtime. Loading both into one process can trigger a hard crash
("OMP: Error #15: ... libomp already initialized"). These must be set BEFORE
faiss/torch are imported anywhere — the package __init__ runs before any
submodule, so this is the correct place.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
