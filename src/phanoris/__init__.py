"""Phanoris platform API.

A single backend that hosts multiple AI products under one package:

    phanoris.products.spokenverse    — PDF/OCR, text->audio (StyleTTS2), Book Alchemy
    phanoris.products.chrono_origin  — trace the origin of a story/myth (Claude grounded search)
    phanoris.products.dossier_nova   — AI research & dossier generation (embeddings + DeepSeek)

Cross-cutting infrastructure lives under ``phanoris.shared`` (AI clients, database,
storage, queue, auth, billing, utils) so products stay thin and reusable pieces
are written once.
"""

__version__ = "0.1.0"
