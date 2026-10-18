"""Phansora platform API.

A single backend that hosts multiple AI products under one package:

    phansora.products.spokenverse    — PDF/OCR, text->audio (CosyVoice), audio->text
    phansora.products.book_alchemy   — a book turned into a narrated audio course
    phansora.products.chrono_origin  — trace the origin of a story/myth (grounded web search)
    phansora.products.research_atlas — supplied research organized into a report (DeepSeek)
    phansora.products.narrava_studio — documentary editor: script, storyboard, AI media
    phansora.products.talari         — illustrated books: writing (DeepSeek), pictures (ComfyUI)

Cross-cutting infrastructure lives under ``phansora.shared`` (AI clients, auth, admin,
contact, conversions, utils) so products stay thin and reusable pieces are written once.
"""

__version__ = "0.1.0"
