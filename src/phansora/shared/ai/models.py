# src/phansora/shared/ai/models.py
#
# Which LLM each product talks to, in one place.
#
# Every product resolves its model the same way, most specific first:
#
#   1. <PRODUCT>_MODEL      — this product only (e.g. BOOK_ALCHEMY_MODEL)
#   2. OPENAI_MODEL / DEEPSEEK_MODEL — every product on that provider
#   3. the built-in default below
#
# The per-product layer exists because the products have genuinely different needs: bulk OCR
# cleanup wants the cheap fast model, while a research/synthesis pass may be worth the
# expensive one. Before this, they all shared one var, so tuning one product silently
# re-pointed the rest.
#
# Keep this module import-light (stdlib only): shared.ai.deepseek imports it, so anything
# heavier would be pulled into every caller.

from __future__ import annotations

import os

# DeepSeek retired the `deepseek-chat` alias — the API now 400s with "The supported API model
# names are deepseek-v4-pro or deepseek-v4-flash". Flash is the default because the heaviest
# DeepSeek caller is bulk OCR cleanup (mechanical repair over many batched pages).
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
# gpt-5-nano is the cheapest capable OpenAI model with the native web_search tool.
DEFAULT_OPENAI_MODEL = "gpt-5-nano"

# Per-product override variables, by product. Documented here so `.env.example`, the README
# and any future product stay in step with what the code actually reads.
PRODUCT_MODEL_VARS = {
    "book_alchemy": "BOOK_ALCHEMY_MODEL",
    "chrono_origin": "CHRONO_MODEL",
    "dossier_nova": "DOSSIER_MODEL",
    "narrava_studio": "NARRAVA_MODEL",
    "spokenverse": "SPOKENVERSE_OCR_MODEL",
}


def _clean(value: str | None) -> str:
    # Tolerate `MODEL=name  # trailing comment`, which .env files pick up as part of the value.
    return (value or "").split("#", 1)[0].strip()


def provider_model(provider: str) -> str:
    """The model every product on ``provider`` uses unless it overrides."""
    if provider.strip().lower() == "openai":
        return _clean(os.getenv("OPENAI_MODEL")) or DEFAULT_OPENAI_MODEL
    return (
        _clean(os.getenv("DEEPSEEK_MODEL"))
        or _clean(os.getenv("DEEPSEEK_CHAT_MODEL"))  # legacy name, still honored
        or DEFAULT_DEEPSEEK_MODEL
    )


def resolve_model(product_var: str | None, *, provider: str = "deepseek") -> str:
    """Model for one product: its own var if set, else the provider-wide var, else default."""
    own = _clean(os.getenv(product_var)) if product_var else ""
    return own or provider_model(provider)
