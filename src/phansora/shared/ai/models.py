# src/phansora/shared/ai/models.py
#
# Which LLM each product talks to, in one place.
#
# Every product resolves its model the same way, most specific first:
#
#   1. <PRODUCT>_MODEL      — this product only (e.g. BOOK_ALCHEMY_MODEL)
#   2. GEMINI_MODEL / OPENAI_MODEL / DEEPSEEK_MODEL — every product on that provider
#   3. nothing — the call raises MissingModelConfig (there is no built-in name)
#
# Judgment calls resolve the same way through the *_REASONING_MODEL variables, falling
# through to the chat chain above when they are blank.
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

# There are deliberately NO built-in model names here.
#
# A hardcoded default is a time bomb: it is correct until the provider retires the name,
# and then it fails from inside the code where an .env edit cannot reach it. That is
# exactly how `deepseek-chat` broke every DeepSeek product — the alias was retired, and a
# default kept handing it to the API long after the .env had moved on.
#
# So the model name comes from the environment or the call fails immediately, with a
# message naming the variable to set. A loud failure at startup is strictly better than a
# provider 400 halfway through a job.


class MissingModelConfig(RuntimeError):
    """No model configured for a product/provider, and there is no fallback by design."""


# Per-product override variables, by product. Documented here so `.env.example`, the README
# and any future product stay in step with what the code actually reads.
PRODUCT_MODEL_VARS = {
    "book_alchemy": "BOOK_ALCHEMY_MODEL",
    "chrono_origin": "CHRONO_MODEL",
    "dossier_nova": "DOSSIER_MODEL",
    "narrava_studio": "NARRAVA_MODEL",
    "spokenverse": "SPOKENVERSE_OCR_MODEL",
}

# The reasoning/judgment model is resolved the same way, one layer at a time, so a
# provider retiring a model name is an .env edit and a restart — never a code change.
# Its own default is the chat chain: leave every *_REASONING_MODEL blank and reasoning
# simply follows the chat model, which is the right behavior for providers that no
# longer ship a separate reasoning SKU.
PRODUCT_REASONING_MODEL_VARS = {
    "book_alchemy": "BOOK_ALCHEMY_REASONING_MODEL",
    "chrono_origin": "CHRONO_REASONING_MODEL",
    "dossier_nova": "DOSSIER_REASONING_MODEL",
    "narrava_studio": "NARRAVA_REASONING_MODEL",
    "spokenverse": "SPOKENVERSE_REASONING_MODEL",
}


def _clean(value: str | None) -> str:
    # Tolerate `MODEL=name  # trailing comment`, which .env files pick up as part of the value.
    return (value or "").split("#", 1)[0].strip()


def _is_openai(provider: str) -> bool:
    return provider.strip().lower() == "openai"


def _is_gemini(provider: str) -> bool:
    return provider.strip().lower() in ("gemini", "google")


def provider_model(provider: str) -> str:
    """The model every product on ``provider`` uses unless it overrides.

    Empty when unset — there is no built-in name to fall back to (see the note above)."""
    if _is_gemini(provider):
        return _clean(os.getenv("GEMINI_MODEL"))
    if _is_openai(provider):
        return _clean(os.getenv("OPENAI_MODEL"))
    return (
        _clean(os.getenv("DEEPSEEK_MODEL"))
        or _clean(os.getenv("DEEPSEEK_CHAT_MODEL"))  # legacy name, still honored
    )


def resolve_model(product_var: str | None, *, provider: str = "deepseek") -> str:
    """Model for one product: its own var if set, else the provider-wide var.

    Raises :class:`MissingModelConfig` when neither is set, rather than inventing a name."""
    own = _clean(os.getenv(product_var)) if product_var else ""
    model = own or provider_model(provider)
    if not model:
        raise MissingModelConfig(_missing_message(product_var, provider))
    return model


def _missing_message(product_var: str | None, provider: str) -> str:
    """Name the variables that would fix this, most specific first.

    Reasoning callers land here too, and correctly: an unset *_REASONING_MODEL is not an
    error — it means "reason with the chat model" — so the chat variables are what to set."""
    provider_var = (
        "GEMINI_MODEL" if _is_gemini(provider)
        else "OPENAI_MODEL" if _is_openai(provider)
        else "DEEPSEEK_MODEL"
    )
    wanted = " or ".join(n for n in (product_var, provider_var) if n)
    return (
        f"No {provider} model is configured. Set {wanted} in .env and restart. "
        f"There is no built-in default on purpose: a hardcoded name silently breaks when "
        f"the provider retires it."
    )


def provider_reasoning_model(provider: str) -> str:
    """The reasoning model every product on ``provider`` uses unless it overrides.

    Empty means "no separate reasoning model" — the caller falls back to the chat
    chain rather than to a hardcoded name that a provider may since have retired."""
    if _is_gemini(provider):
        var = "GEMINI_REASONING_MODEL"
    elif provider.strip().lower() == "openai":
        var = "OPENAI_REASONING_MODEL"
    else:
        var = "DEEPSEEK_REASONING_MODEL"
    return _clean(os.getenv(var))


def resolve_reasoning_model(product_var: str | None, *, provider: str = "deepseek") -> str:
    """Reasoning model for one product, most specific first:

        <PRODUCT>_REASONING_MODEL      this product's judgment calls only
        {DEEPSEEK,OPENAI}_REASONING_MODEL   every product on that provider
        -> then the whole chat chain (see resolve_model)

    ``product_var`` is the product's *chat* variable (e.g. ``BOOK_ALCHEMY_MODEL``);
    the reasoning variable is derived from it, so callers pass one name."""
    reasoning_var = _reasoning_var_for(product_var)
    own = _clean(os.getenv(reasoning_var)) if reasoning_var else ""
    return own or provider_reasoning_model(provider) or resolve_model(product_var, provider=provider)


def _reasoning_var_for(product_var: str | None) -> str:
    """`BOOK_ALCHEMY_MODEL` -> `BOOK_ALCHEMY_REASONING_MODEL`, via the tables above so
    the mapping stays declared in one place rather than string-munged at each call."""
    if not product_var:
        return ""
    for product, chat_var in PRODUCT_MODEL_VARS.items():
        if chat_var == product_var:
            return PRODUCT_REASONING_MODEL_VARS.get(product, "")
    return ""
