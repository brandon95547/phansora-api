import os
from typing import Dict, List

from phansora.shared.ai.deepseek import DeepSeekChatConfig, chat_sync
from phansora.shared.paths import runtime_root

# The default report path lives under the runtime root (CWD / PHANSORA_DATA_DIR), not
# inside the installed package. Overridable via TOC_FULL_PATH.
BASE_DIR = runtime_root()


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class DeepSeekClient:
    """Research Atlas's calls to DeepSeek, through the shared helper in shared.ai.deepseek.

    What is fixed here is what every call in a run shares: how long one may take, how
    often it is tried, and how much it may write. The model is read on each call
    (RESEARCH_ATLAS_MODEL, then DEEPSEEK_MODEL), so a missing one fails that call, which
    the stages already survive, rather than the whole run.
    """

    def __init__(self, *, timeout_s: float, max_retries: int, max_tokens: int) -> None:
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.max_tokens = max_tokens

    def ask(self, messages: List[Dict[str, str]], *, temperature: float) -> str:
        """The reply to ``messages``, as text. Raises on any failure."""
        cfg = DeepSeekChatConfig.from_env(product_var="RESEARCH_ATLAS_MODEL")
        text, _finish = chat_sync(
            cfg=cfg,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=temperature,
            timeout_s=self.timeout_s,
            max_retries=self.max_retries,
        )
        return text


class Config:
    def __init__(self):
        # Bound every LLM call. Left to its defaults, a single slow/hung DeepSeek call
        # could stall a whole generation for ten minutes. Fail fast and retry once.
        #
        # The thinking is off (the helper's default) and every reply is capped, like the
        # rest of the DeepSeek callers: the v4 models reason by default and bill it against
        # the output, which is spent before an extraction record is written. Tokens are
        # billed as used, so a generous cap costs nothing; 8000 is within what every
        # DeepSeek chat model accepts.
        self.deepseek_client = DeepSeekClient(
            timeout_s=_get_float_env("DEEPSEEK_TIMEOUT_SECONDS", 90.0),
            max_retries=_get_int_env("DEEPSEEK_MAX_RETRIES", 1),
            max_tokens=_get_int_env("RESEARCH_ATLAS_MAX_TOKENS", 8000),
        )
        self.toc_full_path = os.getenv("TOC_FULL_PATH", str(BASE_DIR / "toc" / "full.md"))
        self.max_chunk_chars = _get_int_env("MAX_CHUNK_CHARS", 20000)
        self.overlap_paragraphs = _get_int_env("OVERLAP_PARAGRAPHS", 2)

        # How many LLM calls are in flight at once, across every stage. These calls are
        # pure network wait, so the ceiling is the provider's rate limit rather than this
        # box's CPU — 8 was leaving most of the window unused on multi-source dossiers.
        self.llm_max_workers = _get_int_env("LLM_MAX_WORKERS", 16)

        self.clean_extracted_text = os.getenv("CLEAN_EXTRACTED_TEXT", "true").lower() in ("true", "1", "yes")
        self.cleanup_chunk_size = _get_int_env("CLEANUP_CHUNK_SIZE", 12000)
        # Skip the cleanup pass on sources that show no extraction damage. Costs one cheap
        # regex scan and saves an entire LLM stage on any dossier built from article URLs.
        self.skip_clean_when_undamaged = os.getenv(
            "SKIP_CLEAN_WHEN_UNDAMAGED", "true"
        ).lower() in ("true", "1", "yes")

        # Whether to run source profiling (type + role classification) via LLM.
        self.enable_source_profiling = os.getenv("ENABLE_SOURCE_PROFILING", "true").lower() in ("true", "1", "yes")
        # Maximum chars sent to the LLM per source for profiling (first N chars).
        self.profile_sample_chars = _get_int_env("PROFILE_SAMPLE_CHARS", 4000)

    @classmethod
    def from_env(cls):
        return cls()
