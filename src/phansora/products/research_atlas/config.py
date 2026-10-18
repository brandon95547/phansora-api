import os

from dotenv import load_dotenv
import openai

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


class Config:
    def __init__(self):
        load_dotenv()

        api_key = os.getenv("DEEPSEEK_API_KEY")
        # Bound every LLM call. Without these the OpenAI SDK defaults to a 600s
        # (10-minute) timeout with 2 retries, so a single slow/hung DeepSeek call
        # can stall a whole generation for ten minutes. Fail fast and retry once.
        self.deepseek_client = openai.OpenAI(
            api_key=api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            timeout=_get_float_env("DEEPSEEK_TIMEOUT_SECONDS", 90.0),
            max_retries=_get_int_env("DEEPSEEK_MAX_RETRIES", 1),
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
