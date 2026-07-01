import os
from pathlib import Path

from dotenv import load_dotenv
import openai

from phanoris.shared.paths import runtime_root

# Default input/output paths live under the runtime root (CWD / PHANORIS_DATA_DIR),
# not inside the installed package. Both are overridable via env.
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
        self.similarity_threshold = _get_float_env("SIMILARITY_THRESHOLD", 0.75)
        self.dimensions = _get_int_env("EMBEDDING_DIMENSIONS", 384)
        self.input_text_path = os.getenv("INPUT_TEXT_PATH", str(BASE_DIR / "tmp" / "pizza.txt"))
        self.toc_full_path = os.getenv("TOC_FULL_PATH", str(BASE_DIR / "toc" / "full.md"))
        self.max_chunk_chars = _get_int_env("MAX_CHUNK_CHARS", 20000)
        self.conservative_mode = os.getenv("CONSERVATIVE_MODE", "false").lower() in ("true", "1", "yes")
        self.catchall_heading = os.getenv("CATCHALL_HEADING", "Miscellaneous")
        self.toc_target_heading_count = _get_int_env("TOC_TARGET_HEADING_COUNT", 60)
        self.content_similarity_threshold = _get_float_env("CONTENT_SIMILARITY_THRESHOLD", 0.95)
        self.overlap_paragraphs = _get_int_env("OVERLAP_PARAGRAPHS", 2)
        self.coverage_threshold = _get_float_env("COVERAGE_THRESHOLD", 0.85)
        self.clean_extracted_text = os.getenv("CLEAN_EXTRACTED_TEXT", "true").lower() in ("true", "1", "yes")
        self.cleanup_chunk_size = _get_int_env("CLEANUP_CHUNK_SIZE", 12000)
        # Cosine-similarity threshold below which two adjacent paragraphs are
        # treated as a topic boundary in semantic chunking. MiniLM puts most
        # adjacent article paragraphs in the ~0.3–0.6 range, so the old default of
        # 0.5 split almost every paragraph into its own chunk (dozens of tiny
        # fragments → poor coverage, heavy overlap duplication, slow organize).
        # 0.2 only splits on genuine topic shifts.
        self.semantic_similarity_drop = _get_float_env("SEMANTIC_SIMILARITY_DROP", 0.2)

        # --- Dossier pipeline settings ---
        # Maximum share of the final dossier any single source may occupy (0.0–1.0).
        self.max_source_share = _get_float_env("MAX_SOURCE_SHARE", 0.40)
        # Cosine-similarity threshold for claim-level deduplication across sources.
        self.claim_dedup_threshold = _get_float_env("CLAIM_DEDUP_THRESHOLD", 0.90)
        # Maximum duplication ratio allowed before the validation step triggers a warning.
        self.max_duplication_ratio = _get_float_env("MAX_DUPLICATION_RATIO", 0.15)
        # Whether to run source profiling (type + role classification) via LLM.
        self.enable_source_profiling = os.getenv("ENABLE_SOURCE_PROFILING", "true").lower() in ("true", "1", "yes")
        # Maximum chars sent to the LLM per source for profiling (first N chars).
        self.profile_sample_chars = _get_int_env("PROFILE_SAMPLE_CHARS", 4000)

    @classmethod
    def from_env(cls):
        return cls()
