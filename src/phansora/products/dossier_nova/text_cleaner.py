"""
text_cleaner.py

AI-powered text restoration for PDF-extracted / OCR text.

Fixes issues that regex cannot handle:
  - Same-case merged words:  "cannotdevote" → "cannot devote"
  - OCR character errors:    "Catho Hc" → "Catholic"
  - Garbled characters:      "lihrorum" → "librorum"
  - Contextual spacing:      "isnot" → "is not"

Uses the same DeepSeek API already configured for the pipeline.
Text is processed in paragraph-boundary chunks so arbitrarily long
documents work within the model's context window.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any


# ---------------------------------------------------------------------------
# System prompt for the cleanup LLM call
# ---------------------------------------------------------------------------

_CLEANUP_SYSTEM_PROMPT = """\
You are a text restoration specialist. You receive raw text that was \
extracted from a PDF or produced by OCR. Your ONLY job is to fix \
extraction artifacts so the text reads as the original author intended.

WHAT TO FIX:
1. Merged words (missing spaces): "theChurch" → "the Church", \
"cannotdevote" → "cannot devote", "Thisshort" → "This short", \
"isnot" → "is not"
2. OCR character errors: "Catho Hc" → "Catholic", "lihrorum" → \
"librorum", "Biicher" → "Bücher", "verhotenen" → "verbotenen"
3. Punctuation jammed to the next word: ",Books" → ", Books", \
".The" → ". The"
4. Garbled / corrupted characters that are obvious from context
5. Words broken across lines: "m-\\nisit" → "misit"

STRICT RULES:
- Return ONLY the cleaned text — no commentary, no explanations, \
no wrapping.
- DO NOT add, remove, or rearrange any content.
- DO NOT rewrite, paraphrase, or "improve" the writing style.
- DO NOT change the meaning or tone of any passage.
- DO NOT add section headings, bullet points, or formatting that \
was not in the original.
- Preserve all paragraph breaks (blank lines) exactly as they appear.
- Preserve all numbers, dates, prices, and references exactly.
- Preserve foreign-language words and titles unless they are clearly \
garbled OCR output.
- When uncertain whether something is an error or intentional, \
LEAVE IT AS-IS.\
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_extracted_text(
    text: str,
    client: Any,
    chunk_size: int = 6000,
) -> str:
    """
    Send extracted text through DeepSeek to fix OCR / extraction artifacts.

    Processes text in chunks (split at paragraph boundaries) so the model
    never receives more than *chunk_size* characters in a single request.

    Parameters
    ----------
    text : str
        The raw extracted text.
    client : Any
        An OpenAI-compatible client (``openai.OpenAI`` instance).
    chunk_size : int
        Maximum characters per cleanup chunk.  Larger values give the LLM
        more context but use more tokens.

    Returns
    -------
    str
        The cleaned text, or the original text if every chunk fails.
    """
    if not text or not text.strip():
        return text

    chunks = _split_for_cleaning(text.strip(), chunk_size)
    cleaned: list[str] = []
    failures = 0

    # Parallel cleanup — API calls are I/O-bound, safe to overlap
    def _do_clean(idx_chunk: tuple[int, str]) -> tuple[int, str | None]:
        idx, chunk = idx_chunk
        print(f"[CLEAN] Cleaning chunk {idx}/{len(chunks)} ({len(chunk)} chars)...")
        return idx, _clean_chunk(chunk, client)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(_do_clean, enumerate(chunks, 1)))

    for idx, result in results:
        if result is None:
            failures += 1
            cleaned.append(chunks[idx - 1])  # keep original on failure
        else:
            cleaned.append(result)

    if failures:
        print(f"[CLEAN] ⚠️  {failures}/{len(chunks)} chunk(s) fell back to originals.")

    return "\n\n".join(cleaned)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_for_cleaning(text: str, max_chars: int) -> list[str]:
    """Split *text* into chunks at paragraph (double-newline) boundaries."""
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2          # +2 accounts for the "\n\n" joiner
        if current_len + para_len > max_chars and current:
            chunks.append("\n\n".join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


def _clean_chunk(chunk: str, client: Any) -> str | None:
    """
    Send a single chunk to DeepSeek for restoration.

    Returns the cleaned text, or ``None`` if the call fails or returns a
    suspiciously short result (which would indicate truncation / refusal).
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": _CLEANUP_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Restore the following extracted text. Return ONLY "
                        "the corrected text with no other output.\n\n"
                        f"{chunk}"
                    ),
                },
            ],
            temperature=0.1,
        )

        result = (response.choices[0].message.content or "").strip()

        # Safety: if the model returned significantly less text, it
        # probably hallucinated or truncated — keep the original.
        if not result or len(result) < len(chunk) * 0.5:
            print(
                f"  -> Cleanup returned suspicious length "
                f"({len(result)} chars vs {len(chunk)} original), keeping original."
            )
            return None

        return result
    except Exception as e:
        print(f"  -> Cleanup failed: {e}")
        return None
