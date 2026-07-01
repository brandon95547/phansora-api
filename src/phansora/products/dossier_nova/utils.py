"""
utils.py

General utility functions used across the project:
- Hashing text
- Serializing / deserializing vectors
- Loading text files
- Splitting large text into reasonably sized chunks
"""

import hashlib
import re
from pathlib import Path
from typing import List, Union

import numpy as np


# ---------- Hashing ----------

def hash_text(text: str) -> str:
    """
    Return a stable SHA-256 hex digest for the given text.
    Used as a unique ID for content blocks.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------- Embedding (vector) serialization ----------

def serialize_vector(vec: Union[List[float], np.ndarray]) -> bytes:
    """
    Serialize a vector (list or NumPy array) into bytes suitable for SQLite storage.

    - Ensures dtype float32
    - Returns raw bytes via .tobytes()
    """
    arr = np.array(vec, dtype=np.float32)
    return arr.tobytes()


def deserialize_vector(blob: bytes) -> np.ndarray:
    """
    Deserialize bytes from SQLite back into a float32 NumPy array.
    """
    return np.frombuffer(blob, dtype=np.float32)


# ---------- File loading ----------

def load_text(path: Union[str, Path]) -> str:
    """
    Load a UTF-8 text file and return its full contents as a string.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return f.read()


# ---------- Text chunking ----------

def split_text_into_chunks(text: str, max_chars: int = 10000) -> List[str]:
    """
    Split a large text into smaller chunks no longer than `max_chars`,
    preferably breaking at sentence or line boundaries.

    Strategy:
    - While text is longer than max_chars:
        - Try to split at the last '.' before max_chars
        - If no '.', try the last newline before max_chars
        - If neither is found, hard-split at max_chars
    - Strip leading/trailing whitespace from each chunk.
    """
    chunks: List[str] = []
    text = text.strip()

    while len(text) > max_chars:
        # Prefer splitting at a period before max_chars
        split_at = text.rfind('.', 0, max_chars)
        if split_at == -1:
            # Fallback: split at newline
            split_at = text.rfind('\n', 0, max_chars)
        if split_at == -1:
            # As a last resort, hard-split at max_chars
            split_at = max_chars

        # Include the delimiter (e.g., '.') in the chunk
        chunk = text[: split_at + 1].strip()
        if chunk:
            chunks.append(chunk)

        # Remaining text
        text = text[split_at + 1 :].strip()

    # Any leftover text becomes the final chunk
    if text:
        chunks.append(text)

    return chunks


# ---------- Paragraph splitting ----------

def split_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs on double-newline boundaries.
    Never breaks mid-paragraph.  Strips empty results.
    """
    raw = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in raw if p.strip()]


# ---------- Semantic-aware chunking with overlap ----------

def split_text_semantic(
    text: str,
    embedding_fn=None,
    max_chars: int = 10_000,
    overlap_paragraphs: int = 2,
    similarity_drop: float = 0.5,
) -> List[str]:
    """
    Split text into semantically coherent chunks with cross-boundary overlap.

    Strategy:
    1. Split into paragraphs (never break mid-paragraph).
    2. If *embedding_fn* is provided, detect topic-shift boundaries by
       looking for drops in cosine similarity between consecutive paragraphs.
    3. Group adjacent paragraphs into chunks that respect *max_chars*.
    4. Prepend *overlap_paragraphs* trailing paragraphs from the previous
       chunk so the LLM always has cross-boundary context.

    Falls back to simple paragraph grouping when no embedding function
    is provided.
    """
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []

    # --- Detect semantic boundaries ---
    boundaries = set()
    if embedding_fn is not None and len(paragraphs) > 1:
        embeddings = [embedding_fn(p) for p in paragraphs]
        for i in range(len(embeddings) - 1):
            a, b = embeddings[i], embeddings[i + 1]
            if a is None or b is None:
                continue
            norm_a = float(np.linalg.norm(a))
            norm_b = float(np.linalg.norm(b))
            if norm_a == 0 or norm_b == 0:
                continue
            sim = float(np.dot(a, b) / (norm_a * norm_b))
            if sim < similarity_drop:
                boundaries.add(i + 1)

    # --- Group paragraphs into chunks ---
    groups: List[List[int]] = []
    current_group: List[int] = []
    current_chars = 0

    for i, para in enumerate(paragraphs):
        para_len = len(para)
        # Start a new group on semantic boundary OR size overflow
        if current_group and (i in boundaries or current_chars + para_len + 2 > max_chars):
            groups.append(current_group)
            current_group = []
            current_chars = 0
        current_group.append(i)
        current_chars += para_len + 2  # +2 for paragraph separator

    if current_group:
        groups.append(current_group)

    # --- Assemble chunks with overlap ---
    chunks: List[str] = []
    for gi, group in enumerate(groups):
        parts: List[str] = []
        # Overlap: include trailing paragraphs from the previous group
        if gi > 0 and overlap_paragraphs > 0:
            prev = groups[gi - 1]
            for oi in prev[-overlap_paragraphs:]:
                parts.append(paragraphs[oi])

        for pi in group:
            parts.append(paragraphs[pi])

        joined = "\n\n".join(parts)
        if joined.strip():
            chunks.append(joined.strip())

    return chunks if chunks else [text.strip()]
