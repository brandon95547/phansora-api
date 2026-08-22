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

# ---------- Text chunking ----------

# ---------- Paragraph splitting ----------

def split_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs on double-newline boundaries.
    Never breaks mid-paragraph.  Strips empty results.
    """
    raw = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in raw if p.strip()]


# ---------- Semantic-aware chunking with overlap ----------
