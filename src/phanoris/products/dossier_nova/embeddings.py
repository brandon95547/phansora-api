"""
embeddings.py

Handles:
- SQLite storage of embeddings
- FAISS similarity index
- Local sentence-transformers embedding calls (no HTTP)
"""

import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .config import Config
from .utils import serialize_vector, deserialize_vector, hash_text


# Anchor paths to the package directory so running
# `python -m tomeweaver` from another folder still keeps data
# inside the tomeweaver project.
BASE_DIR = Path(__file__).resolve().parent


class EmbeddingStore:
    def __init__(self, config: Config, db_path: Optional[Union[str, Path]] = None):
        """
        Initialize SQLite connection, local embedding model, and FAISS index.

        Args:
            config: Config instance (provides similarity_threshold, dimensions, etc.).
            db_path: Optional override for the SQLite DB path. If None, defaults
                     to <package_dir>/embeddings.db.
        """
        self.config = config
        self.db_path = Path(db_path) if db_path is not None else BASE_DIR / "embeddings.db"

        # --- SQLite setup ---
        self.conn = sqlite3.connect(str(self.db_path))
        self.cursor = self.conn.cursor()
        self._setup_table()

        # --- Local embedding model (force CPU to avoid CUDA issues) ---
        # Uses the same model you were calling via Hugging Face API,
        # but now runs entirely locally.
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )

        # Prefer explicit dimension from config if present; otherwise, ask the model.
        if getattr(self.config, "dimensions", None):
            self.dim = int(self.config.dimensions)
        else:
            self.dim = int(self.model.get_sentence_embedding_dimension())

        # --- FAISS index (cosine similarity via inner product) ---
        self.index = self._load_faiss_index()

    # ---------- SQLite schema ----------

    def _setup_table(self) -> None:
        """
        Ensure the embeddings table exists.
        """
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                chunk TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
            """
        )
        self.conn.commit()

    # ---------- FAISS index ----------

    def _load_faiss_index(self) -> faiss.IndexFlatIP:
        """
        Build an in-memory FAISS index from everything in SQLite.

        This is important so that:
        - Duplicate detection works across runs.
        - You don't lose past embeddings when you restart the script.
        """
        index = faiss.IndexFlatIP(self.dim)

        self.cursor.execute("SELECT embedding FROM embeddings")
        rows = self.cursor.fetchall()

        if not rows:
            return index

        # Deserialize vectors, filtering out any with incorrect dimensions
        vectors = []
        for (blob,) in rows:
            try:
                vec = deserialize_vector(blob)
                if len(vec) == self.dim:
                    vectors.append(vec)
                else:
                    print(f"[WARN] Skipping malformed vector: expected {self.dim} dims, got {len(vec)}")
            except Exception as e:
                print(f"[WARN] Failed to deserialize vector: {e}")
        
        if not vectors:
            return index

        arr = np.vstack(vectors).astype(np.float32)

        # Normalize so inner product = cosine similarity
        faiss.normalize_L2(arr)
        index.add(arr)

        return index

    # ---------- Embedding API (LOCAL) ----------

    def get_embedding(
        self,
        text: str,
        retries: int = 1,
        delay: int = 0,
    ) -> Optional[np.ndarray]:
        """
        Compute a local embedding for `text` using sentence-transformers.

        Returns:
            1D float32 numpy array of length `self.dim`, L2-normalized.

        `retries` and `delay` are kept for backwards compatibility with
        the old HTTP-based implementation, but they are not used here.
        """
        if not text:
            return None

        # model.encode returns shape (dim,) or (1, dim) depending on options.
        vec = self.model.encode(text, convert_to_numpy=True)
        vec = np.asarray(vec, dtype=np.float32).reshape(1, -1)

        # Normalize so we can use inner product as cosine similarity.
        faiss.normalize_L2(vec)

        # Flatten back to (dim,)
        return vec.reshape(-1)

    # ---------- Duplicate detection ----------

    def is_duplicate(self, emb: np.ndarray, threshold: Optional[float] = None) -> bool:
        """
        Check whether `emb` is too similar to something already stored.

        Uses cosine similarity via FAISS inner-product index.
        If *threshold* is provided it overrides ``config.similarity_threshold``.
        """
        if self.index.ntotal == 0:
            return False

        query = emb.reshape(1, -1).astype(np.float32)
        # Defensive normalize in case caller passed an unnormalized vector
        faiss.normalize_L2(query)

        distances, _ = self.index.search(query, k=1)
        best = float(distances[0][0])

        effective = threshold if threshold is not None else float(self.config.similarity_threshold)
        return best >= effective

    # ---------- Insert new block ----------

    def add_block(self, block_text: str, emb: np.ndarray) -> None:
        """
        Insert a new text block + embedding into SQLite and FAISS.
        """
        block_hash = hash_text(block_text)
        emb_bytes = serialize_vector(emb)

        self.cursor.execute(
            "INSERT OR REPLACE INTO embeddings (id, chunk, embedding) VALUES (?, ?, ?)",
            (block_hash, block_text, emb_bytes),
        )
        self.conn.commit()

        # Embedding should already be normalized; normalize defensively anyway.
        vec = emb.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)
        self.index.add(vec)

    # ---------- Nearest-heading lookup ----------

    def find_nearest_heading(
        self,
        text_emb: np.ndarray,
        headings: List[Tuple[str, str]],
    ) -> Optional[str]:
        """
        Return the heading_id whose title embeds closest to *text_emb*.

        Args:
            text_emb: Normalized embedding of the text to classify.
            headings: List of ``(heading_id, title)`` pairs from the TOC.

        Returns:
            The ``heading_id`` of the best match, or ``None``.
        """
        if not headings:
            return None

        best_id: Optional[str] = None
        best_sim = -1.0

        for heading_id, title in headings:
            heading_emb = self.get_embedding(title)
            if heading_emb is None:
                continue
            sim = float(np.dot(text_emb, heading_emb))
            if sim > best_sim:
                best_sim = sim
                best_id = heading_id

        return best_id

    # ---------- Cleanup ----------

    def close(self) -> None:
        """
        Close the underlying SQLite connection.
        """
        try:
            self.conn.close()
        except Exception:
            pass
