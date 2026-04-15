"""
vector_store/faiss_store.py - FAISS vector index wrapper with metadata storage.

=== HOW FAISS INDEXING WORKS ===

FAISS (Facebook AI Similarity Search) is a library for efficient similarity search
over dense vectors. Here's how our setup works:

1. Index Type: IndexFlatIP (Inner Product)
   - We use Inner Product (dot product) as the similarity metric.
   - Since we L2-normalize all vectors before adding them, the inner product
     equals cosine similarity: cos(a, b) = dot(a/|a|, b/|b|).
   - IndexFlatIP performs exact (brute-force) search — no approximation.
   - For our use case (< 100K vectors), brute-force is fast enough (~1ms).
   - For 1M+ vectors, switch to IndexIVFFlat or IndexHNSW for approximate search.

2. Storage:
   - Vectors are stored in a flat numpy array inside FAISS.
   - Metadata (chunk text, source filename, chunk_id) is stored in a parallel
     Python list and serialized to JSON separately.
   - Both must be saved/loaded together to maintain consistency.

3. Thread Safety:
   - All read/write operations are guarded by a threading.Lock because
     FastAPI may serve requests concurrently in the same process, and
     the background ingestion thread writes to the index while queries read from it.
"""

import json
import os
import threading
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import faiss

from utils.logger import get_logger

logger = get_logger(__name__)


class FAISSStore:
    """
    Thread-safe wrapper around a FAISS IndexFlatIP index with parallel metadata storage.

    Attributes:
        dimension:  Embedding dimensionality (384 for all-MiniLM-L6-v2).
        index:      The FAISS index object.
        metadata:   Parallel list of dicts — metadata[i] corresponds to index vector i.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        # IndexFlatIP = exact inner product search (cosine similarity after L2 norm)
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        self.metadata: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        logger.info(f"FAISS store initialized: dimension={dimension}, type=IndexFlatIP")

    def add(
        self,
        vectors: np.ndarray,
        metadata_list: List[Dict[str, Any]],
    ) -> None:
        """
        Add L2-normalized vectors and their metadata to the index.

        Args:
            vectors:       (N, dimension) float32 numpy array of embeddings.
            metadata_list: List of N dicts with keys: chunk_id, text, source.

        The vectors are L2-normalized in-place so that inner product = cosine similarity.
        """
        if vectors.shape[0] != len(metadata_list):
            raise ValueError(
                f"Vector count ({vectors.shape[0]}) != metadata count ({len(metadata_list)})"
            )

        # L2-normalize so dot product ≡ cosine similarity
        faiss.normalize_L2(vectors)

        with self._lock:
            self.index.add(vectors)
            self.metadata.extend(metadata_list)

        logger.info(
            f"Added {vectors.shape[0]} vectors to FAISS index. "
            f"Total vectors: {self.index.ntotal}"
        )

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search the index for the top-k most similar vectors to the query.

        Args:
            query_vector: (1, dimension) float32 numpy array (will be L2-normalized).
            k:            Number of nearest neighbors to retrieve.

        Returns:
            List of (metadata_dict, similarity_score) tuples, sorted by descending score.
            If the index has fewer than k vectors, returns all available results.

        === RETRIEVAL PROCESS ===
        1. The query vector is L2-normalized (same as stored vectors).
        2. FAISS computes dot product between the query and every stored vector.
        3. Since both are unit vectors, dot product = cosine similarity ∈ [-1, 1].
        4. FAISS returns the k highest-scoring vector indices and their scores.
        5. We map those indices back to metadata (chunk text, source, chunk_id).
        """
        # Normalize the query vector
        faiss.normalize_L2(query_vector)

        with self._lock:
            if self.index.ntotal == 0:
                logger.warning("FAISS search called on empty index.")
                return []

            # Clamp k to available vectors
            actual_k = min(k, self.index.ntotal)

            # scores shape: (1, k), indices shape: (1, k)
            scores, indices = self.index.search(query_vector, actual_k)

        results: List[Tuple[Dict[str, Any], float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue  # FAISS returns -1 for unfilled slots
            results.append((self.metadata[idx], float(score)))

        logger.debug(
            f"FAISS search: k={actual_k}, results={len(results)}, "
            f"scores={[round(r[1], 4) for r in results]}"
        )
        return results

    @property
    def total_vectors(self) -> int:
        """Number of vectors currently in the index."""
        return self.index.ntotal

    def save(self, index_path: str, meta_path: str) -> None:
        """
        Persist the FAISS index and metadata to disk.

        Args:
            index_path: File path for the binary FAISS index.
            meta_path:  File path for the JSON metadata file.
        """
        with self._lock:
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            faiss.write_index(self.index, index_path)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False)

        logger.info(
            f"FAISS index saved: {index_path} ({self.index.ntotal} vectors), "
            f"metadata: {meta_path}"
        )

    def load(self, index_path: str, meta_path: str) -> bool:
        """
        Load a previously saved FAISS index and metadata from disk.

        Returns:
            True if loaded successfully, False if files don't exist.
        """
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            logger.info("No existing FAISS index found. Starting fresh.")
            return False

        with self._lock:
            self.index = faiss.read_index(index_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        logger.info(
            f"FAISS index loaded: {self.index.ntotal} vectors from '{index_path}'"
        )
        return True


# ── Global singleton ─────────────────────────────────────────────────────────

_store: Optional[FAISSStore] = None


def get_faiss_store() -> FAISSStore:
    """Get or create the global FAISSStore singleton."""
    global _store
    if _store is None:
        _store = FAISSStore(dimension=384)
    return _store
