"""
services/retrieval.py - Query embedding and FAISS similarity search.

=== RETRIEVAL PROCESS ===

1. The user's question is embedded using the same SentenceTransformer model
   that was used to embed the document chunks (all-MiniLM-L6-v2).

2. The query embedding is passed to the FAISS index, which computes the
   cosine similarity (via inner product on L2-normalized vectors) between
   the query and every stored chunk vector.

3. FAISS returns the top-k chunks with the highest similarity scores.

4. We apply a minimum similarity threshold (default: 0.3). If ALL returned
   chunks score below this threshold, we log a warning and return a fallback
   response — this prevents the LLM from hallucinating based on irrelevant context.

5. Chunks above the threshold are returned with their metadata and scores
   for context construction and transparency.
"""

import time
from typing import List, Tuple, Dict, Any

import numpy as np

from config import settings
from utils.logger import get_logger
from services.ingestion import get_embedding_model
from vector_store.faiss_store import get_faiss_store

logger = get_logger(__name__)


def retrieve_relevant_chunks(
    question: str,
    top_k: int = 5,
    min_similarity: float | None = None,
) -> Tuple[List[Dict[str, Any]], List[float], float]:
    """
    Embed a user question and retrieve the most relevant document chunks.

    Args:
        question:       The user's natural language question.
        top_k:          Number of chunks to retrieve from FAISS.
        min_similarity: Minimum cosine similarity to consider a chunk relevant.
                        Defaults to settings.MIN_SIMILARITY_THRESHOLD (0.3).

    Returns:
        Tuple of:
          - chunks:     List of metadata dicts (chunk_id, text, source, token_count)
          - scores:     List of similarity scores (parallel to chunks)
          - latency_ms: Time taken for embedding + search in milliseconds

    === FAILURE CASE HANDLING ===
    When all retrieved chunks have similarity scores below min_similarity:
      - This indicates the user's question is unrelated to any indexed content.
      - We log a WARNING with the question and scores for debugging.
      - We return empty chunks, signaling the LLM layer to respond with
        "I don't know" instead of generating a plausible-sounding but
        unsupported answer.
    """
    if min_similarity is None:
        min_similarity = settings.MIN_SIMILARITY_THRESHOLD

    start_time = time.time()

    # ── Step 1: Embed the query ──
    model = get_embedding_model()
    query_embedding = model.encode([question], show_progress_bar=False)
    query_vector = np.array(query_embedding, dtype=np.float32)

    # ── Step 2: FAISS similarity search ──
    store = get_faiss_store()
    results = store.search(query_vector, k=top_k)

    latency_ms = (time.time() - start_time) * 1000

    if not results:
        logger.warning(f"No results from FAISS for query: '{question[:100]}'")
        return [], [], latency_ms

    # ── Step 3: Extract chunks and scores ──
    chunks = [r[0] for r in results]
    scores = [r[1] for r in results]

    # ── Step 4: Log similarity scores for metrics tracking ──
    logger.info(
        f"Retrieval complete: query='{question[:80]}...', "
        f"top_scores={[round(s, 4) for s in scores]}, "
        f"latency={latency_ms:.1f}ms"
    )

    # ── Step 5: Check for irrelevant results (failure case) ──
    relevant_chunks = []
    relevant_scores = []

    for chunk, score in zip(chunks, scores):
        if score >= min_similarity:
            relevant_chunks.append(chunk)
            relevant_scores.append(score)
        else:
            # Log each low-scoring chunk for debugging and metrics
            logger.warning(
                f"Low similarity chunk filtered out: "
                f"chunk_id='{chunk['chunk_id']}', score={score:.4f}, "
                f"threshold={min_similarity}"
            )

    if not relevant_chunks:
        # ALL chunks scored below threshold — total retrieval failure
        logger.warning(
            f"RETRIEVAL FAILURE: All {len(chunks)} chunks scored below "
            f"threshold ({min_similarity}) for query: '{question}'. "
            f"Scores were: {[round(s, 4) for s in scores]}. "
            f"Returning fallback 'I don't know' response."
        )
        return [], [], latency_ms

    logger.info(
        f"Relevant chunks: {len(relevant_chunks)}/{len(chunks)} passed "
        f"threshold ({min_similarity})"
    )
    return relevant_chunks, relevant_scores, latency_ms
