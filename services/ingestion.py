"""
services/ingestion.py - Background document ingestion pipeline.

This module orchestrates the full pipeline that runs in a background thread:
  1. Extract text from the uploaded file (PDF or TXT).
  2. Chunk the text using the sliding window strategy.
  3. Generate embeddings for each chunk using SentenceTransformers.
  4. Add the embeddings and metadata to the FAISS index.
  5. Persist the updated index to disk.

The pipeline runs in a separate Python thread so that the /upload endpoint
returns immediately (non-blocking). Document status is tracked in an
in-memory dict so clients can poll for completion.
"""

import threading
import time
from typing import Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer

from config import settings
from utils.logger import get_logger
from utils.pdf_parser import extract_text
from utils.chunker import chunk_text
from vector_store.faiss_store import get_faiss_store

logger = get_logger(__name__)

# ── Document Status Tracking ─────────────────────────────────────────────────
# In-memory dict mapping file_id → status dict.
# In production, this would be backed by Redis or a database.
document_statuses: Dict[str, Dict[str, Any]] = {}
_status_lock = threading.Lock()


def update_status(file_id: str, **kwargs: Any) -> None:
    """Thread-safe update of a document's processing status."""
    with _status_lock:
        if file_id in document_statuses:
            document_statuses[file_id].update(kwargs)


def get_status(file_id: str) -> Dict[str, Any] | None:
    """Thread-safe retrieval of a document's processing status."""
    with _status_lock:
        return document_statuses.get(file_id, None)


def register_document(file_id: str, filename: str) -> None:
    """Register a new document as queued for processing."""
    with _status_lock:
        document_statuses[file_id] = {
            "file_id": file_id,
            "filename": filename,
            "status": "queued",
            "chunk_count": 0,
            "error": None,
        }


# ── Embedding Model (lazy singleton) ─────────────────────────────────────────
# The model is loaded once on first use and reused across all ingestion jobs.
# all-MiniLM-L6-v2 produces 384-dimensional embeddings and is ~80MB.

_embedding_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load the SentenceTransformer model (thread-safe)."""
    global _embedding_model
    if _embedding_model is None:
        with _model_lock:
            if _embedding_model is None:   # double-checked locking
                logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}...")
                _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
                logger.info("Embedding model loaded successfully.")
    return _embedding_model


# ── Ingestion Pipeline ───────────────────────────────────────────────────────

def _run_ingestion(file_id: str, filepath: str, filename: str) -> None:
    """
    Full ingestion pipeline. Runs in a background thread.

    Steps:
      1. Parse document → raw text
      2. Chunk text → list of chunk dicts
      3. Embed chunks → numpy array of shape (N, 384)
      4. Add to FAISS → index + metadata
      5. Save index to disk
    """
    start_time = time.time()
    update_status(file_id, status="processing")

    try:
        # ── Step 1: Extract text ──
        logger.info(f"[{file_id}] Step 1/5: Extracting text from '{filename}'...")
        raw_text = extract_text(filepath)

        if not raw_text.strip():
            raise ValueError(f"No text could be extracted from '{filename}'.")

        # ── Step 2: Chunk text ──
        logger.info(f"[{file_id}] Step 2/5: Chunking text...")
        chunks = chunk_text(
            text=raw_text,
            source_filename=filename,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

        if not chunks:
            raise ValueError(f"Chunking produced 0 chunks for '{filename}'.")

        # ── Step 3: Generate embeddings ──
        logger.info(f"[{file_id}] Step 3/5: Embedding {len(chunks)} chunks...")
        model = get_embedding_model()
        chunk_texts = [c["text"] for c in chunks]
        embeddings = model.encode(chunk_texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)

        # ── Step 4: Add to FAISS index ──
        logger.info(f"[{file_id}] Step 4/5: Adding to FAISS index...")
        store = get_faiss_store()
        metadata_list = [
            {
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "source": c["source"],
                "token_count": c["token_count"],
            }
            for c in chunks
        ]
        store.add(embeddings, metadata_list)

        # ── Step 5: Persist index ──
        logger.info(f"[{file_id}] Step 5/5: Saving index to disk...")
        store.save(settings.FAISS_INDEX_PATH, settings.FAISS_META_PATH)

        elapsed = time.time() - start_time
        update_status(file_id, status="done", chunk_count=len(chunks))
        logger.info(
            f"[{file_id}] Ingestion complete: {len(chunks)} chunks, "
            f"{elapsed:.2f}s elapsed."
        )

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)
        update_status(file_id, status="failed", error=error_msg)
        logger.error(f"[{file_id}] Ingestion failed after {elapsed:.2f}s: {error_msg}")


def start_ingestion(file_id: str, filepath: str, filename: str) -> None:
    """
    Launch the ingestion pipeline in a background thread.
    Returns immediately so the API is non-blocking.
    """
    register_document(file_id, filename)

    thread = threading.Thread(
        target=_run_ingestion,
        args=(file_id, filepath, filename),
        daemon=True,  # thread dies when main process exits
        name=f"ingest-{file_id[:8]}",
    )
    thread.start()
    logger.info(f"Background ingestion started for '{filename}' (file_id={file_id}).")
