"""
utils/chunker.py - Custom token-aware text chunking with sliding window overlap.

=== CHUNKING STRATEGY EXPLAINED ===

Why 400 tokens per chunk?
  - Embedding models like all-MiniLM-L6-v2 have a max sequence length of 256 *word pieces*,
    but we use a slightly larger logical chunk (400 whitespace tokens ≈ 300-350 word pieces)
    because truncation by the model still captures the dominant semantic signal.
  - Empirically, 400 tokens captures ~1-2 full paragraphs — enough context
    for the embedding to represent a coherent idea, but small enough that the
    vector isn't diluted by multiple unrelated topics.
  - Chunks that are too small (< 100 tokens) produce noisy embeddings;
    chunks that are too large (> 1000 tokens) blend multiple topics and hurt
    retrieval precision.

Why 80 tokens of overlap (20%)?
  - Sentences that straddle chunk boundaries would otherwise be split,
    causing incomplete information in both chunks.
  - 20% overlap is a well-studied sweet-spot: it preserves boundary context
    without duplicating too much data (which inflates index size and degrades
    retrieval by returning near-duplicate chunks).

Tokenization approach:
  - We use simple whitespace splitting (not BPE) for chunk boundaries because:
    1. It's deterministic and fast.
    2. The embedding model handles sub-word tokenization internally.
    3. Whitespace tokens correlate well enough with model tokens for
       boundary decisions (±10% variance, acceptable for chunking).
"""

from typing import List, Dict, Any

from utils.logger import get_logger

logger = get_logger(__name__)


def chunk_text(
    text: str,
    source_filename: str,
    chunk_size: int = 400,
    chunk_overlap: int = 80,
) -> List[Dict[str, Any]]:
    """
    Splits text into overlapping chunks using a sliding window over whitespace tokens.

    Algorithm:
      1. Tokenize text by whitespace → list of tokens.
      2. Start at token index 0.
      3. Slice tokens[start : start + chunk_size] → one chunk.
      4. Advance start by (chunk_size - chunk_overlap) → next window.
      5. Repeat until start >= len(tokens).

    Args:
        text:            Raw document text to chunk.
        source_filename: Original filename (stored as metadata for retrieval).
        chunk_size:      Number of whitespace tokens per chunk.
        chunk_overlap:   Number of tokens shared between consecutive chunks.

    Returns:
        List of dicts, each containing:
          - chunk_id:    Unique ID formatted as "{filename}_chunk_{index}"
          - text:        The chunk text (reconstructed from tokens)
          - source:      Source filename
          - token_count: Number of tokens in this chunk
    """
    if not text or not text.strip():
        logger.warning(f"Empty text received for '{source_filename}'. No chunks produced.")
        return []

    # Step 1: Whitespace tokenization
    tokens = text.split()
    total_tokens = len(tokens)
    logger.info(f"Chunking '{source_filename}': {total_tokens} tokens, "
                f"chunk_size={chunk_size}, overlap={chunk_overlap}")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"Overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})."
        )

    chunks: List[Dict[str, Any]] = []
    step = chunk_size - chunk_overlap  # How far the window advances each iteration
    start = 0
    chunk_index = 0

    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text_str = " ".join(chunk_tokens)

        chunk_id = f"{source_filename}_chunk_{chunk_index}"

        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text_str,
            "source": source_filename,
            "token_count": len(chunk_tokens),
        })

        chunk_index += 1
        start += step

    logger.info(f"Produced {len(chunks)} chunks from '{source_filename}'.")
    return chunks
