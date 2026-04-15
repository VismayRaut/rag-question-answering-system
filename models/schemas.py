"""
models/schemas.py - Pydantic models for all request and response bodies.
Using strict typing ensures data validation at the API boundary.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ── Upload ─────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Returned immediately after a file is accepted for background processing."""
    status: str = Field(..., example="processing")
    file_id: str = Field(..., example="a1b2c3d4-...")
    filename: str = Field(..., example="document.pdf")
    message: str = Field(..., example="File received. Processing in background.")


class DocumentStatus(BaseModel):
    """Tracks the ingestion state of an uploaded document."""
    file_id: str
    filename: str
    status: str          # "queued" | "processing" | "done" | "failed"
    chunk_count: int = 0
    error: Optional[str] = None


# ── Query ──────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Incoming query from the user."""
    question: str = Field(..., min_length=3, max_length=1000,
                          example="What is the main topic of the document?")
    top_k: int = Field(default=5, ge=1, le=20,
                       description="Number of chunks to retrieve (1-20)")


class RetrievedChunk(BaseModel):
    """A single chunk returned from FAISS similarity search."""
    chunk_id: str
    source: str          # filename the chunk came from
    text: str            # raw chunk text
    similarity_score: float = Field(...,
        description="Cosine similarity between query and chunk (0-1)")


class QueryResponse(BaseModel):
    """Full response returned to the user after a query."""
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    latency_ms: float = Field(...,
        description="Total pipeline latency in milliseconds")
    similarity_scores: List[float] = Field(...,
        description="Raw similarity scores for transparency")


# ── Health ──────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    llm_provider: str
