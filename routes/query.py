"""
routes/query.py - Question answering endpoint.

POST /query
  - Accepts a user question (and optional top_k parameter).
  - Rate-limited per IP (default: 5 requests/minute).
  - Runs the full RAG pipeline:
      1. Embed the question
      2. Retrieve relevant chunks from FAISS
      3. Generate an answer using the configured LLM
  - Returns the answer, retrieved chunks, similarity scores, and latency.
"""

import time
from typing import List

from fastapi import APIRouter, Request

from models.schemas import QueryRequest, QueryResponse, RetrievedChunk
from services.retrieval import retrieve_relevant_chunks
from services.llm import generate_answer
from utils.rate_limiter import get_rate_limiter
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question about uploaded documents",
    description="Embeds the question, retrieves the most relevant document chunks, "
                "and generates an answer using the configured LLM provider.",
)
async def query_documents(body: QueryRequest, request: Request) -> QueryResponse:
    """
    Full RAG query pipeline.

    Steps:
      1. Rate limit check (per client IP).
      2. Retrieve relevant chunks via embedding + FAISS search.
      3. Generate answer via LLM using retrieved context.
      4. Return structured response with transparency data.

    Metrics tracked:
      - Query latency (embedding + retrieval + LLM, in ms)
      - Similarity scores of retrieved chunks
      - Warnings logged for low-score or empty retrievals
    """
    pipeline_start = time.time()

    # ── Step 1: Rate limiting ──
    limiter = get_rate_limiter()
    limiter.check(request)

    # ── Step 2: Retrieve relevant chunks ──
    chunks, scores, retrieval_latency_ms = retrieve_relevant_chunks(
        question=body.question,
        top_k=body.top_k,
    )

    # ── Step 3: Generate answer ──
    answer = generate_answer(
        chunks=chunks,
        scores=scores,
        question=body.question,
    )

    # ── Step 4: Build response ──
    total_latency_ms = (time.time() - pipeline_start) * 1000

    retrieved_chunks: List[RetrievedChunk] = [
        RetrievedChunk(
            chunk_id=chunk["chunk_id"],
            source=chunk["source"],
            text=chunk["text"],
            similarity_score=round(score, 4),
        )
        for chunk, score in zip(chunks, scores)
    ]

    # ── Metrics logging ──
    logger.info(
        f"QUERY METRICS | question='{body.question[:80]}' | "
        f"retrieval_latency={retrieval_latency_ms:.1f}ms | "
        f"total_latency={total_latency_ms:.1f}ms | "
        f"chunks_returned={len(retrieved_chunks)} | "
        f"scores={[round(s, 4) for s in scores]}"
    )

    return QueryResponse(
        question=body.question,
        answer=answer,
        retrieved_chunks=retrieved_chunks,
        latency_ms=round(total_latency_ms, 2),
        similarity_scores=[round(s, 4) for s in scores],
    )
