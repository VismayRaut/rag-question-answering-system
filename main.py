"""
main.py - FastAPI application entrypoint.

Startup sequence:
  1. Load FAISS index from disk (if exists from previous runs).
  2. Initialize the rate limiter.
  3. Register API routers.
  4. Enable CORS middleware.

Run locally:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routes import upload, query
from vector_store.faiss_store import get_faiss_store
from utils.rate_limiter import init_rate_limiter
from utils.logger import get_logger
from models.schemas import HealthResponse

logger = get_logger(__name__)


# ── Lifespan Events ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    On startup:
      - Load persisted FAISS index (so vectors survive server restarts).
      - Initialize the rate limiter.

    On shutdown:
      - Save FAISS index to disk.
    """
    # ── STARTUP ──
    logger.info("=" * 60)
    logger.info("RAG System starting up...")
    logger.info(f"LLM Provider: {settings.LLM_PROVIDER}")
    logger.info(f"Embedding Model: {settings.EMBEDDING_MODEL}")
    logger.info(f"Chunk Size: {settings.CHUNK_SIZE} tokens, Overlap: {settings.CHUNK_OVERLAP}")
    logger.info("=" * 60)

    # Load existing FAISS index
    store = get_faiss_store()
    loaded = store.load(settings.FAISS_INDEX_PATH, settings.FAISS_META_PATH)
    if loaded:
        logger.info(f"Loaded {store.total_vectors} vectors from disk.")
    else:
        logger.info("No existing index found. Starting with empty index.")

    # Initialize rate limiter
    init_rate_limiter(
        max_requests=settings.RATE_LIMIT_REQUESTS,
        window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
    )
    logger.info(
        f"Rate limiter initialized: {settings.RATE_LIMIT_REQUESTS} "
        f"requests per {settings.RATE_LIMIT_WINDOW_SECONDS}s"
    )

    logger.info("RAG System ready. Accepting requests.")

    yield  # ← Application runs here

    # ── SHUTDOWN ──
    logger.info("Shutting down RAG System...")
    store.save(settings.FAISS_INDEX_PATH, settings.FAISS_META_PATH)
    logger.info("FAISS index saved. Goodbye.")


# ── Application ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Question Answering System",
    description=(
        "A production-ready Retrieval-Augmented Generation API. "
        "Upload PDF/TXT documents and query them using natural language. "
        "Built with FastAPI, SentenceTransformers, and FAISS."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS Middleware ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Route Registration ──
app.include_router(upload.router, prefix="", tags=["Documents"])
app.include_router(query.router, prefix="", tags=["Query"])


# ── Health Endpoint ─────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Health check endpoint. Returns system status and index statistics."""
    store = get_faiss_store()
    return HealthResponse(
        status="healthy",
        documents_indexed=store.total_vectors,
        llm_provider=settings.LLM_PROVIDER,
    )


@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RAG Question Answering System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "upload": "POST /upload",
            "query": "POST /query",
            "status": "GET /status/{file_id}",
            "health": "GET /health",
        },
    }
