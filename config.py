"""
config.py - Application configuration using Pydantic BaseSettings.
All values can be overridden via environment variables or a .env file.
"""

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────────────────────
    # LLM_PROVIDER controls which backend is used for answer generation.
    # Options: "mock" (no API key needed), "openai" (requires OPENAI_API_KEY)
    LLM_PROVIDER: str = "mock"
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-3.5-turbo"

    # ── Embedding ─────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # ── Chunking Strategy ─────────────────────────────────────────────────────
    # Chunk size = 400 tokens:
    #   Chosen to balance context richness vs. embedding noise.
    #   Too small → answers lack context; too large → embeddings become diluted.
    # Overlap = 80 tokens (20% of chunk size):
    #   Prevents information loss at chunk boundaries. A sentence split
    #   across two chunks is still captured in the overlapping region.
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 80

    # ── Retrieval ─────────────────────────────────────────────────────────────
    TOP_K: int = 5
    # Minimum cosine similarity score to consider a chunk relevant.
    # Chunks scoring below this threshold are flagged as irrelevant.
    MIN_SIMILARITY_THRESHOLD: float = 0.3

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    RATE_LIMIT_REQUESTS: int = 5         # max requests
    RATE_LIMIT_WINDOW_SECONDS: int = 60  # per time window

    # ── Paths ─────────────────────────────────────────────────────────────────
    UPLOAD_DIR: str = "uploads"
    FAISS_INDEX_PATH: str = "vector_store/faiss.index"
    FAISS_META_PATH: str = "vector_store/faiss_meta.json"

    class Config:
        env_file = ".env"
        extra = "ignore"


# Singleton settings object imported across the app
settings = Settings()

# Ensure upload directory exists at import time
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH), exist_ok=True)
