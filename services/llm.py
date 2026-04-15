"""
services/llm.py - Modular LLM answer generation.

This module is intentionally modular: you can plug in different LLM backends
by setting `LLM_PROVIDER` in your .env file.

Supported providers:
  - "mock"   → Returns a structured mock answer (no API key needed, ideal for testing)
  - "openai" → Calls OpenAI Chat Completions API (requires OPENAI_API_KEY)

To add a new provider (e.g., local Ollama, HuggingFace, Anthropic):
  1. Add a new function: _generate_<provider>(context, question) -> str
  2. Register it in the PROVIDERS dict below.
"""

from typing import List, Dict, Any

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Prompt Template ──────────────────────────────────────────────────────────
# This structured prompt enforces grounded answers.
# The LLM is instructed to ONLY use the provided context, preventing hallucination.
# If the answer isn't in the context, it must say "I don't know."

SYSTEM_PROMPT = """You are a helpful document assistant. Your job is to answer questions based ONLY on the provided context.

Rules:
1. Answer ONLY using information from the provided context.
2. If the answer is not in the context, say "I don't know based on the provided documents."
3. Be concise and precise.
4. If the context is partially relevant, acknowledge what you can answer and what you cannot.
5. Always cite which source document the information came from when possible."""

USER_PROMPT_TEMPLATE = """Context (retrieved document chunks):
---
{context}
---

Question: {question}

Answer:"""


def _build_context(chunks: List[Dict[str, Any]], scores: List[float]) -> str:
    """
    Build a formatted context string from retrieved chunks.

    Each chunk is labeled with its source file and similarity score
    so the LLM (and human reviewers) can gauge relevance.
    """
    context_parts = []
    for i, (chunk, score) in enumerate(zip(chunks, scores), 1):
        context_parts.append(
            f"[Chunk {i} | Source: {chunk['source']} | Relevance: {score:.4f}]\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(context_parts)


# ── Provider: Mock ──────────────────────────────────────────────────────────

def _generate_mock(context: str, question: str) -> str:
    """
    Mock LLM provider for testing without an API key.

    Returns a structured response that includes the context summary,
    making it useful for verifying the retrieval pipeline works correctly.
    """
    # Count chunks from context
    chunk_count = context.count("[Chunk ")

    return (
        f"[MOCK LLM RESPONSE]\n\n"
        f"Question: {question}\n\n"
        f"Based on {chunk_count} retrieved document chunk(s), here is a summary "
        f"of the relevant context:\n\n"
        f"{context[:500]}{'...' if len(context) > 500 else ''}\n\n"
        f"Note: This is a mock response. Set LLM_PROVIDER=openai in your .env "
        f"file and provide an OPENAI_API_KEY for real LLM-generated answers."
    )


# ── Provider: OpenAI ────────────────────────────────────────────────────────

def _generate_openai(context: str, question: str) -> str:
    """
    OpenAI Chat Completions provider.

    Requires: OPENAI_API_KEY in environment / .env file.
    Uses: gpt-3.5-turbo by default (configurable via OPENAI_MODEL).
    """
    try:
        import openai
    except ImportError:
        raise RuntimeError(
            "openai package not installed. Run: pip install openai"
        )

    if not settings.OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Add it to your .env file or environment."
        )

    client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
    user_prompt = USER_PROMPT_TEMPLATE.format(context=context, question=question)

    try:
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,   # low temperature for factual answers
            max_tokens=1024,
        )
        answer = response.choices[0].message.content.strip()
        logger.info(f"OpenAI response generated: {len(answer)} characters.")
        return answer

    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise RuntimeError(f"OpenAI API error: {e}") from e


# ── Provider Registry ───────────────────────────────────────────────────────

PROVIDERS = {
    "mock": _generate_mock,
    "openai": _generate_openai,
}


# ── Public Interface ────────────────────────────────────────────────────────

def generate_answer(
    chunks: List[Dict[str, Any]],
    scores: List[float],
    question: str,
    provider: str | None = None,
) -> str:
    """
    Generate an answer using the configured LLM provider.

    Args:
        chunks:   List of relevant chunk metadata dicts from retrieval.
        scores:   Parallel list of similarity scores.
        question: The user's original question.
        provider: LLM provider override. Defaults to settings.LLM_PROVIDER.

    Returns:
        The generated answer string.

    If no relevant chunks were found (empty list), returns a fallback
    "I don't know" response immediately without calling the LLM.
    """
    if provider is None:
        provider = settings.LLM_PROVIDER

    # ── Fallback: no relevant context ──
    if not chunks:
        fallback = (
            "I don't know based on the provided documents. "
            "The retrieved content did not contain information relevant to your question."
        )
        logger.info(f"Fallback response (no relevant chunks) for: '{question[:80]}'")
        return fallback

    # ── Build context and call provider ──
    context = _build_context(chunks, scores)

    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. "
            f"Available: {list(PROVIDERS.keys())}"
        )

    logger.info(f"Generating answer with provider='{provider}'...")
    return PROVIDERS[provider](context, question)
