# RAG System — Technical Explanation Document

**Project:** RAG-Based Question Answering System  
**Author:** Vismay Raut  
**Institution:** MIT ADT University  

---

## 1. Why We Chose a Chunk Size of 400 Tokens

### Decision
Each document is split into chunks of **400 whitespace tokens** with an **overlap of 80 tokens (20%)**.

### Reasoning

When building a RAG system, the chunk size is one of the most critical design decisions. Here is why we chose 400:

**Too small (< 100 tokens):**
- A chunk of 50–100 words provides too little context.
- The embedding vector ends up representing a single sentence or partial thought.
- Retrieval becomes unreliable because short chunks often lack enough meaning to match a broad question.
- Example: A chunk containing only *"He graduated in 2026."* cannot answer *"What is the candidate's educational background?"*

**Too large (> 800 tokens):**
- A chunk spanning multiple paragraphs blends several unrelated topics into one embedding.
- The embedding becomes diluted — it weakly represents everything and nothing strongly.
- Retrieval precision drops because a chunk about both skills AND experience scores poorly against a specific skills-related query.
- Also, the `all-MiniLM-L6-v2` model has a maximum sequence length of 256 word-pieces; anything beyond that is truncated internally by the model, so large chunks waste tokens.

**400 tokens (our choice):**
- Corresponds to approximately **1–2 focused paragraphs** of real text.
- Provides enough context for the embedding to capture a complete, coherent idea.
- Stays within the effective range of `all-MiniLM-L6-v2` without truncation risk.
- Balances **retrieval precision** (not too broad) with **context richness** (not too narrow).

### Why 80 Tokens of Overlap?
When we split text at a fixed boundary, important sentences can be cut in half — the beginning lands in chunk N and the end lands in chunk N+1. With **80 tokens of overlap (20%)**, the sliding window ensures that any sentence spanning a boundary is fully captured in at least one chunk. This prevents information loss at chunk edges without duplicating too much content.

---

## 2. One Retrieval Failure Case Observed

### Scenario
**Query asked:**  
*"What is the recipe for chocolate cake?"*

**Documents uploaded:**  
Only a technical resume PDF containing education, skills, and work experience.

### What Happened
The FAISS similarity search returned 5 chunks, but all of them had very low cosine similarity scores:

```
scores = [0.12, 0.09, 0.08, 0.07, 0.05]
```

None of the chunks scored above the minimum threshold of **0.3**.

### System Log (Actual Warning Generated)
```
WARNING | services.retrieval | RETRIEVAL FAILURE: All 5 chunks scored below 
threshold (0.3) for query: 'What is the recipe for chocolate cake?'. 
Scores were: [0.12, 0.09, 0.08, 0.07, 0.05]. 
Returning fallback 'I don't know' response.
```

### Root Cause
The query embedding (representing food/cooking) had very low cosine similarity with all resume chunk embeddings (representing career/technology). The question was semantically unrelated to any content in the indexed documents.

### How Our System Handled It
Instead of passing these irrelevant chunks to the LLM — which might cause it to hallucinate a fake answer — our system:
1. Detected that ALL scores were below the `MIN_SIMILARITY_THRESHOLD = 0.3`
2. Skipped the LLM call entirely
3. Returned a deterministic safe response:

```json
{
  "answer": "I don't know based on the provided documents. The retrieved 
             content did not contain information relevant to your question.",
  "retrieved_chunks": [],
  "similarity_scores": []
}
```

### Lesson Learned
A static threshold of 0.3 works well for clearly on-topic vs. clearly off-topic queries. However, for edge cases where the query is partially related, the threshold may filter out marginally useful chunks. A future improvement would be a **relative threshold** (e.g., drop chunks that score less than 50% of the top score) to handle ambiguous retrievals more gracefully.

---

## 3. One Metric Tracked — Query Latency

### What We Tracked
We tracked **Query Latency in milliseconds**, broken into two phases:
- `retrieval_latency_ms` — time taken for embedding the query + FAISS vector search
- `total_latency_ms` — end-to-end time including LLM generation

### Real Observed Values (from actual system run)
```
question       : "What are the technical skills of this person?"
retrieval_latency_ms : 15.4 ms
total_latency_ms     : 16.7 ms
chunks_returned      : 3
similarity_scores    : [0.3808, 0.3372, 0.3372]
```

### Where It Is Tracked
In `routes/query.py` — we wrap the entire pipeline with `time.time()` and log the result:

```python
# Logged on every query
logger.info(
    f"QUERY METRICS | question='{body.question[:80]}' | "
    f"retrieval_latency={retrieval_latency_ms:.1f}ms | "
    f"total_latency={total_latency_ms:.1f}ms | "
    f"chunks_returned={len(retrieved_chunks)} | "
    f"scores={[round(s, 4) for s in scores]}"
)
```

The metric is also returned to the API client in the response body:
```json
{
  "latency_ms": 16.71,
  "similarity_scores": [0.3808, 0.3372, 0.3372]
}
```

### Why This Metric Matters
| Reason | Explanation |
|--------|-------------|
| **Bottleneck detection** | If `retrieval_latency` spikes, it means FAISS index has grown too large for brute-force search → time to switch to approximate indexing (HNSW) |
| **LLM overhead** | `total_latency - retrieval_latency` = LLM generation time. If this dominates, consider a faster model or caching. |
| **SLA monitoring** | For a production API, queries should respond in < 500ms. Our system achieves ~17ms for retrieval alone — well within limits. |
| **Regression testing** | If latency increases after adding new documents, it signals that index optimization is needed. |

### Similarity Scores as a Quality Metric
We also return raw similarity scores to the client. This provides **retrieval transparency** — the user (or a monitoring system) can inspect whether the retrieved chunks were highly relevant (scores close to 1.0) or marginally relevant (scores near 0.3). This is critical for building trust in RAG-based systems.

---

*This document accompanies the RAG System codebase submitted at:*  
*https://github.com/VismayRaut/rag-question-answering-system*
