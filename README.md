# RAG-Based Question Answering System

A production-ready Retrieval-Augmented Generation (RAG) system built with FastAPI, SentenceTransformers, and FAISS. This system allows users to upload PDF and TXT documents, processes them in the background, and provides a query endpoint to answer questions based on the embedded documents.

## Architecture Description

**System Components & Workflow:**
1. **API Layer (FastAPI):** Exposes `/upload` and `/query` endpoints, handles request validation via Pydantic, and enforces per-IP rate limiting.
2. **Ingestion Pipeline (Background Thread):**
   - **Parser:** Extracts text from uploaded PDFs (`PyMuPDF`) or TXTs.
   - **Chunker:** Splits text into token-aware segments using a sliding window.
   - **Embedder:** Converts text chunks into 384-dimensional dense vectors using `SentenceTransformers (all-MiniLM-L6-v2)`.
   - **Vector Store:** Saves vectors into a local `FAISS (IndexFlatIP)` index, storing metadata separately.
3. **Retrieval Service:** Finds the top-K chunks using cosine similarity. Results are filtered against a minimum similarity threshold to drop irrelevant chunks.
4. **LLM Generation:** Uses retrieved context to formulate answers via a modular LLM interface (defaults to a mock provider for easy local testing, with an OpenAI option available).

---

## Setup Instructions

### 1. Requirements

- Python 3.10+
- Virtual Environment recommended

### 2. Installation

Clone the repository and install dependencies:

```bash
git clone <your-repo-link>
cd rag_system
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Configuration

The application uses a `.env` file for configuration. A `.env.example` is provided:

```bash
cp .env.example .env
```

By default, the system uses a `"mock"` LLM provider so you can test the ingestion and retrieval pipeline without API keys. To use a real LLM, update the `.env` file:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your-api-key-here
```

### 4. Running the Server

Start the FastAPI application via uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API docs will be available at `http://127.0.0.1:8000/docs`.

---

## API Usage Examples

### 1. Upload a Document
This endpoint receives the file and triggers background ingestion. You can upload `.txt` or `.pdf` files.

```bash
curl -X POST "http://127.0.0.1:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.pdf"
```

**Response:**
```json
{
  "status": "processing",
  "file_id": "8bb38d38-2339-4ad3-aee3-bc1a67dcff06",
  "filename": "sample.pdf",
  "message": "File received successfully. Processing in background."
}
```

### 2. Check Processing Status
Use the `file_id` to poll the processing status.

```bash
curl -X GET "http://127.0.0.1:8000/status/8bb38d38-2339-4ad3-aee3-bc1a67dcff06"
```

**Response:**
```json
{
  "file_id": "8bb38d38-2339-4ad3-aee3-bc1a67dcff06",
  "filename": "sample.pdf",
  "status": "done",
  "chunk_count": 14,
  "error": null
}
```

### 3. Query the System
Ask a question. The system will embed your query, retrieve relevant chunks, and pass them to the LLM.

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the main topic of the document?", "top_k":5}'
```

**Response:**
```json
{
  "question": "What is the main topic of the document?",
  "answer": "Based on the provided context, the main topic is...",
  "retrieved_chunks": [
    {
      "chunk_id": "sample.pdf_chunk_0",
      "source": "sample.pdf",
      "text": "The main topic discussed in this paper is...",
      "similarity_score": 0.8123
    }
  ],
  "latency_ms": 145.2,
  "similarity_scores": [0.8123, 0.7654, 0.6543]
}
```

---

## Mandatory Explanations

### Chunking Strategy Explained
- **Why 400 tokens per chunk?** The `all-MiniLM-L6-v2` embedding model has a maximum sequence length, but more importantly, 400 tokens (effectively 1-2 paragraphs) provides enough context for the vector to capture a coherent idea without diluting the semantic meaning. Too large, and specific topics get drowned out. Too small, and there isn't enough context to form a meaningful answer.
- **Why 80 tokens of overlap (20%)?** Overlap prevents critical information from being split apart at chunk boundaries. If a sentence spans across a division, a 20% sliding window guarantees it is fully encapsulated in at least one chunk. We use simple whitespace splits as a quick, deterministic approximation of tokens.

### Retrieval Failure Case Example
- **Scenario:** A user queries: "What is the recipe for chocolate cake?" but the uploaded documents only contain financial reports.
- **Observation:** The vector search returns top-K results, but all have very low cosine similarity scores (e.g., `< 0.15`).
- **Handling:** Instead of feeding this wildly irrelevant context to the LLM (which might cause it to hallucinate), our system enforces a `MIN_SIMILARITY_THRESHOLD = 0.3`. Because the scores are below this threshold, the chunks are discarded. The system immediately skips the LLM call and returns a deterministic, safe fallback response: *"I don't know based on the provided documents."* The failure is then logged with a `WARNING` for debugging.

### Tracked Metric Explanation
- **Metric Tracked:** **Query Latency (ms)** track overall time vs retrieval time.
- **Where & Why:** In `routes/query.py`, we measure and log the pipeline execution time (`retrieval_latency_ms` vs `total_latency_ms`). Tracking this is crucial for RAG systems because embedding latency, vector search speed, and external API requests (like OpenAI) vary heavily. By logging the latency of the retrieval phase separately from the total request duration, we can instantly identify bottlenecks — whether FAISS brute-force search is slowing down due to a massive index size or the OpenAI API is experiencing delays.
