"""
routes/upload.py - Document upload endpoint.

POST /upload
  - Accepts PDF and TXT files via multipart form upload.
  - Saves the file to the uploads/ directory.
  - Spawns a background thread for document ingestion.
  - Returns immediately with file_id and status.
"""

import os
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException

from config import settings
from models.schemas import UploadResponse
from services.ingestion import start_ingestion, get_status
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Documents"])

# Allowed file extensions (enforced strictly)
ALLOWED_EXTENSIONS = {".pdf", ".txt"}
# Max file size: 50MB
MAX_FILE_SIZE = 50 * 1024 * 1024


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a document for RAG indexing",
    description="Upload a PDF or TXT file. The file will be processed in the background "
                "(chunked, embedded, and added to the FAISS index).",
)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload a document for background ingestion.

    1. Validate file type (PDF or TXT only).
    2. Read and save to uploads/ directory.
    3. Launch background ingestion thread.
    4. Return file_id for status polling.
    """
    # ── Validate file extension ──
    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # ── Read file content ──
    content = await file.read()

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content)} bytes). Max: {MAX_FILE_SIZE} bytes.",
        )

    # ── Generate unique ID and save ──
    file_id = str(uuid.uuid4())
    safe_filename = f"{file_id}_{file.filename}"
    filepath = os.path.join(settings.UPLOAD_DIR, safe_filename)

    with open(filepath, "wb") as f:
        f.write(content)

    logger.info(
        f"File saved: '{safe_filename}' ({len(content)} bytes) → '{filepath}'"
    )

    # ── Start background ingestion ──
    start_ingestion(file_id=file_id, filepath=filepath, filename=file.filename)

    return UploadResponse(
        status="processing",
        file_id=file_id,
        filename=file.filename,
        message="File received successfully. Processing in background.",
    )


@router.get(
    "/status/{file_id}",
    summary="Check document processing status",
    description="Returns the current ingestion status of an uploaded document.",
)
async def check_status(file_id: str) -> dict:
    """Poll the processing status of an uploaded document."""
    status = get_status(file_id)
    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"No document found with file_id: '{file_id}'.",
        )
    return status
