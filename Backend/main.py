"""
main.py
-------
FastAPI backend for IMDU (Intelligent Multilingual Document Understanding).

Endpoints:
    POST /upload          Upload a PDF or image → triggers full processing pipeline
    POST /ask             Ask a question against the last processed document
    GET  /document/status Current processing status
    GET  /health          Health check

State is held in memory for simplicity (fine for single-user MVP).
For multi-user or persistent state, replace `_state` with a DB + file store.

Run with:
    uvicorn main:app --reload --port 8000
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pipeline.orchestrator import process_document
from pipeline.rag import ask_document

# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="IMDU API",
    description="Intelligent Multilingual Document Understanding",
    version="1.0.0",
)

# Allow frontend (served on any port) to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock this down in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend from /frontend
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# ── In-memory state ──────────────────────────────────────────────────────────
# Holds the last processed document's chunks for RAG queries.
_state: dict = {
    "status": "idle",       # idle | processing | ready | error
    "document_type": None,
    "metadata": {},
    "text": "",
    "chunks": [],
    "error": None,
    "filename": None,
}

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}

# ── Request/Response models ───────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str

class StatusResponse(BaseModel):
    status: str
    document_type: Optional[str]
    metadata: dict
    filename: Optional[str]
    chunk_count: int
    error: Optional[str]

class AnswerResponse(BaseModel):
    answer: str
    source_chunks: list

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def serve_frontend():
    """Serve the HTML frontend when visiting the root URL."""
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "IMDU API is running. Use /docs for API documentation."}


@app.get("/document/status", response_model=StatusResponse)
def get_status():
    return StatusResponse(
        status=_state["status"],
        document_type=_state["document_type"],
        metadata=_state["metadata"],
        filename=_state["filename"],
        chunk_count=len(_state["chunks"]),
        error=_state["error"],
    )


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or image file.

    The file is saved to a temp directory, processed through the full
    pipeline (text extraction or OCR + layout), and the resulting chunks
    are stored in memory for subsequent /ask queries.
    """

    # Validate file extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Reset state
    _state.update({
        "status": "processing",
        "document_type": None,
        "metadata": {},
        "text": "",
        "chunks": [],
        "error": None,
        "filename": file.filename,
    })

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        shutil.copyfileobj(file.file, tmp)

    try:
        result = process_document(tmp_path)

        _state.update({
            "status": "ready",
            "document_type": result["document_type"],
            "metadata": result["metadata"],
            "text": result["text"],
            "chunks": result["chunks"],
            "error": None,
        })

        return {
            "status": "ready",
            "document_type": result["document_type"],
            "metadata": result["metadata"],
            "chunk_count": len(result["chunks"]),
            "text_preview": result["text"][:500] + ("..." if len(result["text"]) > 500 else ""),
        }

    except Exception as e:
        _state.update({"status": "error", "error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Always clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/ask", response_model=AnswerResponse)
def ask_question(body: QuestionRequest):
    """
    Ask a question against the currently loaded document.

    Requires a document to have been successfully uploaded first.
    """
    if _state["status"] != "ready":
        raise HTTPException(
            status_code=400,
            detail="No document loaded. Please upload a document first."
        )

    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = ask_document(body.question, _state["chunks"])
        return AnswerResponse(
            answer=result["answer"],
            source_chunks=result["source_chunks"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
