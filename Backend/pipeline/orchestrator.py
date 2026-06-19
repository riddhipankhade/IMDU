"""
orchestrator.py
---------------
The main pipeline coordinator. Given a file path, runs the correct pipeline
branch and returns a unified result dict.

Two paths:
─────────────────────────────────────────────────────
FAST PATH  (Digital PDF with embedded text)
    PDF → has_text_layer? YES
        → extract_pdf_text()
        → create_chunks_from_text()

OCR PATH  (Scanned PDF or image file)
    file → load_document()  (render pages to PIL Images)
         → classify_document()  (Gemini vision: doc type + language)
         → detect_layout()  (DocLayout-YOLO: find blocks)
         → extract_text_from_layout()  (PaddleOCR on each block)
         → create_layout_chunks()  (split blocks into RAG chunks)
─────────────────────────────────────────────────────

Return shape (both paths):
    {
        "document_type": "digital_pdf" | "ocr_document",
        "metadata":      { document_type, language, pages },
        "text":          str,       ← full concatenated text
        "chunks":        list[dict] ← list of {page, type, text}
    }
"""

from pipeline.ingest import load_document
from pipeline.metadata import classify_document
from pipeline.layout import detect_layout
from pipeline.ocr import extract_text_from_layout
from pipeline.text_extraction import has_text_layer, extract_pdf_text
from pipeline.rag import create_layout_chunks, create_chunks_from_text


def process_document(file_path: str) -> dict:
    """
    Route the file through the correct processing pipeline.
    Raises exceptions on failure — caught by the FastAPI handler.
    """

    # ── Fast Path: Digital PDF ───────────────────────────────────────────────
    if file_path.lower().endswith(".pdf") and has_text_layer(file_path):

        text = extract_pdf_text(file_path)
        chunks = create_chunks_from_text(text)

        return {
            "document_type": "digital_pdf",
            "metadata": {
                "document_type": "digital_pdf",
                "language": "detected_from_text",
                "pages": text.count("\n\n"),  # rough page count from separators
            },
            "text": text,
            "chunks": chunks,
        }

    # ── OCR Path: Scanned PDF or Image ──────────────────────────────────────
    document = load_document(file_path)

    # Gemini classifies doc type + language from first page image
    metadata = classify_document(document["pages"])

    # YOLO detects layout regions (title, paragraph, table, figure, ...)
    layout = detect_layout(document["pages"])

    # PaddleOCR extracts text from each region crop
    ocr_result = extract_text_from_layout(document["pages"], layout)

    # Split into RAG-friendly chunks
    chunks = create_layout_chunks(ocr_result)

    # Concatenate all chunk text for the raw text field
    text = "\n\n".join(chunk["text"] for chunk in chunks)

    return {
        "document_type": "ocr_document",
        "metadata": metadata,
        "text": text,
        "chunks": chunks,
    }
