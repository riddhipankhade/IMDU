from pipeline.ingest import load_document
from pipeline.metadata import classify_document
from pipeline.layout import detect_layout
from pipeline.ocr import extract_text_from_layout

from pipeline.text_extraction import (
    has_text_layer,
    extract_pdf_text
)

from pipeline.rag import (
    chunk_text,
    document_to_text
)


def process_document(file_path: str) -> dict:
    """
    Complete document processing pipeline.

    Fast Path:
        Digital PDF
        -> Direct text extraction

    OCR Path:
        Scanned PDF / Images
        -> Layout Detection
        -> OCR

    Returns:
        metadata
        text
        chunks
    """

    # -------------------------------
    # Fast Path (Digital PDF)
    # -------------------------------
    if file_path.lower().endswith(".pdf") and has_text_layer(file_path):

        text = extract_pdf_text(file_path)

        chunks = chunk_text(text)

        return {
            "document_type": "digital_pdf",
            "metadata": {
                "pages": None
            },
            "text": text,
            "chunks": chunks
        }

    # -------------------------------
    # OCR Path
    # -------------------------------

    # Load document
    document = load_document(file_path)

    # Metadata
    metadata = classify_document(
        document["pages"]
    )

    # Layout Detection
    layout = detect_layout(
        document["pages"]
    )

    # OCR
    ocr_result = extract_text_from_layout(
        document["pages"],
        layout
    )

    # Convert OCR JSON -> plain text
    text = document_to_text(ocr_result)

    # Chunk text
    chunks = chunk_text(text)

    return {
        "document_type": "ocr_document",
        "metadata": metadata,
        "text": text,
        "chunks": chunks,
        "pages": ocr_result["pages"]
    }