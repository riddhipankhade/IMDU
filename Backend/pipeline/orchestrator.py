from pipeline.ingest import load_document
from pipeline.metadata import classify_document
from pipeline.layout import detect_layout
from pipeline.ocr import extract_text_from_layout


def process_document(file_path: str) -> dict:
    """
    Complete document processing pipeline

    Steps:
    1. Load document
    2. Extract metadata
    3. Detect layout
    4. Extract text from layout regions
    5. Return structured result
    """

    # Step 1: Load document
    document = load_document(file_path)

    # Step 2: Metadata
    metadata = classify_document(
        document["pages"]
    )

    # Step 3: Layout Detection
    layout = detect_layout(
        document["pages"]
    )

    # Step 4: OCR
    ocr_result = extract_text_from_layout(
        document["pages"],
        layout
    )

    # Step 5: Final Output
    return {
        "metadata": metadata,
        "pages": ocr_result["pages"]
    }