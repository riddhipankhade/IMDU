"""
ocr.py
------
Runs PaddleOCR on cropped layout blocks to extract text from each region.

Why block-level OCR?
    Running OCR on full pages works but loses structure.
    By cropping to detected layout blocks first, we preserve:
    - Which block is a title vs paragraph vs table
    - Reading order (top-to-bottom, left-to-right within the page)

Input:
    pages          → List[PIL.Image]  (full page images)
    layout_results → output from layout.detect_layout()

Output:
    {
        "pages": [
            {
                "page": 1,
                "blocks": [
                    {
                        "label": "paragraph",
                        "score": 0.92,
                        "bbox": [x1, y1, x2, y2],
                        "text": "extracted text..."
                    },
                    ...
                ]
            }
        ]
    }
"""

import os
# Disable MKL-DNN to avoid crashes on systems without AVX512 support
os.environ["FLAGS_use_mkldnn"] = "0"

import numpy as np
from typing import List
from PIL import Image
from paddleocr import PaddleOCR


# Load OCR model once — loading is slow (~5s); reloading per request would kill perf
_ocr_model = PaddleOCR(
    use_textline_orientation=True,
    lang="en",
    enable_mkldnn=False,  # suppress PaddleOCR verbose output
)


def extract_text_from_layout(
    pages: List[Image.Image],
    layout_results: dict,
) -> dict:
    """
    For each layout block on each page:
    1. Crop the block region from the full page image
    2. Run PaddleOCR on the crop
    3. Join all recognized text lines into a single string
    """
    output_pages = []

    for page_img, layout_page in zip(pages, layout_results["pages"]):
        page_num = layout_page["page"]
        blocks_output = []

        for block in layout_page["blocks"]:
            x1, y1, x2, y2 = map(int, block["bbox"])

            # Guard against degenerate bboxes (can happen with low-conf detections)
            if x2 <= x1 or y2 <= y1:
                continue

            # Crop the region
            crop = page_img.crop((x1, y1, x2, y2))
            crop_np = np.array(crop)

            # Run OCR
            extracted_text = ""
            try:
                result = _ocr_model.predict(crop_np)
                if result and result[0] and "rec_texts" in result[0]:
                    extracted_text = " ".join(result[0]["rec_texts"])
            except Exception as e:
                # Don't crash the whole pipeline on a single bad block
                print(f"[OCR] Warning: block OCR failed on page {page_num}: {e}")

            blocks_output.append({
                "label": block["label"],
                "score": block["score"],
                "bbox": block["bbox"],
                "text": extracted_text.strip(),
            })

        output_pages.append({
            "page": page_num,
            "blocks": blocks_output,
        })

    return {"pages": output_pages}
