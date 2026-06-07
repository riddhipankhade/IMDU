import os
os.environ["FLAGS_use_mkldnn"] = "0"

from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
from typing import List


# ---------- Load OCR Model ----------
ocr_model = PaddleOCR(
    use_textline_orientation=True,
    lang="en",
    enable_mkldnn=False
)
# ------------------------------------


def test_ocr(image_path: str):

    """
    Run OCR on a full image and print extracted text
    """

    img = Image.open(image_path).convert("RGB")

    img_np = np.array(img)

    result = ocr_model.predict(img_np)

    print("\n===== OCR TEXT =====\n")

    for line in result[0]["rec_texts"]:
        print(line)


def extract_text_from_layout(
    pages: List[Image.Image],
    layout_results: dict
) -> dict:

    print("Starting OCR extraction...")

    final_output = {
        "pages": []
    }

    for page, layout_page in zip(
        pages,
        layout_results["pages"]
    ):

        print(f"\nProcessing page {layout_page['page']}")

        page_number = layout_page["page"]

        blocks_output = []

        print("Total blocks:", len(layout_page["blocks"]))

        # TEST ONLY FIRST BLOCK
        for block in layout_page["blocks"][:3]:

            print("\n----- NEW BLOCK -----")
            print("Label:", block["label"])

            x1, y1, x2, y2 = block["bbox"]

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            print("BBox:", x1, y1, x2, y2)

            crop = page.crop((x1, y1, x2, y2))

            print("Crop size:", crop.size)

            crop_np = np.array(crop)

            print("Running OCR...")

            result = ocr_model.predict(crop_np)

            print("OCR finished")

            extracted_text = ""

            try:
                extracted_text = " ".join(
                    result[0]["rec_texts"]
                )

                print("Text:", extracted_text[:100])

            except Exception as e:
                print("OCR Parse Error:", e)

            blocks_output.append({
                "label": block["label"],
                "score": block["score"],
                "bbox": block["bbox"],
                "text": extracted_text
            })

        final_output["pages"].append({
            "page": page_number,
            "blocks": blocks_output
        })

    return final_output