"""
layout.py
---------
Detects document layout regions (title, paragraph, table, figure, etc.)
using the DocLayout-YOLO model.

Input:  List[PIL.Image]  (one image per page)
Output: {
    "pages": [
        {
            "page": 1,
            "blocks": [
                {"label": "paragraph", "score": 0.92, "bbox": [x1, y1, x2, y2]},
                ...
            ]
        },
        ...
    ]
}

Notes:
- MODEL_PATH must point to the local .pt file shipped with the project.
- Falls back to CPU automatically if CUDA is not available.
- Windows paths used backslash; we use os.path.join for cross-platform safety.
"""

import os
from typing import List
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from doclayout_yolo import YOLOv10


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Resolve model path relative to this file so it works from any working dir
MODEL_PATH = Path(__file__).parent / "models" / "layout" / "doclayout_yolo.pt"

# Load model once at import time (expensive — do not reload per request)
_model = YOLOv10(str(MODEL_PATH))


def detect_layout(pages: List[Image.Image]) -> dict:
    """
    Run layout detection on each page image.
    Returns structured dict with per-page block information.
    """
    results = []

    for i, page in enumerate(pages):
        img_np = np.array(page)

        preds = _model.predict(
            img_np,
            imgsz=1024,
            conf=0.25,
            device=DEVICE,
            verbose=False,  # suppress YOLO console spam
        )

        r = preds[0]
        page_blocks = []

        for box in r.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls_id = box
            page_blocks.append({
                "label": r.names[int(cls_id)],
                "score": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
            })

        results.append({
            "page": i + 1,
            "blocks": page_blocks,
        })

    return {"pages": results}
