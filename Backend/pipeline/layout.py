from typing import List
from PIL import Image
import numpy as np
from doclayout_yolo import YOLOv10
import torch
from huggingface_hub import PyTorchModelHubMixin

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLOv10(r"pipeline\models\layout\doclayout_yolo.pt")


def detect_layout(pages: List[Image.Image]) -> dict:

    results = []

    for i, page in enumerate(pages):

        img = np.array(page)

        preds = model.predict(
            img,
            imgsz=1024,
            conf=0.25,
            device=DEVICE
        )

        r = preds[0]

        page_blocks = []

        for box in r.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls_id = box

            page_blocks.append({
                "label": r.names[int(cls_id)],
                "score": round(conf, 3),
                "bbox": [x1, y1, x2, y2]
            })

        results.append({
            "page": i + 1,
            "blocks": page_blocks
        })

    return {"pages": results}