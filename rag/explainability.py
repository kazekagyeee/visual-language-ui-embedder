# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def draw_similarity_boxes(page_image, results, max_items=12):
    img = Image.open(page_image).convert("RGB")
    draw = ImageDraw.Draw(img)

    if not results:
        return img

    scores = np.array([r["score"] for r in results[:max_items]], dtype=np.float32)
    min_s = float(scores.min())
    max_s = float(scores.max())

    def norm_score(s):
        if max_s - min_s < 1e-8:
            return 1.0
        return float((s - min_s) / (max_s - min_s))

    for result in results[:max_items]:
        item = result["item"]
        bbox = item["bbox"]
        score = norm_score(result["score"])

        x0, y0, x1, y1 = bbox

        # Чем выше score, тем толще рамка.
        width = 2 + int(score * 7)

        for i in range(width):
            draw.rectangle(
                [x0 - i, y0 - i, x1 + i, y1 + i],
                outline="red",
            )

        label = f"{item.get('text', '')} {result['score']:.2f}"
        draw.text((x0, max(0, y0 - 16)), label, fill="red")

    return img


def make_interface_crop(page_image, boxes, pad=260):
    img = Image.open(page_image).convert("RGB")
    w, h = img.size

    ux0 = min(b[0] for b in boxes)
    uy0 = min(b[1] for b in boxes)
    ux1 = max(b[2] for b in boxes)
    uy1 = max(b[3] for b in boxes)

    cx0 = max(0, ux0 - pad)
    cy0 = max(0, uy0 - pad)
    cx1 = min(w, ux1 + pad)
    cy1 = min(h, uy1 + pad)

    crop = img.crop((cx0, cy0, cx1, cy1))

    shifted_boxes = []
    for box in boxes:
        x0, y0, x1, y1 = box
        shifted_boxes.append([x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0])

    return crop, shifted_boxes


def draw_boxes_on_crop(crop, boxes):
    img = crop.convert("RGB")
    draw = ImageDraw.Draw(img)

    for box in boxes:
        x0, y0, x1, y1 = box
        for i in range(7):
            draw.rectangle(
                [x0 - i, y0 - i, x1 + i, y1 + i],
                outline="green",
            )

    return img
