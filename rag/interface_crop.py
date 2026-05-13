# -*- coding: utf-8 -*-

from PIL import Image, ImageDraw


def union_bbox(boxes):
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def make_interface_crop(page_image, boxes, pad=260):
    img = Image.open(page_image).convert("RGB")
    w, h = img.size

    ux0, uy0, ux1, uy1 = union_bbox(boxes)

    cx0 = max(0, ux0 - pad)
    cy0 = max(0, uy0 - pad)
    cx1 = min(w, ux1 + pad)
    cy1 = min(h, uy1 + pad)

    crop = img.crop((cx0, cy0, cx1, cy1))

    shifted_boxes = []

    for box in boxes:
        x0, y0, x1, y1 = box
        shifted_boxes.append([
            x0 - cx0,
            y0 - cy0,
            x1 - cx0,
            y1 - cy0,
        ])

    return crop, shifted_boxes


def draw_boxes(img, boxes, color="green"):
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    for box in boxes:
        x0, y0, x1, y1 = box

        for i in range(7):
            draw.rectangle(
                [x0 - i, y0 - i, x1 + i, y1 + i],
                outline=color,
            )

    return img
