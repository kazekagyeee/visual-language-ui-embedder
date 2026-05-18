# -*- coding: utf-8 -*-

from pathlib import Path
from PIL import Image, ImageDraw


def _pad_box(box, pad, w, h):
    x0, y0, x1, y1 = [int(v) for v in box]
    return [
        max(0, x0 - pad),
        max(0, y0 - pad),
        min(w, x1 + pad),
        min(h, y1 + pad),
    ]


def _union_box(boxes):
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def make_ui_focus_image(page_image_path, matched_elements, out_path="temp/ui_focus.png", pad=35):
    """
    Показывает НЕ маленький кроп элемента, а всю найденную картинку интерфейса из PDF.
    Если у элемента есть zone_bbox — берем всю UI-зону.
    Если zone_bbox нет — берем область вокруг найденных элементов.
    """
    page_image_path = Path(page_image_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(page_image_path).convert("RGB")
    w, h = img.size

    zones = [
        el.get("zone_bbox")
        for el in matched_elements
        if el.get("zone_bbox")
    ]

    if zones:
        focus_box = _union_box(zones)
        focus_box = _pad_box(focus_box, pad, w, h)
    else:
        boxes = [el["bbox"] for el in matched_elements if el.get("bbox")]
        focus_box = _pad_box(_union_box(boxes), 160, w, h)

    crop = img.crop(tuple(focus_box))
    draw = ImageDraw.Draw(crop)

    colors = [
        (220, 30, 30),
        (20, 130, 40),
        (30, 90, 220),
        (230, 150, 0),
    ]

    for idx, el in enumerate(matched_elements):
        box = el.get("bbox")

        if not box:
            continue

        x0, y0, x1, y1 = [int(v) for v in box]

        local_box = [
            x0 - focus_box[0],
            y0 - focus_box[1],
            x1 - focus_box[0],
            y1 - focus_box[1],
        ]

        color = colors[idx % len(colors)]

        for k in range(4):
            draw.rectangle(
                [
                    local_box[0] - k,
                    local_box[1] - k,
                    local_box[2] + k,
                    local_box[3] + k,
                ],
                outline=color,
            )

    crop.save(out_path)
    return str(out_path)


def make_full_debug_image(page_image_path, matched_elements, out_path="temp/ui_debug_full.png"):
    page_image_path = Path(page_image_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(page_image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for el in matched_elements:
        box = el.get("bbox")

        if not box:
            continue

        x0, y0, x1, y1 = [int(v) for v in box]

        for k in range(5):
            draw.rectangle(
                [x0 - k, y0 - k, x1 + k, y1 + k],
                outline=(220, 30, 30),
            )

    img.save(out_path)
    return str(out_path)
