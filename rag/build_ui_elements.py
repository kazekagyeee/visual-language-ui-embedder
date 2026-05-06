# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path

import cv2
import easyocr
import numpy as np
from PIL import Image


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text(text):
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def box_from_easyocr(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def crop_box(image_path, bbox, out_path, pad=6):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)

    crop = img.crop((x0, y0, x1, y1))

    if crop.width < 8 or crop.height < 8:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    return True


def is_probably_ui_text(text):
    text_norm = normalize_text(text)

    if len(text_norm) < 3:
        return False

    bad = [
        "раздел",
        "страница",
        "инструкция",
        "реквизит",
        "заполняется",
        "необходимо",
        "показатели занесенные",
        "по кнопке",
    ]

    if any(b in text_norm for b in bad):
        return False

    return True


def is_probably_ui_zone(bbox, page_w, page_h):
    x0, y0, x1, y1 = bbox

    # Отсекаем верхние заголовки PDF
    if y0 < 110:
        return False

    # Отсекаем низ с обычным текстом инструкции.
    # Для инструкций 1С интерфейсы чаще в верхней/средней части страницы.
    if y0 > page_h * 0.78:
        return False

    # Слишком широкие абзацы — не UI.
    if (x1 - x0) > page_w * 0.65:
        return False

    return True


def merge_nearby_words(elements):
    """
    EasyOCR часто возвращает слова отдельно.
    Склеиваем близкие слова в UI-фразы: "Показатели контроля", "Входной контроль".
    """
    by_page = {}

    for el in elements:
        by_page.setdefault(el["page"], []).append(el)

    merged = []

    for page, rows in by_page.items():
        rows = sorted(rows, key=lambda e: (e["bbox"][1], e["bbox"][0]))

        used = set()

        for i, el in enumerate(rows):
            if i in used:
                continue

            group = [el]
            used.add(i)

            x0, y0, x1, y1 = el["bbox"]

            for j, other in enumerate(rows):
                if j in used:
                    continue

                ox0, oy0, ox1, oy1 = other["bbox"]

                same_line = abs(oy0 - y0) < 18
                close_x = 0 <= ox0 - x1 < 45

                if same_line and close_x:
                    group.append(other)
                    used.add(j)
                    x1 = max(x1, ox1)
                    y1 = max(y1, oy1)

            text = " ".join(g["text"] for g in group)
            bbox = [
                min(g["bbox"][0] for g in group),
                min(g["bbox"][1] for g in group),
                max(g["bbox"][2] for g in group),
                max(g["bbox"][3] for g in group),
            ]

            base = dict(group[0])
            base["text"] = clean_text(text)
            base["normalized_text"] = normalize_text(text)
            base["bbox"] = bbox
            merged.append(base)

    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-dir", default="data/pdf_rag")
    parser.add_argument("--langs", nargs="+", default=["ru", "en"])
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    rag_dir = Path(args.rag_dir)
    pages_dir = rag_dir / "pages"
    crops_dir = rag_dir / "ui_element_crops"
    out_path = rag_dir / "ui_elements.jsonl"

    reader = easyocr.Reader(args.langs, gpu=args.gpu)

    page_images = sorted(pages_dir.glob("*.png"))

    all_elements = []
    crop_id = 0

    for page_image in page_images:
        page_num_match = re.search(r"page_(\d+)", page_image.stem)
        if not page_num_match:
            continue

        page = int(page_num_match.group(1))

        img = Image.open(page_image).convert("RGB")
        page_w, page_h = img.size

        results = reader.readtext(str(page_image), detail=1, paragraph=False)

        for points, text, conf in results:
            text = clean_text(text)

            if conf < 0.25:
                continue

            bbox = box_from_easyocr(points)

            if not is_probably_ui_text(text):
                continue

            if not is_probably_ui_zone(bbox, page_w, page_h):
                continue

            crop_path = crops_dir / f"page_{page:04d}_ui_{crop_id:06d}.png"

            ok = crop_box(page_image, bbox, crop_path)
            if not ok:
                continue

            all_elements.append({
                "id": f"page_{page}_ui_{crop_id}",
                "page": page,
                "text": text,
                "normalized_text": normalize_text(text),
                "confidence": float(conf),
                "bbox": bbox,
                "page_image": str(page_image).replace("\\", "/"),
                "crop_image": str(crop_path).replace("\\", "/"),
            })

            crop_id += 1

    all_elements = merge_nearby_words(all_elements)

    # Пересохраняем кропы после merge.
    final_elements = []
    final_crops_dir = rag_dir / "ui_element_crops_merged"
    final_id = 0

    for el in all_elements:
        crop_path = final_crops_dir / f"page_{el['page']:04d}_ui_{final_id:06d}.png"

        ok = crop_box(el["page_image"], el["bbox"], crop_path)
        if not ok:
            continue

        el["id"] = f"page_{el['page']}_ui_{final_id}"
        el["crop_image"] = str(crop_path).replace("\\", "/")
        final_elements.append(el)
        final_id += 1

    with open(out_path, "w", encoding="utf-8") as f:
        for el in final_elements:
            f.write(json.dumps(el, ensure_ascii=False) + "\n")

    print(f"Saved UI elements: {out_path}")
    print(f"Elements: {len(final_elements)}")


if __name__ == "__main__":
    main()
