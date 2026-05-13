# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path

import easyocr
from PIL import Image

from rag.ocr_cleaning import normalize_ocr_text
from rag.ui_type_detector import detect_ui_type
from rag.ui_candidate_filter import is_likely_ui_candidate


def clean_text(text):
    text = re.sub(r"\s+", " ", str(text))
    return text.strip()


def box_from_easyocr(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def box_height(box):
    return box[3] - box[1]


def box_width(box):
    return box[2] - box[0]


def center_y(box):
    return (box[1] + box[3]) / 2


def same_line(a, b):
    h = max(box_height(a), box_height(b), 1)
    return abs(center_y(a) - center_y(b)) <= h * 0.65


def horizontal_gap(a, b):
    return b[0] - a[2]


def should_merge_words(a, b):
    box_a = a["bbox"]
    box_b = b["bbox"]

    if not same_line(box_a, box_b):
        return False

    gap = horizontal_gap(box_a, box_b)
    h = max(box_height(box_a), box_height(box_b), 1)

    # Разрешаем небольшое наложение bbox, потому что OCR часто режет слова криво.
    if gap < -h * 1.5:
        return False

    if gap > h * 3.5:
        return False

    text_a = normalize_ocr_text(a["text"])
    text_b = normalize_ocr_text(b["text"])

    if not text_a or not text_b:
        return False

    merged_words = (text_a + " " + text_b).split()

    if len(merged_words) > 6:
        return False

    # Специально разрешаем важные UI-фразы.
    important_pairs = {
        ("показатели", "контроля"),
        ("виды", "контроля"),
        ("группы", "прочности"),
        ("входной", "контроль"),
        ("заявки", "на"),
        ("на", "контроль"),
        ("акты", "входного"),
        ("входного", "контроля"),
        ("выполнения", "входного"),
    }

    if (text_a, text_b) in important_pairs:
        return True

    return True


def merge_two(a, b):
    text = clean_text(a["text"] + " " + b["text"])

    ax0, ay0, ax1, ay1 = a["bbox"]
    bx0, by0, bx1, by1 = b["bbox"]

    merged = dict(a)
    merged["text"] = text
    merged["normalized_text"] = normalize_ocr_text(text)
    merged["bbox"] = [
        min(ax0, bx0),
        min(ay0, by0),
        max(ax1, bx1),
        max(ay1, by1),
    ]
    merged["confidence"] = min(float(a.get("confidence", 1.0)), float(b.get("confidence", 1.0)))
    merged["merged"] = True

    return merged


def merge_nearby_words(elements):
    by_page = {}

    for el in elements:
        by_page.setdefault(el["page"], []).append(el)

    merged_all = []

    for page, page_elements in by_page.items():
        page_elements = sorted(page_elements, key=lambda e: (e["bbox"][1], e["bbox"][0]))
        used = set()

        for i, item in enumerate(page_elements):
            if i in used:
                continue

            current = dict(item)
            used.add(i)

            changed = True
            while changed:
                changed = False

                for j, other in enumerate(page_elements):
                    if j in used:
                        continue

                    if should_merge_words(current, other):
                        current = merge_two(current, other)
                        used.add(j)
                        changed = True

            merged_all.append(current)

    return merged_all


def crop_box(image_path, bbox, out_path, pad=6):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    x0, y0, x1, y1 = bbox
    x0 = max(0, int(x0) - pad)
    y0 = max(0, int(y0) - pad)
    x1 = min(w, int(x1) + pad)
    y1 = min(h, int(y1) + pad)

    crop = img.crop((x0, y0, x1, y1))

    if crop.width < 8 or crop.height < 8:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    return True


def is_probably_ui_text(text):
    norm = normalize_ocr_text(text)

    if len(norm) < 2:
        return False

    bad_parts = [
        "страница",
        "инструкция",
        "рис",
        "рисунок",
        "таблица",
        "листинг",
        "глава",
        "раздел",
    ]

    if any(bad == norm or norm.startswith(bad + " ") for bad in bad_parts):
        return False

    return True


def is_probably_ui_zone(bbox, page_w, page_h):
    x0, y0, x1, y1 = bbox
    bw = x1 - x0
    bh = y1 - y0

    if bw < 8 or bh < 8:
        return False

    # Самый верх PDF — чаще номер страницы / заголовок.
    if y0 < page_h * 0.10:
        return False

    # Самый низ — чаще текст инструкции.
    if y0 > page_h * 0.92:
        return False

    # Очень длинные строки — чаще абзацы.
    if bw > page_w * 0.70:
        return False

    return True


def save_final_crops(elements, out_dir):
    final = []
    seen = set()

    for idx, el in enumerate(elements):
        key = (
            el["page"],
            el["normalized_text"],
            tuple(el["bbox"]),
        )

        if key in seen:
            continue

        seen.add(key)

        page = int(el["page"])
        crop_path = out_dir / f"page_{page:04d}_ui_{idx:06d}.png"

        if not crop_box(el["page_image"], el["bbox"], crop_path, pad=7):
            continue

        item = dict(el)
        item["id"] = f"page_{page}_ui_{idx}"
        item["crop_image"] = str(crop_path).replace("\\", "/")
        item["ui_type"] = detect_ui_type(item["text"], item["bbox"])
        final.append(item)

    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-dir", default="data/pdf_rag")
    parser.add_argument("--langs", nargs="+", default=["ru", "en"])
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--max-pages", type=int, default=120)
    parser.add_argument("--min-conf", type=float, default=0.25)
    args = parser.parse_args()

    rag_dir = Path(args.rag_dir)
    pages_dir = rag_dir / "pages"
    raw_crops_dir = rag_dir / "ui_element_crops"
    merged_crops_dir = rag_dir / "ui_element_crops_merged"
    out_path = rag_dir / "ui_elements.jsonl"

    if not pages_dir.exists():
        raise FileNotFoundError(f"Pages directory not found: {pages_dir}")

    reader = easyocr.Reader(args.langs, gpu=args.gpu)

    page_images = sorted(pages_dir.glob("*.png"))[: args.max_pages]

    raw_elements = []
    crop_id = 0

    for page_image in page_images:
        page_num_match = re.search(r"page_(\d+)", page_image.stem)

        if not page_num_match:
            continue

        page = int(page_num_match.group(1))

        img = Image.open(page_image).convert("RGB")
        page_w, page_h = img.size

        print(f"OCR page {page}: {page_image}")

        results = reader.readtext(
            str(page_image),
            detail=1,
            paragraph=False,
        )

        for points, text, conf in results:
            text = clean_text(text)

            if conf < args.min_conf:
                continue

            bbox = box_from_easyocr(points)

            if not is_probably_ui_text(text):
                continue

            if not is_probably_ui_zone(bbox, page_w, page_h):
                continue

            if not is_likely_ui_candidate(text, bbox, page_w, page_h):
                continue

            ui_type = detect_ui_type(text, bbox, page_width=page_w)

            crop_path = raw_crops_dir / f"page_{page:04d}_ui_{crop_id:06d}.png"

            if not crop_box(page_image, bbox, crop_path, pad=5):
                continue

            raw_elements.append({
                "id": f"page_{page}_ui_raw_{crop_id}",
                "page": page,
                "text": text,
                "normalized_text": normalize_ocr_text(text),
                "ui_type": ui_type,
                "confidence": float(conf),
                "bbox": bbox,
                "page_image": str(page_image).replace("\\", "/"),
                "crop_image": str(crop_path).replace("\\", "/"),
            })

            crop_id += 1

    merged_elements = merge_nearby_words(raw_elements)
    final_elements = save_final_crops(merged_elements, merged_crops_dir)

    with open(out_path, "w", encoding="utf-8") as f:
        for el in final_elements:
            f.write(json.dumps(el, ensure_ascii=False) + "\n")

    print("=" * 80)
    print(f"Saved UI elements: {out_path}")
    print(f"Raw elements: {len(raw_elements)}")
    print(f"Merged elements: {len(final_elements)}")


if __name__ == "__main__":
    main()
