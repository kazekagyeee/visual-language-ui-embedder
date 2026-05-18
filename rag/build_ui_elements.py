# -*- coding: utf-8 -*-

import argparse
import json
import re
import gc
from pathlib import Path

import cv2
import easyocr
import numpy as np
from PIL import Image

from rag.ocr_cleaning import normalize_ocr_text
from rag.ui_type_detector import detect_ui_type
from rag.ui_candidate_filter import is_likely_ui_candidate


def clean_text(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def box_from_easyocr(points, scale_back=1.0, offset=(0, 0)):
    ox, oy = offset
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [
        int(min(xs) * scale_back + ox),
        int(min(ys) * scale_back + oy),
        int(max(xs) * scale_back + ox),
        int(max(ys) * scale_back + oy),
    ]


def normalize_box(box):
    return [int(x) for x in box]


def box_area(box):
    x0, y0, x1, y1 = box
    return max(0, x1 - x0) * max(0, y1 - y0)


def box_inside(inner, outer, tolerance=8):
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    return (
        ix0 >= ox0 - tolerance
        and iy0 >= oy0 - tolerance
        and ix1 <= ox1 + tolerance
        and iy1 <= oy1 + tolerance
    )


def boxes_intersect(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def union_box(boxes):
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def detect_ui_zones(page_image_path, min_area_ratio=0.004):
    """
    Ищет именно встроенные скриншоты интерфейса, а не весь текстовый блок страницы.
    Логика:
    1) ищем прямоугольники/рамки интерфейса;
    2) ищем цветные UI-области;
    3) НЕ склеиваем зоны через большие вертикальные промежутки;
    4) НЕ берем обычные текстовые абзацы.
    """
    img = cv2.imread(str(page_image_path))

    if img is None:
        return []

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    zones = []

    # =====================================================
    # 1. Рамки интерфейсов: окна, формы, таблицы, панели
    # =====================================================
    edges = cv2.Canny(gray, 50, 160)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 35))

    horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_h, iterations=1)
    vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_v, iterations=1)

    lines = cv2.bitwise_or(horizontal, vertical)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    closed = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)

        area = bw * bh

        if area < w * h * min_area_ratio:
            continue

        if bw < 120 or bh < 60:
            continue

        if y < h * 0.05 or y > h * 0.95:
            continue

        aspect = bw / max(bh, 1)

        if aspect > 12 and bh < 120:
            continue

        crop = img[y:y + bh, x:x + bw]
        crop_gray = gray[y:y + bh, x:x + bw]

        edge_density = float(np.mean(cv2.Canny(crop_gray, 50, 160) > 0))
        color_std = float(np.std(crop.reshape(-1, 3), axis=0).mean())

        if edge_density < 0.018 and color_std < 14:
            continue

        zones.append([x, y, x + bw, y + bh])

    # =====================================================
    # 2. Цветные UI-зоны: желтые меню, серые панели, кнопки
    # =====================================================
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # желтые/серые/голубые области интерфейса 1С
    yellow_mask = cv2.inRange(hsv, np.array([15, 20, 120]), np.array([45, 255, 255]))
    gray_mask = cv2.inRange(hsv, np.array([0, 0, 120]), np.array([180, 45, 245]))

    ui_mask = cv2.bitwise_or(yellow_mask, gray_mask)

    kernel_ui = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 12))
    ui_mask = cv2.morphologyEx(ui_mask, cv2.MORPH_CLOSE, kernel_ui, iterations=2)

    contours, _ = cv2.findContours(ui_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)

        area = bw * bh

        if area < w * h * min_area_ratio:
            continue

        if bw < 100 or bh < 35:
            continue

        if y < h * 0.05 or y > h * 0.95:
            continue

        zones.append([x, y, x + bw, y + bh])

    # =====================================================
    # 3. Убираем вложенные и слишком текстовые зоны
    # =====================================================
    cleaned = []

    for z in zones:
        x0, y0, x1, y1 = z
        bw = x1 - x0
        bh = y1 - y0

        if bw <= 0 or bh <= 0:
            continue

        # обычные длинные строки текста
        if bw / max(bh, 1) > 16:
            continue

        # слишком большая зона почти на всю страницу — опасно, там текст инструкции
        if bw > w * 0.92 and bh > h * 0.45:
            continue

        is_inside_existing = False

        for old in cleaned:
            if box_inside(z, old, tolerance=20):
                is_inside_existing = True
                break

        if not is_inside_existing:
            cleaned.append(z)

    # =====================================================
    # 4. Аккуратное объединение только реально близких зон
    # =====================================================
    merged = []

    for z in sorted(cleaned, key=lambda b: (b[1], b[0])):
        placed = False

        for i, old in enumerate(merged):
            vertical_gap = max(0, max(z[1], old[1]) - min(z[3], old[3]))
            horizontal_overlap = min(z[2], old[2]) - max(z[0], old[0])

            # объединяем только если зоны почти соприкасаются
            if horizontal_overlap > 0 and vertical_gap < 20:
                merged[i] = union_box([old, z])
                placed = True
                break

        if not placed:
            merged.append(z)

    return merged


def crop_image_region(image_path, box):
    img = Image.open(image_path).convert("RGB")
    x0, y0, x1, y1 = normalize_box(box)
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(img.width, x1)
    y1 = min(img.height, y1)
    return img.crop((x0, y0, x1, y1))


def safe_read_ocr_on_zone(reader, page_image, zone, max_width=1000):
    zone_img = crop_image_region(page_image, zone)
    zw, zh = zone_img.size

    scale_back = 1.0

    if zw > max_width:
        scale = max_width / zw
        zone_img = zone_img.resize((int(zw * scale), int(zh * scale)))
        scale_back = 1.0 / scale

    img_np = np.array(zone_img)

    try:
        results = reader.readtext(
            img_np,
            detail=1,
            paragraph=False,
            batch_size=1,
            canvas_size=1024,
            mag_ratio=1.0,
        )
    finally:
        del img_np
        del zone_img
        gc.collect()

    return results, scale_back


def box_height(box):
    return box[3] - box[1]


def center_y(box):
    return (box[1] + box[3]) / 2


def same_line(a, b):
    h = max(box_height(a), box_height(b), 1)
    return abs(center_y(a) - center_y(b)) <= h * 0.75


def should_merge_words(a, b):
    box_a = a["bbox"]
    box_b = b["bbox"]

    if not same_line(box_a, box_b):
        return False

    gap = box_b[0] - box_a[2]
    h = max(box_height(box_a), box_height(box_b), 1)

    if gap < -h * 1.2:
        return False

    if gap > h * 3.5:
        return False

    text_a = normalize_ocr_text(a["text"])
    text_b = normalize_ocr_text(b["text"])

    if not text_a or not text_b:
        return False

    if len((text_a + " " + text_b).split()) > 7:
        return False

    return True


def merge_two(a, b):
    text = clean_text(a["text"] + " " + b["text"])

    ax0, ay0, ax1, ay1 = a["bbox"]
    bx0, by0, bx1, by1 = b["bbox"]

    item = dict(a)
    item["text"] = text
    item["normalized_text"] = normalize_ocr_text(text)
    item["bbox"] = [
        min(ax0, bx0),
        min(ay0, by0),
        max(ax1, bx1),
        max(ay1, by1),
    ]
    item["confidence"] = min(float(a.get("confidence", 1.0)), float(b.get("confidence", 1.0)))
    item["merged"] = True
    return item


def merge_nearby_words(elements):
    by_page_zone = {}

    for el in elements:
        key = (el["page"], el.get("zone_id", 0))
        by_page_zone.setdefault(key, []).append(el)

    merged_all = []

    for _, page_elements in by_page_zone.items():
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


def crop_box(image_path, bbox, out_path, pad=7):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    x0, y0, x1, y1 = bbox

    x0 = max(0, int(x0) - pad)
    y0 = max(0, int(y0) - pad)
    x1 = min(w, int(x1) + pad)
    y1 = min(h, int(y1) + pad)

    if x1 <= x0 or y1 <= y0:
        return False

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

    bad_exact = {
        "рис",
        "таблица",
        "страница",
        "инструкция",
        "оглавление",
    }

    if norm in bad_exact:
        return False

    bad_prefixes = [
        "если ",
        "при ",
        "после ",
        "далее ",
        "на рисунке",
        "в результате",
        "для того чтобы",
        "необходимо",
    ]

    if any(norm.startswith(p) for p in bad_prefixes):
        return False

    if len(norm.split()) > 8:
        return False

    return True


def save_final_crops(elements, out_dir):
    final = []
    seen = set()

    for idx, el in enumerate(elements):
        key = (el["page"], el.get("normalized_text", ""), tuple(el["bbox"]))

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

        try:
            item["ui_type"] = detect_ui_type(item["text"], item["bbox"], page_width=item.get("page_width"))
        except TypeError:
            item["ui_type"] = detect_ui_type(item["text"], item["bbox"])

        final.append(item)

    return final


def save_debug_zones(page_image_path, zones, out_path):
    img = cv2.imread(str(page_image_path))

    if img is None:
        return

    for i, z in enumerate(zones):
        x0, y0, x1, y1 = z
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 180, 0), 4)
        cv2.putText(img, f"UI zone {i}", (x0, max(20, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 120, 0), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-dir", default="data/pdf_rag")
    parser.add_argument("--langs", nargs="+", default=["ru", "en"])
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--max-pages", type=int, default=120)
    parser.add_argument("--min-conf", type=float, default=0.20)
    parser.add_argument("--max-ocr-width", type=int, default=1000)
    parser.add_argument("--debug-zones", action="store_true")
    args = parser.parse_args()

    rag_dir = Path(args.rag_dir)
    pages_dir = rag_dir / "pages"

    raw_crops_dir = rag_dir / "ui_element_crops"
    merged_crops_dir = rag_dir / "ui_element_crops_merged"
    debug_zones_dir = rag_dir / "debug_ui_zones"

    out_path = rag_dir / "ui_elements.jsonl"

    if not pages_dir.exists():
        raise FileNotFoundError(f"Pages directory not found: {pages_dir}")

    reader = easyocr.Reader(args.langs, gpu=args.gpu, quantize=True)

    page_images = sorted(pages_dir.glob("*.png"))[:args.max_pages]

    raw_elements = []
    crop_id = 0
    total_zones = 0

    for page_image in page_images:
        page_num_match = re.search(r"page_(\d+)", page_image.stem)

        if not page_num_match:
            continue

        page = int(page_num_match.group(1))

        pil_page = Image.open(page_image).convert("RGB")
        page_w, page_h = pil_page.size
        pil_page.close()

        zones = detect_ui_zones(page_image)
        total_zones += len(zones)

        print(f"OCR page {page}: {page_image} | UI zones: {len(zones)}")

        if args.debug_zones:
            save_debug_zones(page_image, zones, debug_zones_dir / f"{page_image.stem}_zones.png")

        if not zones:
            continue

        for zone_id, zone in enumerate(zones):
            try:
                results, scale_back = safe_read_ocr_on_zone(
                    reader,
                    page_image,
                    zone,
                    max_width=args.max_ocr_width,
                )
            except Exception as exc:
                print(f"WARNING: OCR failed page={page} zone={zone_id}: {exc}")
                continue

            for points, text, conf in results:
                text = clean_text(text)

                if conf < args.min_conf:
                    continue

                bbox = box_from_easyocr(points, scale_back=scale_back, offset=(zone[0], zone[1]))

                if not box_inside(bbox, zone, tolerance=12):
                    continue

                if not is_probably_ui_text(text):
                    continue

                if not is_likely_ui_candidate(text, bbox, page_w, page_h):
                    continue

                crop_path = raw_crops_dir / f"page_{page:04d}_ui_{crop_id:06d}.png"

                if not crop_box(page_image, bbox, crop_path, pad=5):
                    continue

                try:
                    ui_type = detect_ui_type(text, bbox, page_width=page_w)
                except TypeError:
                    ui_type = detect_ui_type(text, bbox)

                raw_elements.append({
                    "id": f"page_{page}_ui_raw_{crop_id}",
                    "page": page,
                    "text": text,
                    "normalized_text": normalize_ocr_text(text),
                    "ui_type": ui_type,
                    "confidence": float(conf),
                    "bbox": bbox,
                    "zone_id": zone_id,
                    "zone_bbox": zone,
                    "page_width": page_w,
                    "page_height": page_h,
                    "page_image": str(page_image).replace("\\", "/"),
                    "crop_image": str(crop_path).replace("\\", "/"),
                })

                crop_id += 1

            del results
            gc.collect()

    merged_elements = merge_nearby_words(raw_elements)
    final_elements = save_final_crops(merged_elements, merged_crops_dir)

    with open(out_path, "w", encoding="utf-8") as f:
        for el in final_elements:
            f.write(json.dumps(el, ensure_ascii=False) + "\n")

    print("=" * 80)
    print(f"Saved UI elements: {out_path}")
    print(f"UI zones: {total_zones}")
    print(f"Raw elements: {len(raw_elements)}")
    print(f"Merged elements: {len(final_elements)}")


if __name__ == "__main__":
    main()
