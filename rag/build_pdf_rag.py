# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import cv2
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer


def detect_red_annotations(page_image_path):
    img = cv2.imread(str(page_image_path))
    if img is None:
        return []

    b, g, r = cv2.split(img)
    mask = ((r > 150) & (g < 120) & (b < 120)).astype("uint8") * 255

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    boxes = []
    h_img, w_img = img.shape[:2]

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area < 40:
            continue
        if w < 8 or h < 8:
            continue
        if w > w_img * 0.98 or h > h_img * 0.98:
            continue

        boxes.append([int(x), int(y), int(x + w), int(y + h)])

    return boxes


def detect_pdf_image_blocks(page, zoom):
    result = []
    page_dict = page.get_text("dict")

    for block in page_dict.get("blocks", []):
        if block.get("type") != 1:
            continue

        x0, y0, x1, y1 = block["bbox"]
        w = x1 - x0
        h = y1 - y0

        if w < 60 or h < 40:
            continue

        box_px = [
            int(x0 * zoom),
            int(y0 * zoom),
            int(x1 * zoom),
            int(y1 * zoom),
        ]

        result.append(box_px)

    return result


def detect_auto_visual_blocks(page, page_image_path, zoom):
    # 1) лучший вариант: реальные image-блоки PDF
    boxes = detect_pdf_image_blocks(page, zoom)

    if boxes:
        return boxes

    # 2) fallback: ищем крупные области с рамками, но НЕ текстовые абзацы
    img = cv2.imread(str(page_image_path))
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 80, 180)

    kernel = np.ones((10, 10), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = img.shape[:2]
    boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if area < w_img * h_img * 0.025:
            continue

        if w < 120 or h < 80:
            continue

        if w > w_img * 0.98 or h > h_img * 0.98:
            continue

        # текстовые абзацы обычно имеют много длинных горизонтальных строк,
        # а скриншоты/рисунки имеют рамки и плотную структуру
        roi = gray[y:y+h, x:x+w]
        dark_ratio = np.mean(roi < 120)

        if dark_ratio < 0.015:
            continue

        boxes.append([int(x), int(y), int(x + w), int(y + h)])

    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return boxes[:4]


def nearest_target_for_text(text_bbox_px, target_boxes_px):
    if not target_boxes_px:
        return [], []

    tx0, ty0, tx1, ty1 = text_bbox_px
    tcx = (tx0 + tx1) / 2
    tcy = (ty0 + ty1) / 2

    scored = []

    for box in target_boxes_px:
        x0, y0, x1, y1 = box
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2

        dist = abs(tcx - cx) + abs(tcy - cy)

        # если картинка рядом ниже или выше текста — бонус
        if abs(cy - tcy) < 900:
            dist *= 0.75

        scored.append((dist, box))

    scored.sort(key=lambda x: x[0])

    best = [scored[0][1]]
    return best, [target_boxes_px.index(best[0])]


def crop_from_page_image(page_image_path, box_px, out_path, pad=8):
    img = cv2.imread(str(page_image_path))
    if img is None:
        return False

    h, w = img.shape[:2]
    x0, y0, x1, y1 = box_px

    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)

    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return False

    cv2.imwrite(str(out_path), crop)
    return True


def save_pdf_crop(page, bbox, matrix, path):
    rect = fitz.Rect(*bbox)
    pix = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)
    pix.save(str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--mode", choices=["red", "auto"], default="auto")
    parser.add_argument("--max-pages", type=int, default=120)
    parser.add_argument("--zoom", type=float, default=2.0)
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    pages_dir = out_dir / "pages"
    text_crops_dir = out_dir / "crops"
    target_crops_dir = out_dir / "target_crops"

    pages_dir.mkdir(parents=True, exist_ok=True)
    text_crops_dir.mkdir(parents=True, exist_ok=True)
    target_crops_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(args.pdf)
    model = SentenceTransformer(args.model)
    matrix = fitz.Matrix(args.zoom, args.zoom)

    items = []

    for page_idx in range(min(args.max_pages, len(doc))):
        page = doc[page_idx]
        page_number = page_idx + 1

        page_image_path = pages_dir / f"page_{page_number:04d}.png"
        page_pix = page.get_pixmap(matrix=matrix, alpha=False)
        page_pix.save(str(page_image_path))

        if args.mode == "red":
            page_targets_px = detect_red_annotations(page_image_path)
        else:
            page_targets_px = detect_auto_visual_blocks(page, page_image_path, args.zoom)

        page_target_crop_paths = []

        for target_id, target_box in enumerate(page_targets_px):
            target_crop_path = target_crops_dir / f"page_{page_number:04d}_target_{target_id:03d}.png"

            ok = crop_from_page_image(
                page_image_path=page_image_path,
                box_px=target_box,
                out_path=target_crop_path,
            )

            if ok:
                page_target_crop_paths.append(str(target_crop_path).replace("\\", "/"))

        blocks = page.get_text("blocks")
        block_id = 0

        for block in blocks:
            x0, y0, x1, y1, text, *_ = block
            text = text.strip()

            if len(text) < 20:
                continue

            text_bbox_pdf = [x0, y0, x1, y1]
            text_bbox_px = [
                int(x0 * args.zoom),
                int(y0 * args.zoom),
                int(x1 * args.zoom),
                int(y1 * args.zoom),
            ]

            selected_targets_px, selected_ids = nearest_target_for_text(
                text_bbox_px,
                page_targets_px,
            )

            selected_crop_paths = [
                page_target_crop_paths[i]
                for i in selected_ids
                if i < len(page_target_crop_paths)
            ]

            text_crop_path = text_crops_dir / f"page_{page_number:04d}_block_{block_id:03d}.png"
            save_pdf_crop(page, text_bbox_pdf, matrix, text_crop_path)

            items.append({
                "id": f"page_{page_number}_block_{block_id}",
                "page": page_number,
                "block_id": block_id,
                "text": text,

                "bbox": text_bbox_pdf,
                "crop_image": str(text_crop_path).replace("\\", "/"),

                "target_mode": args.mode,
                "target_bboxes_px": selected_targets_px,
                "target_crop_images": selected_crop_paths,

                "all_page_targets_px": page_targets_px,
                "page_image": str(page_image_path).replace("\\", "/"),
            })

            block_id += 1

    texts = [item["text"] for item in items]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    with open(out_dir / "items.jsonl", "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    np.save(out_dir / "embeddings.npy", np.array(embeddings, dtype=np.float32))

    print(f"PDF: {args.pdf}")
    print(f"Mode: {args.mode}")
    print(f"Saved items: {len(items)}")
    print(f"Saved target crops: {target_crops_dir}")


if __name__ == "__main__":
    main()

