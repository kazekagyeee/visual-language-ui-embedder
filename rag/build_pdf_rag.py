# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import cv2
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer


def detect_red_annotations(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return []

    # BGR
    b, g, r = cv2.split(img)

    mask = (r > 150) & (g < 110) & (b < 110)
    mask = mask.astype("uint8") * 255

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

        if w > w_img * 0.95 or h > h_img * 0.95:
            continue

        boxes.append([int(x), int(y), int(x + w), int(y + h)])

    # убираем вложенные мелкие дубли
    filtered = []
    for box in boxes:
        x0, y0, x1, y1 = box
        area = (x1 - x0) * (y1 - y0)

        duplicate = False
        for other in filtered:
            ox0, oy0, ox1, oy1 = other
            if x0 >= ox0 and y0 >= oy0 and x1 <= ox1 and y1 <= oy1:
                duplicate = True
                break

        if not duplicate:
            filtered.append(box)

    return filtered


def save_crop(page, bbox, matrix, path):
    rect = fitz.Rect(*bbox)
    pix = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)
    pix.save(str(path))


def crop_from_page_image(page_image_path, box_px, out_path):
    img = cv2.imread(str(page_image_path))
    x0, y0, x1, y1 = box_px
    crop = img[y0:y1, x0:x1]
    cv2.imwrite(str(out_path), crop)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default="data_source/instruction.pdf")
    parser.add_argument("--out-dir", default="data/pdf_rag")
    parser.add_argument("--max-pages", type=int, default=120)
    parser.add_argument("--zoom", type=float, default=2.0)
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    pages_dir = out_dir / "pages"
    crops_dir = out_dir / "crops"
    target_crops_dir = out_dir / "target_crops"

    pages_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    target_crops_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(args.pdf)
    model = SentenceTransformer(args.model)

    items = []
    matrix = fitz.Matrix(args.zoom, args.zoom)

    for page_idx in range(min(args.max_pages, len(doc))):
        page = doc[page_idx]
        page_number = page_idx + 1

        page_image_path = pages_dir / f"page_{page_number:04d}.png"
        page_pix = page.get_pixmap(matrix=matrix, alpha=False)
        page_pix.save(str(page_image_path))

        red_boxes_px = detect_red_annotations(page_image_path)

        target_crop_path = None
        if red_boxes_px:
            # берем самую большую красную аннотацию как главный UI-target
            main_box = max(
                red_boxes_px,
                key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
            )
            target_crop_path = target_crops_dir / f"page_{page_number:04d}_target.png"
            crop_from_page_image(page_image_path, main_box, target_crop_path)

        blocks = page.get_text("blocks")
        block_id = 0

        for block in blocks:
            x0, y0, x1, y1, text, *_ = block
            text = text.strip()

            if len(text) < 20:
                continue

            text_bbox = [x0, y0, x1, y1]
            text_crop_path = crops_dir / f"page_{page_number:04d}_block_{block_id:03d}.png"
            save_crop(page, text_bbox, matrix, text_crop_path)

            items.append({
                "id": f"page_{page_number}_block_{block_id}",
                "page": page_number,
                "block_id": block_id,
                "text": text,

                "bbox": text_bbox,
                "crop_image": str(text_crop_path).replace("\\", "/"),

                "red_bboxes_px": red_boxes_px,
                "target_crop_image": str(target_crop_path).replace("\\", "/") if target_crop_path else None,

                "page_image": str(page_image_path).replace("\\", "/"),
            })

            block_id += 1

    texts = [item["text"] for item in items]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    with open(out_dir / "items.jsonl", "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    np.save(out_dir / "embeddings.npy", np.array(embeddings, dtype=np.float32))

    print(f"Saved items: {len(items)}")
    print(f"Saved red UI targets into: {target_crops_dir}")


if __name__ == "__main__":
    main()
