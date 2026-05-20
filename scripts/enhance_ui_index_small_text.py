# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from rag.ocr_cleanup import cleanup_ocr_text


def normalize(text):
    text = cleanup_ocr_text(text)
    text = str(text).lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_jsonl(path):
    rows = []
    if not Path(path).exists():
        return rows

    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def bbox_from_points(points, scale=1.0):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    return [
        int(min(xs) / scale),
        int(min(ys) / scale),
        int(max(xs) / scale),
        int(max(ys) / scale),
    ]


def make_variants(img):
    rgb = np.array(img.convert("RGB"))
    variants = []

    variants.append(("orig", rgb, 1.0))

    big = cv2.resize(rgb, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    variants.append(("big3", big, 3.0))

    huge = cv2.resize(rgb, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
    variants.append(("big4", huge, 4.0))

    gray = cv2.cvtColor(huge, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    variants.append(("gray4", cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), 4.0))

    return variants


def is_useful_small_text(text):
    n = normalize(text)

    if len(n) < 4:
        return False

    keywords = [
        "монитор",
        "интернет",
        "поддерж",
        "заполнить",
        "начните",
        "отсюда",
        "инн",
        "контрагент",
    ]

    return any(k in n for k in keywords)


def rebuild_embeddings(index_dir, model_name):
    items = load_jsonl(index_dir / "ui_items.jsonl")
    model = SentenceTransformer(model_name)

    texts = [
        f"{item.get('text', '')} {item.get('ui_type', '')} страница {item.get('page', '')}"
        for item in items
    ]

    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    np.save(index_dir / "ui_embeddings.npy", np.asarray(emb, dtype=np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default="data/ui_index")
    parser.add_argument("--pdf-name", default=None)
    parser.add_argument("--page", type=int, default=None)
    parser.add_argument("--min-conf", type=float, default=0.05)
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    args = parser.parse_args()

    import easyocr
    reader = easyocr.Reader(["ru", "en"], gpu=False)

    index_dir = Path(args.index_dir)
    items_path = index_dir / "ui_items.jsonl"
    items = load_jsonl(items_path)

    screenshots = {}

    for item in items:
        if args.pdf_name and item.get("pdf_name") != args.pdf_name:
            continue
        if args.page and int(item.get("page", -1)) != args.page:
            continue

        path = item.get("screenshot_image")
        if path:
            screenshots[path] = item

    print(f"screenshots: {len(screenshots)}")

    new_items = []
    existing = {
        (
            x.get("pdf_name"),
            x.get("page"),
            x.get("screenshot_idx"),
            normalize(x.get("text")),
            tuple(x.get("bbox") or []),
        )
        for x in items
    }

    for screenshot_path, base_item in screenshots.items():
        path = Path(screenshot_path)

        if not path.exists():
            continue

        img = Image.open(path).convert("RGB")

        for variant_name, arr, scale in make_variants(img):
            try:
                ocr = reader.readtext(arr, detail=1, paragraph=False)
            except Exception:
                continue

            local_i = 0

            for points, text, conf in ocr:
                text = cleanup_ocr_text(text)

                if conf < args.min_conf:
                    continue

                if not is_useful_small_text(text):
                    continue

                bbox = bbox_from_points(points, scale=scale)

                local_i += 1

                row = {
                    "id": f"{base_item.get('pdf_name')}_p{base_item.get('page')}_s{base_item.get('screenshot_idx')}_small_{variant_name}_{local_i}",
                    "pdf_name": base_item.get("pdf_name"),
                    "page": base_item.get("page"),
                    "screenshot_idx": base_item.get("screenshot_idx"),
                    "screenshot_image": screenshot_path,
                    "text": text,
                    "normalized_text": normalize(text),
                    "bbox": bbox,
                    "ui_type": "small_text",
                    "confidence": float(conf),
                    "source": "small_text_enhance",
                }

                key = (
                    row.get("pdf_name"),
                    row.get("page"),
                    row.get("screenshot_idx"),
                    normalize(row.get("text")),
                    tuple(row.get("bbox") or []),
                )

                if key in existing:
                    continue

                existing.add(key)
                new_items.append(row)

    print(f"new small text items: {len(new_items)}")

    all_items = items + new_items
    save_jsonl(items_path, all_items)
    rebuild_embeddings(index_dir, args.model)

    print("[DONE]")


if __name__ == "__main__":
    main()
