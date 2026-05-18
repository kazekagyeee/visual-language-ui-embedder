# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from rag.ocr_cleanup import cleanup_ocr_text


def normalize(text):
    text = cleanup_ocr_text(text)
    text = str(text).lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def union_bbox(items):
    xs0, ys0, xs1, ys1 = [], [], [], []

    for item in items:
        b = item.get("bbox")
        if not b:
            continue
        xs0.append(b[0])
        ys0.append(b[1])
        xs1.append(b[2])
        ys1.append(b[3])

    if not xs0:
        return None

    return [min(xs0), min(ys0), max(xs1), max(ys1)]


def same_line(a, b, y_tol=18):
    ay = (a["bbox"][1] + a["bbox"][3]) / 2
    by = (b["bbox"][1] + b["bbox"][3]) / 2
    return abs(ay - by) <= y_tol


def make_merged_items(items):
    grouped = {}

    for item in items:
        key = (
            item.get("pdf_name"),
            item.get("page"),
            item.get("screenshot_idx"),
            item.get("screenshot_image"),
        )
        grouped.setdefault(key, []).append(item)

    merged = []

    for key, group in grouped.items():
        group = [x for x in group if x.get("bbox") and normalize(x.get("text"))]
        group.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))

        lines = []

        for item in group:
            placed = False

            for line in lines:
                if same_line(line[0], item):
                    line.append(item)
                    placed = True
                    break

            if not placed:
                lines.append([item])

        for line_idx, line in enumerate(lines, start=1):
            line.sort(key=lambda x: x["bbox"][0])

            text = " ".join(cleanup_ocr_text(x.get("text", "")) for x in line)
            text = re.sub(r"\s+", " ", text).strip()

            n = normalize(text)

            if len(n) < 4:
                continue

            if len(n.split()) > 10:
                continue

            bbox = union_bbox(line)
            if not bbox:
                continue

            first = line[0]
            merged.append(
                {
                    "id": f"{first.get('pdf_name')}_p{first.get('page')}_s{first.get('screenshot_idx')}_merged_{line_idx}",
                    "pdf_name": first.get("pdf_name"),
                    "page": first.get("page"),
                    "screenshot_idx": first.get("screenshot_idx"),
                    "screenshot_image": first.get("screenshot_image"),
                    "text": text,
                    "normalized_text": n,
                    "bbox": bbox,
                    "ui_type": "merged_text",
                    "confidence": 1.0,
                    "source": "postprocess_merge",
                }
            )

    return merged


def dedupe(items):
    result = []
    seen = set()

    for item in items:
        key = (
            item.get("pdf_name"),
            item.get("page"),
            item.get("screenshot_idx"),
            normalize(item.get("text")),
            tuple(item.get("bbox") or []),
        )

        if key in seen:
            continue

        seen.add(key)

        item["text"] = cleanup_ocr_text(item.get("text", ""))
        item["normalized_text"] = normalize(item.get("text", ""))

        result.append(item)

    return result


def rebuild_embeddings(index_dir, model_name):
    items = load_jsonl(index_dir / "ui_items.jsonl")

    model = SentenceTransformer(model_name)

    texts = [
        f"{item.get('text', '')} {item.get('ui_type', '')} страница {item.get('page', '')}"
        for item in items
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    np.save(index_dir / "ui_embeddings.npy", np.asarray(embeddings, dtype=np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default="data/ui_index")
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    items_path = index_dir / "ui_items.jsonl"

    items = load_jsonl(items_path)
    print(f"Original items: {len(items)}")

    merged = make_merged_items(items)
    print(f"Merged items: {len(merged)}")

    all_items = dedupe(items + merged)
    print(f"Final items: {len(all_items)}")

    save_jsonl(items_path, all_items)
    rebuild_embeddings(index_dir, args.model)

    print("[DONE]")


if __name__ == "__main__":
    main()
