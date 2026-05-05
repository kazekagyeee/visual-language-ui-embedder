# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_items(rag_dir):
    items = []

    with open(Path(rag_dir) / "items.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    return items


def group_by_page(items):
    pages = {}

    for item in items:
        page = item["page"]
        pages.setdefault(page, []).append(item)

    return pages


def make_page_description(page_items):
    texts = []

    for item in page_items:
        text = clean_text(item["text"])
        if text:
            texts.append(text)

    joined = " ".join(texts)

    return clean_text(
        "Описание всей страницы интерфейса: "
        + joined[:2500]
    )


def make_crop_description(item, page_description):
    text = clean_text(item["text"])

    return clean_text(
        "Описание UI-элемента или фрагмента интерфейса: "
        f"{text}. "
        "Этот фрагмент относится к странице, где описано: "
        + page_description[:1200]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-dir", default="data/pdf_rag")
    args = parser.parse_args()

    rag_dir = Path(args.rag_dir)
    items = load_items(rag_dir)
    pages = group_by_page(items)

    output = []

    for page, page_items in pages.items():
        page_description = make_page_description(page_items)

        page_image = page_items[0].get("page_image")

        output.append({
            "type": "page",
            "page": page,
            "text": page_description,
            "page_image": page_image,
            "target_crop_image": None,
            "source_item_id": None,
        })

        for item in page_items:
            crop_images = item.get("target_crop_images") or []

            if not crop_images:
                crop_images = [item.get("crop_image")]

            for crop in crop_images:
                if not crop:
                    continue

                output.append({
                    "type": "crop",
                    "page": item["page"],
                    "block_id": item["block_id"],
                    "text": make_crop_description(item, page_description),
                    "page_image": item.get("page_image"),
                    "target_crop_image": crop,
                    "source_item_id": item.get("id"),
                })

    out_path = rag_dir / "visual_descriptions.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for row in output:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved visual descriptions: {out_path}")
    print(f"Rows: {len(output)}")


if __name__ == "__main__":
    main()
