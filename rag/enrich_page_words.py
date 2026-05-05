# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import fitz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--rag-dir", required=True)
    parser.add_argument("--zoom", type=float, default=2.0)
    args = parser.parse_args()

    rag_dir = Path(args.rag_dir)
    items_path = rag_dir / "items.jsonl"

    doc = fitz.open(args.pdf)

    page_words = {}

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_number = page_idx + 1

        words = []

        for w in page.get_text("words"):
            x0, y0, x1, y1, text, *_ = w

            if not text.strip():
                continue

            words.append({
                "text": text.strip(),
                "bbox_px": [
                    int(x0 * args.zoom),
                    int(y0 * args.zoom),
                    int(x1 * args.zoom),
                    int(y1 * args.zoom),
                ],
            })

        page_words[page_number] = words

    updated = []

    with open(items_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            item["page_words"] = page_words.get(item["page"], [])
            updated.append(item)

    with open(items_path, "w", encoding="utf-8") as f:
        for item in updated:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Updated items with page words: {items_path}")
    print(f"Items: {len(updated)}")


if __name__ == "__main__":
    main()
