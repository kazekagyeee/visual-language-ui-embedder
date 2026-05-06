# -*- coding: utf-8 -*-

import argparse
import json
import random
from pathlib import Path


def load_elements(path):
    elements = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            el = json.loads(line)

            if not Path(el["crop_image"]).exists():
                continue

            if len(el["normalized_text"]) < 3:
                continue

            elements.append(el)

    return elements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-dir", default="data/pdf_rag")
    parser.add_argument("--out", default="data/ui_element_pairs.jsonl")
    parser.add_argument("--negatives-per-positive", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    rag_dir = Path(args.rag_dir)
    elements_path = rag_dir / "ui_elements.jsonl"

    elements = load_elements(elements_path)

    if len(elements) < 2:
        raise RuntimeError("Too few UI elements. Run build_ui_elements.py first.")

    pairs = []

    for el in elements:
        pairs.append({
            "text": el["text"],
            "image": el["crop_image"],
            "label": 1,
            "page": el["page"],
            "bbox_px": el["bbox"],
            "ui_element_id": el["id"],
        })

        for _ in range(args.negatives_per_positive):
            neg = random.choice(elements)

            tries = 0
            while neg["normalized_text"] == el["normalized_text"] and tries < 30:
                neg = random.choice(elements)
                tries += 1

            pairs.append({
                "text": el["text"],
                "image": neg["crop_image"],
                "label": 0,
                "page": neg["page"],
                "bbox_px": neg["bbox"],
                "ui_element_id": el["id"],
                "negative_text": neg["text"],
            })

    random.shuffle(pairs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"UI elements: {len(elements)}")
    print(f"Saved pairs: {out_path}")
    print(f"Pairs: {len(pairs)}")

    print("\nExamples:")
    for el in elements[:30]:
        print(f"- page={el['page']} text={el['text']} crop={el['crop_image']}")


if __name__ == "__main__":
    main()
