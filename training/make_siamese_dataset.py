# -*- coding: utf-8 -*-

import argparse
import json
import random
from pathlib import Path


def load_items(rag_dir):
    items = []

    with open(Path(rag_dir) / "items.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            crops = item.get("target_crop_images") or []
            if not crops:
                crops = [item.get("crop_image")]

            crops = [c for c in crops if c and Path(c).exists()]

            if not crops:
                continue

            item["siamese_crops"] = crops
            items.append(item)

    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-dir", default="data/pdf_rag")
    parser.add_argument("--out", default="data/siamese_pairs.jsonl")
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    items = load_items(args.rag_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = []

    for item in items:
        text = item["text"]

        for crop in item["siamese_crops"]:
            pairs.append({
                "text": text,
                "image": crop,
                "label": 1,
                "page": item["page"],
                "source_id": item["id"],
            })

            for _ in range(args.negatives_per_positive):
                neg = random.choice(items)

                tries = 0
                while neg["id"] == item["id"] and tries < 20:
                    neg = random.choice(items)
                    tries += 1

                neg_crop = random.choice(neg["siamese_crops"])

                pairs.append({
                    "text": text,
                    "image": neg_crop,
                    "label": 0,
                    "page": item["page"],
                    "source_id": item["id"],
                    "negative_source_id": neg["id"],
                })

    random.shuffle(pairs)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved siamese pairs: {out_path}")
    print(f"Pairs: {len(pairs)}")
    print(f"Items used: {len(items)}")


if __name__ == "__main__":
    main()
