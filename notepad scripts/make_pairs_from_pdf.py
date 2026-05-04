import argparse
import json
import random
from pathlib import Path

import fitz
import numpy as np


def fake_embedding(text: str, dim: int = 4):
    random.seed(abs(hash(text)) % (10**8))
    return [round(random.random(), 4) for _ in range(dim)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--out", default="data/pairs.jsonl")
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--max-pages", type=int, default=30)
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(args.pdf)
    pages = []

    for i, page in enumerate(doc[: args.max_pages]):
        text = page.get_text("text").strip()
        if len(text) < 20:
            continue

        pages.append({
            "page": i + 1,
            "text": text,
            "text_embedding": fake_embedding("TEXT:" + text, args.dim),
            "image_embedding": fake_embedding("IMAGE:" + text, args.dim),
        })

    rows = []

    for p in pages:
        rows.append({
            "text_embedding": p["text_embedding"],
            "image_embedding": p["image_embedding"],
            "label": 1,
            "metadata": {
                "page": p["page"],
                "type": "positive",
                "text": p["text"][:500],
            }
        })

    for p in pages:
        other = random.choice([x for x in pages if x["page"] != p["page"]])
        rows.append({
            "text_embedding": p["text_embedding"],
            "image_embedding": other["image_embedding"],
            "label": 0,
            "metadata": {
                "text_page": p["page"],
                "image_page": other["page"],
                "type": "negative",
                "text": p["text"][:500],
            }
        })

    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} pairs to {args.out}")


if __name__ == "__main__":
    main()