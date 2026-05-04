import argparse
import json
import random
from pathlib import Path

import fitz


def fake_embedding(text: str, dim: int = 4):
    random.seed(abs(hash(text)) % (10**8))
    return [round(random.random(), 4) for _ in range(dim)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default="data_source/instruction.pdf")
    parser.add_argument("--pairs-out", default="data/pairs.jsonl")
    parser.add_argument("--index-out", default="data/index_items.jsonl")
    parser.add_argument("--dim", type=int, default=4)
    parser.add_argument("--max-pages", type=int, default=80)
    args = parser.parse_args()

    Path(args.pairs_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.index_out).parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(args.pdf)
    pages = []

    for i in range(min(args.max_pages, len(doc))):
        page = doc[i]
        text = page.get_text("text").strip()

        if len(text) < 20:
            continue

        text_vec = fake_embedding("TEXT:" + text, args.dim)
        image_vec = fake_embedding("IMAGE:" + text, args.dim)

        pages.append(
            {
                "id": f"page_{i + 1}",
                "page": i + 1,
                "text": text,
                "text_vec": text_vec,
                "image_vec": image_vec,
            }
        )

    rows = []

    for p in pages:
        rows.append(
            {
                "text_vec": p["text_vec"],
                "image_vec": p["image_vec"],
                "label": 1,
                "metadata": {
                    "page": p["page"],
                    "type": "positive",
                    "text": p["text"][:700],
                },
            }
        )

    for p in pages:
        negatives = [x for x in pages if x["page"] != p["page"]]
        if not negatives:
            continue

        other = random.choice(negatives)
        rows.append(
            {
                "text_vec": p["text_vec"],
                "image_vec": other["image_vec"],
                "label": 0,
                "metadata": {
                    "text_page": p["page"],
                    "image_page": other["page"],
                    "type": "negative",
                    "text": p["text"][:700],
                },
            }
        )

    with open(args.pairs_out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(args.index_out, "w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Saved pairs: {len(rows)} -> {args.pairs_out}")
    print(f"Saved index items: {len(pages)} -> {args.index_out}")


if __name__ == "__main__":
    main()
