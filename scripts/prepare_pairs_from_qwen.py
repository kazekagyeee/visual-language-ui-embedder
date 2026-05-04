from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def load_items(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ["items", "components", "embeddings", "data"]:
            if key in data and isinstance(data[key], list):
                return data[key]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported embeddings.json format. Expected list or dict with items/components/embeddings/data.")


def get_vec(item: Dict[str, Any]) -> list[float]:
    for key in ["embedding", "qwen_long_vec", "vector", "vec"]:
        if key in item:
            return item[key]
    raise KeyError(f"No vector field in item keys={list(item.keys())}")


def get_text(item: Dict[str, Any]) -> str:
    for key in ["text", "description", "label", "name", "caption"]:
        if item.get(key):
            return str(item[key])
    return "ui component"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output/embeddings.json")
    parser.add_argument("--out-pairs", default="data/pairs.jsonl")
    parser.add_argument("--out-reference", default="data/reference_items.jsonl")
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    args = parser.parse_args()

    items = load_items(Path(args.input))
    pairs_out = Path(args.out_pairs)
    ref_out = Path(args.out_reference)
    pairs_out.parent.mkdir(parents=True, exist_ok=True)
    ref_out.parent.mkdir(parents=True, exist_ok=True)

    normalized = []
    for i, item in enumerate(items):
        vec = get_vec(item)
        normalized.append({
            "id": str(item.get("id", f"item_{i:06d}")),
            "text": get_text(item),
            "vec": vec,
            "meta": {k: v for k, v in item.items() if k not in {"embedding", "qwen_long_vec", "vector", "vec"}},
        })

    with ref_out.open("w", encoding="utf-8") as f:
        for item in normalized:
            f.write(json.dumps({
                "id": item["id"],
                "title": item["text"],
                "text": item["text"],
                "image_vec": item["vec"],
                "qwen_long_vec": item["vec"],
                "meta": item["meta"],
            }, ensure_ascii=False) + "\n")

    with pairs_out.open("w", encoding="utf-8") as f:
        for item in normalized:
            f.write(json.dumps({
                "id": f"pos_{item['id']}",
                "text": item["text"],
                "text_vec": item["vec"],
                "image_vec": item["vec"],
                "qwen_long_vec": item["vec"],
                "label": 1,
                "meta": item["meta"],
            }, ensure_ascii=False) + "\n")
            negatives = [x for x in normalized if x["id"] != item["id"]]
            for neg in random.sample(negatives, k=min(args.negatives_per_positive, len(negatives))):
                f.write(json.dumps({
                    "id": f"neg_{item['id']}_{neg['id']}",
                    "text": item["text"],
                    "text_vec": item["vec"],
                    "image_vec": neg["vec"],
                    "label": 0,
                    "meta": {"query_item": item["meta"], "negative_item": neg["meta"]},
                }, ensure_ascii=False) + "\n")

    print(f"Wrote {pairs_out}")
    print(f"Wrote {ref_out}")


if __name__ == "__main__":
    main()
