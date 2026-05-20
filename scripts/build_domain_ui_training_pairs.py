# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from rag.domain_1c_dictionary import (
    CANONICAL_1C_TERMS,
    expand_training_queries_for_term,
    similarity,
)


def read_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def find_items_file(index_dir: Path) -> Path:
    for name in ["ui_items.jsonl", "items.jsonl"]:
        p = index_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Не найден ui_items.jsonl/items.jsonl в {index_dir}")


def item_text(item: dict) -> str:
    return item.get("text") or item.get("normalized_text") or item.get("raw_text") or ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ui-index-dir", default="data/ui_index")
    parser.add_argument("--base-pairs", default="data/ui_training_pairs.jsonl")
    parser.add_argument("--out", default="data/ui_training_pairs_domain.jsonl")
    parser.add_argument("--max-negatives-per-positive", type=int, default=2)
    args = parser.parse_args()

    random.seed(42)

    items = read_jsonl(find_items_file(Path(args.ui_index_dir)))

    pairs = []
    base_path = Path(args.base_pairs)
    if base_path.exists():
        pairs.extend(read_jsonl(base_path))

    positive_added = 0
    negative_added = 0

    all_items = [x for x in items if item_text(x)]

    for canonical in CANONICAL_1C_TERMS:
        positives = [x for x in all_items if similarity(item_text(x), canonical) >= 0.78]
        if not positives:
            continue

        queries = expand_training_queries_for_term(canonical)

        for query in queries:
            for pos in positives[:8]:
                pairs.append({
                    "query": query,
                    "text": item_text(pos),
                    "ui_text": item_text(pos),
                    "label": 1,
                    "target": canonical,
                    "source": "domain_dictionary_positive",
                })
                positive_added += 1

                negatives = [
                    x for x in all_items
                    if similarity(item_text(x), canonical) < 0.45
                ]
                random.shuffle(negatives)

                for neg in negatives[:args.max_negatives_per_positive]:
                    pairs.append({
                        "query": query,
                        "text": item_text(neg),
                        "ui_text": item_text(neg),
                        "label": 0,
                        "target": canonical,
                        "source": "domain_dictionary_negative",
                    })
                    negative_added += 1

    write_jsonl(Path(args.out), pairs)

    print("[OK] domain training pairs built")
    print("base_pairs:", len(pairs) - positive_added - negative_added)
    print("positive_added:", positive_added)
    print("negative_added:", negative_added)
    print("total:", len(pairs))
    print("saved:", args.out)


if __name__ == "__main__":
    main()