# -*- coding: utf-8 -*-

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/ui_element_pairs.jsonl")
    parser.add_argument("--train-out", default="data/ui_element_pairs.train.jsonl")
    parser.add_argument("--test-out", default="data/ui_element_pairs.test.jsonl")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    rows = []
    with open(args.pairs, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    random.shuffle(rows)

    test_size = max(1, int(len(rows) * args.test_ratio))
    test_rows = rows[:test_size]
    train_rows = rows[test_size:]

    Path(args.train_out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.train_out, "w", encoding="utf-8") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(args.test_out, "w", encoding="utf-8") as f:
        for row in test_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Total: {len(rows)}")
    print(f"Train: {len(train_rows)} -> {args.train_out}")
    print(f"Test: {len(test_rows)} -> {args.test_out}")


if __name__ == "__main__":
    main()
