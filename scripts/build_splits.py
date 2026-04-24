from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.audit import load_json_samples
from src.data.splits import build_grouped_splits, save_split_mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = load_json_samples(args.json_path)
    split_mapping = build_grouped_splits(
        samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    save_split_mapping(split_mapping, args.output_path)
    print(f"saved {len(split_mapping)} grouped image-path assignments to {args.output_path}")


if __name__ == "__main__":
    main()
