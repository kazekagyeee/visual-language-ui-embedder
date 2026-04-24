from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import random


def build_grouped_splits(
    samples: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, str]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        grouped[str(sample["image_path"])].append(idx)

    image_paths = list(grouped.keys())
    random.Random(seed).shuffle(image_paths)

    total = len(image_paths)
    train_cut = int(total * train_ratio)
    val_cut = train_cut + int(total * val_ratio)

    mapping: dict[str, str] = {}
    for image_path in image_paths[:train_cut]:
        mapping[image_path] = "train"
    for image_path in image_paths[train_cut:val_cut]:
        mapping[image_path] = "val"
    for image_path in image_paths[val_cut:]:
        mapping[image_path] = "test"
    return mapping


def save_split_mapping(split_mapping: dict[str, str], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split_mapping, ensure_ascii=False, indent=2), encoding="utf-8")


def load_split_mapping(path: str | Path) -> dict[str, str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
