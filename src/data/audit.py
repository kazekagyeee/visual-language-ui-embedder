from __future__ import annotations

from collections import Counter
from pathlib import Path
import json

from PIL import Image

from .bbox_utils import denormalize_bbox, is_valid_normalized_bbox


def load_json_samples(json_path: str | Path) -> list[dict]:
    return json.loads(Path(json_path).read_text(encoding="utf-8"))


def resolve_image_path(image_path: str | Path, json_path: str | Path | None = None) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    if json_path is not None:
        base_dir = Path(json_path).resolve().parent
        return (base_dir / path).resolve()
    return path.resolve()


def audit_samples(
    samples: list[dict],
    json_path: str | Path | None = None,
    bbox_epsilon: float = 1e-3,
    min_crop_size_px: int = 4,
) -> dict:
    stats = Counter()
    bbox_sizes: list[tuple[int, int]] = []
    unique_pairs = set()

    for sample in samples:
        image_path = str(sample.get("image_path", ""))
        text = str(sample.get("text", "")).strip()
        pos_bbox = sample.get("pos_bbox")
        neg_bbox = sample.get("neg_bbox")

        if not text:
            stats["empty_text"] += 1
            continue
        if not is_valid_normalized_bbox(pos_bbox or [], bbox_epsilon):
            stats["invalid_pos_bbox"] += 1
            continue
        if not is_valid_normalized_bbox(neg_bbox or [], bbox_epsilon):
            stats["invalid_neg_bbox"] += 1
            continue
        try:
            resolved_image_path = resolve_image_path(image_path, json_path=json_path)
            with Image.open(resolved_image_path) as img:
                width, height = img.size
        except Exception:
            stats["unreadable_image"] += 1
            continue

        pos_px = denormalize_bbox(pos_bbox, width, height)
        neg_px = denormalize_bbox(neg_bbox, width, height)
        if pos_px.width < min_crop_size_px or pos_px.height < min_crop_size_px:
            stats["small_pos_bbox"] += 1
            continue
        if neg_px.width < min_crop_size_px or neg_px.height < min_crop_size_px:
            stats["small_neg_bbox"] += 1
            continue

        bbox_sizes.extend([(pos_px.width, pos_px.height), (neg_px.width, neg_px.height)])
        stats["valid_samples"] += 1
        dedup_key = (image_path, tuple(round(v, 6) for v in pos_bbox), text.strip().lower())
        if dedup_key in unique_pairs:
            stats["duplicate_anchor_pos"] += 1
        else:
            unique_pairs.add(dedup_key)

    widths = sorted(size[0] for size in bbox_sizes)
    heights = sorted(size[1] for size in bbox_sizes)

    def pct(values: list[int], q: float) -> float:
        if not values:
            return 0.0
        idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
        return float(values[idx])

    return {
        "counts": dict(stats),
        "bbox_width_p10": pct(widths, 0.1),
        "bbox_width_p50": pct(widths, 0.5),
        "bbox_width_p90": pct(widths, 0.9),
        "bbox_height_p10": pct(heights, 0.1),
        "bbox_height_p50": pct(heights, 0.5),
        "bbox_height_p90": pct(heights, 0.9),
    }
