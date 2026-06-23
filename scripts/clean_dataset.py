from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.audit import load_json_samples, resolve_image_path
from src.data.bbox_utils import clamp_bbox_norm, denormalize_bbox, is_valid_normalized_bbox


def round_bbox_key(bbox: list[float], ndigits: int = 6) -> tuple[float, ...]:
    return tuple(round(float(v), ndigits) for v in bbox)


def clean_samples(
    samples: list[dict],
    json_path: str | Path,
    bbox_epsilon: float,
    min_crop_size_px: int,
    drop_duplicates: bool,
) -> tuple[list[dict], dict]:
    cleaned: list[dict] = []
    stats = Counter()
    seen = set()

    for idx, sample in enumerate(samples):
        image_path = str(sample.get("image_path", "")).strip()
        text = str(sample.get("text", "")).strip()
        pos_bbox = sample.get("pos_bbox")
        neg_bbox = sample.get("neg_bbox")

        if not image_path:
            stats["empty_image_path"] += 1
            continue
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
                img.verify()
        except Exception:
            stats["unreadable_image"] += 1
            continue

        pos_bbox_clean = clamp_bbox_norm(pos_bbox)
        neg_bbox_clean = clamp_bbox_norm(neg_bbox)
        pos_px = denormalize_bbox(pos_bbox_clean, width, height)
        neg_px = denormalize_bbox(neg_bbox_clean, width, height)

        if pos_px.width < min_crop_size_px or pos_px.height < min_crop_size_px:
            stats["small_pos_bbox"] += 1
            continue
        if neg_px.width < min_crop_size_px or neg_px.height < min_crop_size_px:
            stats["small_neg_bbox"] += 1
            continue

        dedup_key = (
            image_path,
            text.lower(),
            round_bbox_key(pos_bbox_clean),
            round_bbox_key(neg_bbox_clean),
        )
        if drop_duplicates and dedup_key in seen:
            stats["duplicate_sample"] += 1
            continue
        seen.add(dedup_key)

        item = dict(sample)
        item["image_path"] = image_path
        item["text"] = text
        item["pos_bbox"] = pos_bbox_clean
        item["neg_bbox"] = neg_bbox_clean
        cleaned.append(item)
        stats["kept_samples"] += 1

    stats["input_samples"] = len(samples)
    stats["removed_samples"] = len(samples) - len(cleaned)
    return cleaned, dict(stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", required=True)
    parser.add_argument("--output-json-path", required=True)
    parser.add_argument("--report-path")
    parser.add_argument("--min-crop-size-px", type=int, default=4)
    parser.add_argument("--bbox-epsilon", type=float, default=1e-3)
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()

    samples = load_json_samples(args.json_path)
    if not samples and not args.allow_empty:
        raise SystemExit(
            f"input dataset is empty: {args.json_path}. "
            "Pass --allow-empty only if you intentionally want to write an empty dataset."
        )

    cleaned, report = clean_samples(
        samples=samples,
        json_path=args.json_path,
        bbox_epsilon=args.bbox_epsilon,
        min_crop_size_px=args.min_crop_size_px,
        drop_duplicates=not args.keep_duplicates,
    )

    output_path = Path(args.output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")

    report_payload = {
        "json_path": str(args.json_path),
        "output_json_path": str(output_path),
        "min_crop_size_px": args.min_crop_size_px,
        "bbox_epsilon": args.bbox_epsilon,
        "drop_duplicates": not args.keep_duplicates,
        "allow_empty": args.allow_empty,
        "counts": report,
    }

    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
