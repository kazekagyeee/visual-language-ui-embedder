from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.audit import audit_samples, load_json_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", required=True)
    parser.add_argument("--min-crop-size-px", type=int, default=4)
    parser.add_argument("--bbox-epsilon", type=float, default=1e-3)
    args = parser.parse_args()

    samples = load_json_samples(args.json_path)
    report = audit_samples(
        samples,
        json_path=args.json_path,
        bbox_epsilon=args.bbox_epsilon,
        min_crop_size_px=args.min_crop_size_px,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
