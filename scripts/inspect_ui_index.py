# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default="data/ui_index")
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    path = Path(args.index_dir) / "ui_items.jsonl"

    if not path.exists():
        print(f"Не найден файл: {path}")
        return

    items = []

    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    print(f"UI items: {len(items)}")

    print("\nPDF:")
    for k, v in Counter(x.get("pdf_name") for x in items).most_common():
        print(f"  {k}: {v}")

    print("\nUI types:")
    for k, v in Counter(x.get("ui_type") for x in items).most_common():
        print(f"  {k}: {v}")

    print("\nExamples:")
    for item in items[:args.limit]:
        print(
            f"[{item.get('pdf_name')} p.{item.get('page')} s.{item.get('screenshot_idx')}] "
            f"{item.get('ui_type')} | {item.get('text')}"
        )


if __name__ == "__main__":
    main()
