# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.domain_1c_dictionary import repair_ui_item_text, normalize_1c_text


def read_jsonl(path: Path) -> list[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def find_items_file(index_dir: Path) -> Path:
    for name in ["ui_items.jsonl", "items.jsonl"]:
        p = index_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Не найден ui_items.jsonl/items.jsonl в {index_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", default="data/ui_index")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    out_dir = Path(args.out_dir) if args.out_dir else index_dir

    src_items = find_items_file(index_dir)
    dst_items = out_dir / src_items.name

    items = read_jsonl(src_items)

    changed = 0
    for item in items:
        raw = item.get("raw_text") or item.get("text") or item.get("label") or ""
        repaired = repair_ui_item_text(raw)
        normalized = normalize_1c_text(repaired)

        item["raw_text"] = raw
        item["text"] = repaired
        item["normalized_text"] = normalized
        item["domain_repaired"] = repaired != raw

        if repaired != raw:
            changed += 1

    write_jsonl(dst_items, items)

    print("[OK] domain dictionary applied")
    print("items:", len(items))
    print("changed:", changed)
    print("saved:", dst_items)


if __name__ == "__main__":
    main()