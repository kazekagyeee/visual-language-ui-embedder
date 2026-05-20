# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import Counter

import pandas as pd


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)

    text_items = load_jsonl("data/all_pdf_rag/items.jsonl")
    ui_items = load_jsonl("data/ui_index/ui_items.jsonl")

    pdfs = Counter(x.get("pdf_name") for x in text_items)
    ui_types = Counter(x.get("ui_type") for x in ui_items)
    ui_sources = Counter(x.get("source", "base") for x in ui_items)

    stats = {
        "pdf_count": len(pdfs),
        "text_chunks": len(text_items),
        "ui_items": len(ui_items),
        "pages_with_text": len({(x.get("pdf_name"), x.get("page")) for x in text_items}),
        "pages_with_ui": len({(x.get("pdf_name"), x.get("page")) for x in ui_items}),
        "screenshots": len({x.get("screenshot_image") for x in ui_items}),
        "ui_types": dict(ui_types),
        "ui_sources": dict(ui_sources),
        "pdf_distribution": dict(pdfs),
    }

    with open(out_dir / "dataset_statistics.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    rows = []
    rows.append(["PDF documents", stats["pdf_count"]])
    rows.append(["Text chunks", stats["text_chunks"]])
    rows.append(["UI elements", stats["ui_items"]])
    rows.append(["Pages with text", stats["pages_with_text"]])
    rows.append(["Pages with UI", stats["pages_with_ui"]])
    rows.append(["Screenshots", stats["screenshots"]])

    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    df.to_csv(out_dir / "dataset_statistics.csv", index=False, encoding="utf-8-sig")

    print("=== DATASET STATISTICS ===")
    for k, v in stats.items():
        print(k, ":", v)

    print("\nSaved:")
    print("reports/dataset_statistics.json")
    print("reports/dataset_statistics.csv")


if __name__ == "__main__":
    main()
