# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

from rag.ocr_cleaning import normalize_ocr_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="reports/ui_retrieval_metrics.json")
    parser.add_argument("--out", default="reports/error_analysis.md")
    args = parser.parse_args()

    with open(args.metrics, "r", encoding="utf-8") as f:
        report = json.load(f)

    rows = report["rows"]

    errors = [r for r in rows if not r.get("top1")]

    lines = []
    lines.append("# Error Analysis")
    lines.append("")
    lines.append(f"Total queries: {len(rows)}")
    lines.append(f"Top-1 errors: {len(errors)}")
    lines.append("")

    for row in errors:
        query = row["query"]
        target = row["target_text"]
        best = row["best_prediction"]
        page = row["best_page"]

        lines.append("## Ошибка")
        lines.append("")
        lines.append(f"- Query: `{query}`")
        lines.append(f"- Target: `{target}`")
        lines.append(f"- Best prediction: `{best}`")
        lines.append(f"- Best page: `{page}`")
        lines.append("")
        lines.append("Возможная причина:")
        lines.append("")
        lines.append("- OCR-ошибка;")
        lines.append("- похожий UI-элемент;")
        lines.append("- элемент встречается на нескольких страницах;")
        lines.append("- недостаточно hard negatives;")
        lines.append("- bbox элемента слишком маленький или обрезан.")
        lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved error analysis: {out}")


if __name__ == "__main__":
    main()
