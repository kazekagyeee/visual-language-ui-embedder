# -*- coding: utf-8 -*-

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from rag.ocr_cleaning import normalize_ocr_text


TEMPLATES_SINGLE = [
    "где находится {x}",
    "покажи {x}",
    "где кнопка {x}",
    "где ссылка {x}",
    "найди элемент {x}",
]

TEMPLATES_MULTI = [
    "где находятся {a} и {b}",
    "покажи {a} и {b}",
    "найди элементы {a} и {b}",
]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def clean_display_text(text):
    text = str(text).strip()
    text = text.strip("[]{}()«»\"'")
    text = text.replace(":", "").strip()
    return text


def is_good_target(item):
    text = clean_display_text(item.get("text", ""))
    norm = normalize_ocr_text(text)

    if len(norm) < 3:
        return False

    if len(norm.split()) > 4:
        return False

    bad = {
        "главное",
        "см также",
        "поиск",
        "ctrl f",
        "страница",
        "рис",
        "рисунок",
        "таблица",
    }

    if norm in bad:
        return False

    if item.get("ui_type") not in {"button", "hyperlink", "sidebar_item", "tab", "input"}:
        return False

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-dir", default="data/pdf_rag")
    parser.add_argument("--out", default="data/test_queries.json")
    parser.add_argument("--max-single", type=int, default=20)
    parser.add_argument("--max-multi", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    ui_path = Path(args.rag_dir) / "ui_elements.jsonl"
    items = [x for x in load_jsonl(ui_path) if is_good_target(x)]

    by_norm = defaultdict(list)

    for item in items:
        text = clean_display_text(item["text"])
        norm = normalize_ocr_text(text)
        by_norm[norm].append(item)

    canonical = []

    for norm, group in by_norm.items():
        group.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)

        best = group[0]
        pages = sorted({int(x["page"]) for x in group})

        canonical.append({
            "text": clean_display_text(best["text"]),
            "norm": norm,
            "pages": pages,
            "ui_type": best.get("ui_type", "unknown"),
            "count": len(group),
        })

    important_order = [
        "госты",
        "показатели контроля",
        "виды контроля",
        "группы прочности",
        "входной контроль",
        "заявки на контроль",
        "выполнения входного контроля",
        "акты входного контроля",
        "добавить",
        "записать",
        "записать и закрыть",
    ]

    canonical.sort(
        key=lambda x: (
            0 if x["norm"] in important_order else 1,
            -x["count"],
            x["norm"],
        )
    )

    queries = []

    for item in canonical[:args.max_single]:
        q = random.choice(TEMPLATES_SINGLE).format(x=item["text"])

        queries.append({
            "query": q,
            "targets": [
                {
                    "text": item["text"],
                    "normalized_text": item["norm"],
                    "target_pages": item["pages"],
                    "ui_type": item["ui_type"],
                }
            ],
            "type": "single",
        })

    by_page = defaultdict(list)

    for item in canonical:
        for page in item["pages"]:
            by_page[page].append(item)

    multi_added = 0

    for page, page_items in sorted(by_page.items()):
        unique = {}
        for item in page_items:
            unique[item["norm"]] = item

        page_items = list(unique.values())[:8]

        if len(page_items) < 2:
            continue

        for i in range(len(page_items)):
            for j in range(i + 1, len(page_items)):
                a = page_items[i]
                b = page_items[j]

                q = random.choice(TEMPLATES_MULTI).format(a=a["text"], b=b["text"])

                queries.append({
                    "query": q,
                    "targets": [
                        {
                            "text": a["text"],
                            "normalized_text": a["norm"],
                            "target_pages": a["pages"],
                            "ui_type": a["ui_type"],
                        },
                        {
                            "text": b["text"],
                            "normalized_text": b["norm"],
                            "target_pages": b["pages"],
                            "ui_type": b["ui_type"],
                        },
                    ],
                    "type": "multi",
                })

                multi_added += 1

                if multi_added >= args.max_multi:
                    break

            if multi_added >= args.max_multi:
                break

        if multi_added >= args.max_multi:
            break

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8-sig") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"Saved test queries: {out_path}")
    print(f"Queries: {len(queries)}")
    print(f"Single: {sum(1 for q in queries if q['type'] == 'single')}")
    print(f"Multi: {sum(1 for q in queries if q['type'] == 'multi')}")


if __name__ == "__main__":
    main()
