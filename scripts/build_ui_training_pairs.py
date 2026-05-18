# -*- coding: utf-8 -*-

import argparse
import json
import random
import re
from pathlib import Path


def normalize(text):
    text = str(text).lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


TRAIN_QUERIES = [
    {
        "query": "где найти входной контроль",
        "positives": ["входной контроль"],
        "hard_negatives": ["арм входной контроль", "заявки на контроль", "выполнения входного контроля"],
    },
    {
        "query": "как создать заявку на контроль",
        "positives": ["входной контроль", "арм входной контроль", "заявки на контроль", "создать"],
        "hard_negatives": ["создать документы выполнения контроля", "создать акт входного контроля"],
    },
    {
        "query": "где найти монитор интернет поддержки",
        "positives": ["монитор интернет поддержки", "монитор интернет-поддержки"],
        "hard_negatives": ["подключение интернет поддержки пользователей", "интернет поддержка пользователей"],
    },
    {
        "query": "как создать нового контрагента",
        "positives": ["контрагенты", "создать"],
        "hard_negatives": ["организации", "партнеры", "номенклатура"],
    },
    {
        "query": "как заполнить реквизиты контрагента по инн",
        "positives": ["инн", "начните отсюда", "заполнить"],
        "hard_negatives": ["кпп", "наименование", "создать"],
    },
]


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def text_match(pattern, text):
    p = normalize(pattern)
    t = normalize(text)

    if not p or not t:
        return False

    if p == t:
        return True

    if p in t or t in p:
        return True

    p_words = set(p.split())
    t_words = set(t.split())

    if not p_words:
        return False

    overlap = len(p_words & t_words) / max(1, len(p_words))
    return overlap >= 0.65


def find_items(items, patterns):
    found = []

    for item in items:
        text = item.get("normalized_text") or normalize(item.get("text", ""))

        for pattern in patterns:
            if text_match(pattern, text):
                found.append(item)
                break

    return found


def make_pair(query, item, label):
    return {
        "query": query,
        "ui_text": item.get("text", ""),
        "ui_type": item.get("ui_type", ""),
        "pdf_name": item.get("pdf_name"),
        "page": item.get("page"),
        "screenshot_idx": item.get("screenshot_idx"),
        "screenshot_image": item.get("screenshot_image"),
        "bbox": item.get("bbox"),
        "label": int(label),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ui-index-dir", default="data/ui_index")
    parser.add_argument("--out", default="data/ui_training_pairs.jsonl")
    parser.add_argument("--random-negatives", type=int, default=4)
    args = parser.parse_args()

    items = load_jsonl(Path(args.ui_index_dir) / "ui_items.jsonl")

    pairs = []

    for spec in TRAIN_QUERIES:
        query = spec["query"]

        positives = find_items(items, spec["positives"])
        hard_negatives = find_items(items, spec["hard_negatives"])

        for item in positives:
            pairs.append(make_pair(query, item, 1))

        for item in hard_negatives:
            pairs.append(make_pair(query, item, 0))

        random_pool = [
            item for item in items
            if item not in positives and item not in hard_negatives
        ]

        for item in random.sample(random_pool, min(args.random_negatives, len(random_pool))):
            pairs.append(make_pair(query, item, 0))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"[OK] pairs: {len(pairs)}")
    print(f"[OK] saved: {out}")


if __name__ == "__main__":
    main()
