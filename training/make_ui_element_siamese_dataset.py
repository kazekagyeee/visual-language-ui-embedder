# -*- coding: utf-8 -*-

import argparse
import json
import random
from pathlib import Path

from rag.ocr_cleaning import normalize_ocr_text


def load_elements(path, max_elements=None):
    elements = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            el = json.loads(line)

            if not Path(el["crop_image"]).exists():
                continue

            clean_text = normalize_ocr_text(el.get("text", ""))

            if len(clean_text) < 2:
                continue

            el["clean_text"] = clean_text
            elements.append(el)

    # убираем точные дубли
    unique = []
    seen = set()

    for el in elements:
        key = (
            el["page"],
            el["clean_text"],
            tuple(el["bbox"]),
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(el)

    if max_elements:
        random.shuffle(unique)
        unique = unique[:max_elements]

    return unique


def token_overlap(a, b):
    a_tokens = set(str(a).split())
    b_tokens = set(str(b).split())

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / max(len(a_tokens), len(b_tokens))


def choose_hard_negative(pos, elements):
    scored = []

    for el in elements:
        if el["id"] == pos["id"]:
            continue

        if el["clean_text"] == pos["clean_text"]:
            continue

        score = token_overlap(pos["clean_text"], el["clean_text"])

        if el["page"] == pos["page"]:
            score += 0.4

        if el.get("ui_type") == pos.get("ui_type"):
            score += 0.2

        scored.append((score, el))

    scored.sort(key=lambda x: x[0], reverse=True)

    if scored:
        return scored[0][1]

    return random.choice(elements)


def choose_random_negative(pos, elements):
    neg = random.choice(elements)

    tries = 0
    while neg["clean_text"] == pos["clean_text"] and tries < 50:
        neg = random.choice(elements)
        tries += 1

    return neg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-dir", default="data/pdf_rag")
    parser.add_argument("--out", default="data/ui_element_pairs.jsonl")
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--hard-negatives-ratio", type=float, default=0.5)
    parser.add_argument("--max-elements", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    rag_dir = Path(args.rag_dir)
    elements_path = rag_dir / "ui_elements.jsonl"

    elements = load_elements(elements_path, max_elements=args.max_elements)

    if len(elements) < 2:
        raise RuntimeError("Too few UI elements. Run rag.build_ui_elements first.")

    pairs = []

    for el in elements:
        pairs.append({
            "text": el["clean_text"],
            "raw_text": el["text"],
            "image": el["crop_image"],
            "label": 1,
            "page": el["page"],
            "bbox_px": el["bbox"],
            "ui_type": el.get("ui_type", "unknown"),
            "ui_element_id": el["id"],
            "negative_type": None,
        })

        for _ in range(args.negatives_per_positive):
            if random.random() < args.hard_negatives_ratio:
                neg = choose_hard_negative(el, elements)
                negative_type = "hard"
            else:
                neg = choose_random_negative(el, elements)
                negative_type = "random"

            pairs.append({
                "text": el["clean_text"],
                "raw_text": el["text"],
                "image": neg["crop_image"],
                "label": 0,
                "page": neg["page"],
                "bbox_px": neg["bbox"],
                "ui_type": neg.get("ui_type", "unknown"),
                "ui_element_id": el["id"],
                "negative_text": neg["clean_text"],
                "negative_type": negative_type,
            })

    random.shuffle(pairs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=" * 80)
    print(f"UI elements used: {len(elements)}")
    print(f"Pairs: {len(pairs)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
