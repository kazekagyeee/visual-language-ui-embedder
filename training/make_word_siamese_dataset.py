# -*- coding: utf-8 -*-

import argparse
import json
import random
import re
from pathlib import Path

from PIL import Image


def normalize_text(text: str) -> str:
    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_items(rag_dir: Path):
    items = []

    with open(rag_dir / "items.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("page_words"):
                items.append(item)

    return items


def merge_phrase_words(words, phrase_tokens):
    normalized_words = [normalize_text(w["text"]) for w in words]
    phrase = [normalize_text(t) for t in phrase_tokens]

    phrase = [p for p in phrase if p]
    if not phrase:
        return []

    matches = []

    for i in range(len(normalized_words) - len(phrase) + 1):
        window = normalized_words[i:i + len(phrase)]

        ok = True
        for a, b in zip(window, phrase):
            if a != b:
                ok = False
                break

        if ok:
            selected = words[i:i + len(phrase)]
            x0 = min(w["bbox_px"][0] for w in selected)
            y0 = min(w["bbox_px"][1] for w in selected)
            x1 = max(w["bbox_px"][2] for w in selected)
            y1 = max(w["bbox_px"][3] for w in selected)

            matches.append([x0, y0, x1, y1])

    return matches


def crop_box(page_image_path, bbox, out_path, pad=8):
    img = Image.open(page_image_path).convert("RGB")
    w, h = img.size

    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(w, x1 + pad)
    y1 = min(h, y1 + pad)

    crop = img.crop((x0, y0, x1, y1))

    if crop.width < 8 or crop.height < 8:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path)
    return True


def extract_candidates_from_page_words(item, allowed_phrases=None):
    words = item.get("page_words", [])

    candidates = []

    if allowed_phrases:
        for phrase in allowed_phrases:
            tokens = phrase.split()
            boxes = merge_phrase_words(words, tokens)

            for box in boxes:
                candidates.append({
                    "text": phrase,
                    "bbox_px": box,
                    "page": item["page"],
                    "source_item_id": item["id"],
                })

    else:
        for word in words:
            text = normalize_text(word["text"])

            if len(text) < 3:
                continue

            candidates.append({
                "text": word["text"],
                "bbox_px": word["bbox_px"],
                "page": item["page"],
                "source_item_id": item["id"],
            })

    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-dir", default="data/pdf_rag")
    parser.add_argument("--out", default="data/word_siamese_pairs.jsonl")
    parser.add_argument("--crops-dir", default="data/word_crops")
    parser.add_argument("--negatives-per-positive", type=int, default=4)
    parser.add_argument(
        "--phrases",
        nargs="*",
        default=[
            "ГОСТы",
            "Показатели контроля",
            "Виды контроля",
            "Входной контроль",
            "Заявки на контроль",
            "Выполнения входного контроля",
            "Акты входного контроля",
            "Создать",
            "Записать",
            "Записать и закрыть",
            "Добавить",
        ],
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    rag_dir = Path(args.rag_dir)
    crops_dir = Path(args.crops_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = load_items(rag_dir)

    positives = []
    crop_id = 0

    for item in items:
        page_image = item.get("page_image")
        if not page_image or not Path(page_image).exists():
            continue

        candidates = extract_candidates_from_page_words(
            item,
            allowed_phrases=args.phrases,
        )

        for cand in candidates:
            crop_path = crops_dir / f"page_{cand['page']:04d}_word_{crop_id:06d}.png"

            ok = crop_box(
                page_image_path=page_image,
                bbox=cand["bbox_px"],
                out_path=crop_path,
            )

            if not ok:
                continue

            positives.append({
                "text": cand["text"],
                "image": str(crop_path).replace("\\", "/"),
                "bbox_px": cand["bbox_px"],
                "page": cand["page"],
                "label": 1,
                "source_item_id": cand["source_item_id"],
            })

            crop_id += 1

    pairs = []

    for pos in positives:
        pairs.append(pos)

        for _ in range(args.negatives_per_positive):
            neg = random.choice(positives)

            tries = 0
            while normalize_text(neg["text"]) == normalize_text(pos["text"]) and tries < 30:
                neg = random.choice(positives)
                tries += 1

            pairs.append({
                "text": pos["text"],
                "image": neg["image"],
                "bbox_px": neg["bbox_px"],
                "page": neg["page"],
                "label": 0,
                "source_item_id": pos["source_item_id"],
                "negative_text": neg["text"],
            })

    random.shuffle(pairs)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Positive UI word crops: {len(positives)}")
    print(f"Saved pairs: {out_path}")
    print(f"Pairs: {len(pairs)}")


if __name__ == "__main__":
    main()
