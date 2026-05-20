# -*- coding: utf-8 -*-

import argparse
import io
import json
import re
from pathlib import Path

import fitz
import cv2
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from rag.ocr_cleanup import cleanup_ocr_text


def normalize(text):
    text = cleanup_ocr_text(text)
    text = str(text).lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_jsonl(path):
    rows = []
    if not Path(path).exists():
        return rows

    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def bbox_from_points(points, scale=1.0):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    return [
        int(min(xs) / scale),
        int(min(ys) / scale),
        int(max(xs) / scale),
        int(max(ys) / scale),
    ]


def make_variants(img):
    rgb = np.array(img.convert("RGB"))
    variants = []

    variants.append(("native", rgb, 1.0))

    big2 = cv2.resize(rgb, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    variants.append(("big2", big2, 2.0))

    big3 = cv2.resize(rgb, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    variants.append(("big3", big3, 3.0))

    gray = cv2.cvtColor(big3, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    variants.append(("gray3", cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), 3.0))

    return variants


def useful_text(text):
    n = normalize(text)

    if len(n) < 2:
        return False

    bad = {"и", "в", "на", "по", "для", "как", "что", "или"}
    if n in bad:
        return False

    if len(n.split()) > 12:
        return False

    return True


def guess_ui_type(text):
    n = normalize(text)

    if any(x in n for x in ["создать", "заполн", "сохран", "далее", "ок", "выбрать"]):
        return "button"

    if any(x in n for x in ["монитор", "интернет", "контрагент", "инн", "начните", "отсюда"]):
        return "small_text"

    return "native_ocr"


def rebuild_embeddings(index_dir, model_name):
    items = load_jsonl(index_dir / "ui_items.jsonl")
    model = SentenceTransformer(model_name)

    texts = [
        f"{item.get('text', '')} {item.get('ui_type', '')} страница {item.get('page', '')}"
        for item in items
    ]

    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    np.save(index_dir / "ui_embeddings.npy", np.asarray(emb, dtype=np.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", default="data_source")
    parser.add_argument("--index-dir", default="data/ui_index")
    parser.add_argument("--pdf-name", default=None)
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=None)
    parser.add_argument("--min-conf", type=float, default=0.05)
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    args = parser.parse_args()

    import easyocr
    reader = easyocr.Reader(["ru", "en"], gpu=False)

    pdf_dir = Path(args.pdf_dir)
    index_dir = Path(args.index_dir)
    items_path = index_dir / "ui_items.jsonl"
    native_dir = index_dir / "native_screenshots"

    native_dir.mkdir(parents=True, exist_ok=True)

    items = load_jsonl(items_path)

    existing = {
        (
            x.get("pdf_name"),
            x.get("page"),
            x.get("screenshot_idx"),
            normalize(x.get("text")),
            tuple(x.get("bbox") or []),
        )
        for x in items
    }

    pdfs = sorted(pdf_dir.glob("*.pdf"))

    if args.pdf_name:
        pdfs = [p for p in pdfs if p.name == args.pdf_name]

    new_items = []

    for pdf_path in pdfs:
        print(f"[PDF] {pdf_path.name}")
        doc = fitz.open(pdf_path)

        end_page = args.end_page or len(doc)
        end_page = min(end_page, len(doc))

        for page_num in range(args.start_page, end_page + 1):
            page = doc[page_num - 1]
            images = page.get_images(full=True)

            if not images:
                continue

            print(f"  page {page_num}: native images={len(images)}")

            for image_idx, image_info in enumerate(images, start=1):
                xref = image_info[0]

                try:
                    extracted = doc.extract_image(xref)
                    image_bytes = extracted["image"]
                    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                except Exception:
                    continue

                if img.width < 120 or img.height < 70:
                    continue

                screenshot_name = f"{pdf_path.stem}_p{page_num:04d}_native_{image_idx:02d}.png"
                screenshot_path = native_dir / screenshot_name
                img.save(screenshot_path)

                local_i = 0

                for variant_name, arr, scale in make_variants(img):
                    try:
                        ocr = reader.readtext(arr, detail=1, paragraph=False)
                    except Exception:
                        continue

                    for points, text, conf in ocr:
                        text = cleanup_ocr_text(text)

                        if conf < args.min_conf:
                            continue

                        if not useful_text(text):
                            continue

                        bbox = bbox_from_points(points, scale=scale)

                        local_i += 1

                        row = {
                            "id": f"{pdf_path.name}_p{page_num}_native_{image_idx}_{variant_name}_{local_i}",
                            "pdf_name": pdf_path.name,
                            "page": page_num,
                            "screenshot_idx": 1000 + image_idx,
                            "screenshot_image": str(screenshot_path).replace("\\", "/"),
                            "text": text,
                            "normalized_text": normalize(text),
                            "bbox": bbox,
                            "ui_type": guess_ui_type(text),
                            "confidence": float(conf),
                            "source": "native_pdf_image_ocr",
                        }

                        key = (
                            row.get("pdf_name"),
                            row.get("page"),
                            row.get("screenshot_idx"),
                            normalize(row.get("text")),
                            tuple(row.get("bbox") or []),
                        )

                        if key in existing:
                            continue

                        existing.add(key)
                        new_items.append(row)

    print(f"[OK] new native OCR items: {len(new_items)}")

    all_items = items + new_items
    save_jsonl(items_path, all_items)
    rebuild_embeddings(index_dir, args.model)

    print("[DONE]")


if __name__ == "__main__":
    main()
