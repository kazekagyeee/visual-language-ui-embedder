# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path

import fitz
import cv2
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer


def normalize(text):
    text = str(text).lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text):
    text = str(text).replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_jsonl(path):
    if not Path(path).exists():
        return []

    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def append_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_reader(gpu=False):
    import easyocr
    return easyocr.Reader(["ru", "en"], gpu=gpu)


def get_image_blocks(page, max_blocks=3):
    info = page.get_text("dict")
    blocks = []

    for block in info.get("blocks", []):
        if block.get("type") != 1:
            continue

        x0, y0, x1, y1 = block["bbox"]
        rect = fitz.Rect(x0, y0, x1, y1)

        if rect.width < 120 or rect.height < 80:
            continue

        if rect.width > 1400 or rect.height > 1400:
            continue

        blocks.append(rect)

    blocks.sort(key=lambda r: (r.y0, r.x0))
    return blocks[:max_blocks]


def render_crop(page, rect, scale=2):
    pix = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        clip=rect,
        alpha=False,
    )

    return Image.frombytes(
        "RGB",
        [pix.width, pix.height],
        pix.samples,
    )


def bbox_from_easyocr(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    return [
        int(min(xs)),
        int(min(ys)),
        int(max(xs)),
        int(max(ys)),
    ]


def area(bbox):
    return max(1, bbox[2] - bbox[0]) * max(1, bbox[3] - bbox[1])


def guess_ui_type(text, bbox):
    t = normalize(text)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    if any(x in t for x in [
        "создать", "добавить", "ок", "далее",
        "подключить", "записать", "сохранить",
        "выбрать", "открыть"
    ]):
        return "button"

    if any(x in t for x in [
        "монитор", "ссылка", "интернет-поддержка",
        "заявки", "контроль", "арм"
    ]):
        return "menu_item"

    if w > 120 and h < 60:
        return "menu_item"

    return "text"


def is_bad_text(text):
    t = normalize(text)

    if len(t) < 3:
        return True

    if t in {
        "и", "в", "на", "по", "из", "как", "что",
        "для", "или", "стр", "рис", "ок"
    }:
        return True

    if t.isdigit():
        return True

    if len(t.split()) > 12:
        return True

    return False


def make_item_id(pdf_name, page_num, screenshot_idx, item_idx):
    stem = Path(pdf_name).stem
    return f"{stem}_p{page_num:04d}_s{screenshot_idx:02d}_e{item_idx:04d}"


def build_embeddings(out_dir, model_name):
    out_dir = Path(out_dir)
    items_path = out_dir / "ui_items.jsonl"
    embeddings_path = out_dir / "ui_embeddings.npy"

    items = load_jsonl(items_path)

    if not items:
        print("[WARN] Нет UI items для embeddings")
        return

    model = SentenceTransformer(model_name)

    texts = [
        f"{item['text']} {item.get('ui_type', '')} страница {item.get('page')}"
        for item in items
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    np.save(embeddings_path, np.asarray(embeddings, dtype=np.float32))
    print(f"[OK] embeddings saved: {embeddings_path}")


def build_index(args):
    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)

    screenshots_dir = out_dir / "screenshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    items_path = out_dir / "ui_items.jsonl"

    existing_items = load_jsonl(items_path) if args.resume else []
    processed_keys = {
        (
            item.get("pdf_name"),
            int(item.get("page", -1)),
            int(item.get("screenshot_idx", -1)),
        )
        for item in existing_items
    }

    pdfs = sorted(pdf_dir.glob("*.pdf"))

    if args.pdf_name:
        pdfs = [p for p in pdfs if args.pdf_name.lower() in p.name.lower()]

    if not pdfs:
        raise FileNotFoundError(f"PDF не найдены в {pdf_dir}")

    reader = get_reader(gpu=args.gpu)

    total_new_items = 0

    for pdf_path in pdfs:
        print(f"\n[PDF] {pdf_path.name}")

        doc = fitz.open(pdf_path)

        start_page = max(1, args.start_page)
        end_page = len(doc)

        if args.max_pages:
            end_page = min(end_page, start_page + args.max_pages - 1)

        for page_num in range(start_page, end_page + 1):
            page = doc[page_num - 1]

            blocks = get_image_blocks(
                page,
                max_blocks=args.max_blocks_per_page,
            )

            if not blocks:
                continue

            print(f"[PAGE] {page_num}: image blocks={len(blocks)}")

            for screenshot_idx, rect in enumerate(blocks, start=1):
                key = (pdf_path.name, page_num, screenshot_idx)

                if key in processed_keys:
                    print(f"  skip p{page_num} s{screenshot_idx} already indexed")
                    continue

                img = render_crop(
                    page,
                    rect,
                    scale=args.scale,
                )

                screenshot_name = f"{pdf_path.stem}_p{page_num:04d}_s{screenshot_idx:02d}.png"
                screenshot_path = screenshots_dir / screenshot_name
                img.save(screenshot_path)

                arr = np.array(img)
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

                try:
                    ocr = reader.readtext(arr)
                except Exception as exc:
                    print(f"  OCR error p{page_num} s{screenshot_idx}: {exc}")
                    continue

                new_rows = []
                item_idx = 0

                for points, text, conf in ocr:
                    text = clean_text(text)

                    if conf < args.min_conf:
                        continue

                    if is_bad_text(text):
                        continue

                    bbox = bbox_from_easyocr(points)

                    if area(bbox) < args.min_area:
                        continue

                    item_idx += 1

                    ui_type = guess_ui_type(text, bbox)

                    new_rows.append(
                        {
                            "id": make_item_id(pdf_path.name, page_num, screenshot_idx, item_idx),
                            "pdf_name": pdf_path.name,
                            "page": page_num,
                            "screenshot_idx": screenshot_idx,
                            "screenshot_image": str(screenshot_path).replace("\\", "/"),
                            "text": text,
                            "normalized_text": normalize(text),
                            "bbox": bbox,
                            "ui_type": ui_type,
                            "confidence": float(conf),
                        }
                    )

                if new_rows:
                    append_jsonl(items_path, new_rows)
                    total_new_items += len(new_rows)

                print(f"  p{page_num} s{screenshot_idx}: items={len(new_rows)}")

    meta = {
        "pdf_dir": str(pdf_dir),
        "out_dir": str(out_dir),
        "new_items": total_new_items,
        "resume": args.resume,
        "scale": args.scale,
        "max_blocks_per_page": args.max_blocks_per_page,
        "min_conf": args.min_conf,
    }

    with open(out_dir / "ui_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] new UI items: {total_new_items}")

    if args.no_embeddings:
        print("[SKIP] embeddings disabled")
    else:
        build_embeddings(out_dir, args.model)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pdf-dir", default="data_source")
    parser.add_argument("--out-dir", default="data/ui_index")

    parser.add_argument("--pdf-name", default=None)
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-blocks-per-page", type=int, default=3)

    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--min-conf", type=float, default=0.20)
    parser.add_argument("--min-area", type=int, default=40)

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--no-embeddings", action="store_true")

    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )

    args = parser.parse_args()
    build_index(args)


if __name__ == "__main__":
    main()
