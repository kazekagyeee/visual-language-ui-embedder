# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import fitz


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = text.replace("\u00ad", "")
    text = " ".join(text.split())
    return text.strip()


def extract_blocks(page, page_num: int, page_image_path: str):
    blocks = []
    raw_blocks = page.get_text("blocks")

    block_id = 0

    for b in raw_blocks:
        x0, y0, x1, y1, text, *_ = b
        text = clean_text(text)

        if not text:
            continue

        blocks.append({
            "id": f"page_{page_num}_block_{block_id}",
            "page": page_num,
            "block": block_id,
            "text": text,
            "bbox": [float(x0), float(y0), float(x1), float(y1)],
            "page_image": page_image_path,
        })

        block_id += 1

    return blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to PDF")
    parser.add_argument("--out", required=True, help="Output RAG directory")
    parser.add_argument("--start-page", type=int, default=1, help="1-based first page")
    parser.add_argument("--max-pages", type=int, default=None, help="How many pages to process")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    out_dir = Path(args.out)
    pages_dir = out_dir / "pages"

    ensure_dir(out_dir)
    ensure_dir(pages_dir)

    doc = fitz.open(pdf_path)

    total_pages = len(doc)
    start_page = max(1, args.start_page)

    if args.max_pages is None:
        end_page = total_pages
    else:
        end_page = min(total_pages, start_page + args.max_pages - 1)

    if start_page > total_pages:
        raise ValueError(f"start-page={start_page} больше количества страниц PDF: {total_pages}")

    text_blocks = []

    zoom = args.dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    print("=" * 80)
    print("BUILD PDF RAG")
    print("=" * 80)
    print(f"PDF: {pdf_path}")
    print(f"OUT: {out_dir}")
    print(f"Pages: {start_page}..{end_page} / {total_pages}")
    print(f"DPI: {args.dpi}")

    for page_num in range(start_page, end_page + 1):
        page = doc[page_num - 1]

        image_name = f"page_{page_num:04d}.png"
        image_path = pages_dir / image_name
        rel_image_path = str(image_path).replace("\\", "/")

        if args.force or not image_path.exists():
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pix.save(image_path)

        blocks = extract_blocks(page, page_num, rel_image_path)
        text_blocks.extend(blocks)

        print(f"page {page_num}: blocks={len(blocks)} image={image_path}")

    # Основной формат для текстового поиска
    with open(out_dir / "text_blocks.jsonl", "w", encoding="utf-8") as f:
        for row in text_blocks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Дублируем под разные старые имена, чтобы не ломать уже написанные модули
    for duplicate_name in ["blocks.jsonl", "chunks.jsonl", "pdf_blocks.jsonl"]:
        with open(out_dir / duplicate_name, "w", encoding="utf-8") as f:
            for row in text_blocks:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "pdf": str(pdf_path).replace("\\", "/"),
        "out_dir": str(out_dir).replace("\\", "/"),
        "pages_dir": str(pages_dir).replace("\\", "/"),
        "total_pdf_pages": total_pages,
        "start_page": start_page,
        "end_page": end_page,
        "processed_pages": end_page - start_page + 1,
        "dpi": args.dpi,
        "text_blocks": len(text_blocks),
        "files": {
            "text_blocks": "text_blocks.jsonl",
            "blocks": "blocks.jsonl",
            "chunks": "chunks.jsonl",
            "pages": "pages/page_XXXX.png"
        }
    }

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"Saved RAG: {out_dir}")
    print(f"Pages saved: {pages_dir}")
    print(f"Text blocks: {len(text_blocks)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
