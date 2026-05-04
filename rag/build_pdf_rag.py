# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import fitz
import numpy as np
from sentence_transformers import SentenceTransformer


def fix_text(text: str) -> str:
    text = text.strip()
    if "Р" in text or "С" in text:
        try:
            return text.encode("latin1").decode("utf-8").strip()
        except Exception:
            return text
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default="data_source/instruction.pdf")
    parser.add_argument("--out-dir", default="data/pdf_rag")
    parser.add_argument("--max-pages", type=int, default=120)
    parser.add_argument("--zoom", type=float, default=2.0)
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    pages_dir = out_dir / "pages"
    crops_dir = out_dir / "crops"
    pages_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(args.pdf)
    model = SentenceTransformer(args.model)

    items = []
    matrix = fitz.Matrix(args.zoom, args.zoom)

    for page_idx in range(min(args.max_pages, len(doc))):
        page = doc[page_idx]
        page_number = page_idx + 1

        page_image_path = pages_dir / f"page_{page_number:04d}.png"
        page_pix = page.get_pixmap(matrix=matrix, alpha=False)
        page_pix.save(str(page_image_path))

        blocks = page.get_text("blocks")
        block_id = 0

        for block in blocks:
            x0, y0, x1, y1, text, *_ = block
            text = fix_text(text)

            if len(text) < 20:
                continue

            rect = fitz.Rect(x0, y0, x1, y1)
            crop_pix = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)

            crop_path = crops_dir / f"page_{page_number:04d}_block_{block_id:03d}.png"
            crop_pix.save(str(crop_path))

            items.append({
                "id": f"page_{page_number}_block_{block_id}",
                "page": page_number,
                "block_id": block_id,
                "text": text,
                "bbox": [x0, y0, x1, y1],
                "page_image": str(page_image_path).replace("\\", "/"),
                "crop_image": str(crop_path).replace("\\", "/"),
            })

            block_id += 1

    texts = [item["text"] for item in items]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    with open(out_dir / "items.jsonl", "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    np.save(out_dir / "embeddings.npy", np.array(embeddings, dtype=np.float32))

    print(f"Saved items: {len(items)}")
    print(f"Saved metadata: {out_dir / 'items.jsonl'}")
    print(f"Saved embeddings: {out_dir / 'embeddings.npy'}")
    print(f"Saved crops: {crops_dir}")


if __name__ == "__main__":
    main()
