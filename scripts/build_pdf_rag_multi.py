# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path

import fitz
import numpy as np
from sentence_transformers import SentenceTransformer


def clean_text(text):
    text = str(text).replace("\x00", " ")
    text = text.replace("-\n", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_chunks(text, max_chars=900, overlap=120):
    text = clean_text(text)

    if len(text) <= max_chars:
        return [text] if text else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]

        last_dot = max(
            chunk.rfind("."),
            chunk.rfind("?"),
            chunk.rfind("!"),
            chunk.rfind(";"),
        )

        if last_dot > 250:
            chunk = chunk[:last_dot + 1]

        chunk = clean_text(chunk)

        if chunk:
            chunks.append(chunk)

        start = start + max_chars - overlap

    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", default="data_source")
    parser.add_argument("--out-dir", default="data/all_pdf_rag")
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)
    pages_dir = out_dir / "pages"

    out_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.glob("*.pdf"))

    if not pdfs:
        raise FileNotFoundError(f"В папке {pdf_dir} нет PDF-файлов")

    items = []

    for pdf_path in pdfs:
        print(f"[PDF] {pdf_path.name}")

        doc = fitz.open(pdf_path)
        doc_id = pdf_path.stem

        for page_index in range(len(doc)):
            page = doc[page_index]
            page_num = page_index + 1

            text = clean_text(page.get_text("text"))

            if not text:
                continue

            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            page_image = pages_dir / f"{doc_id}_page_{page_num:04d}.png"
            pix.save(page_image)

            chunks = split_chunks(text)

            for chunk_id, chunk in enumerate(chunks):
                items.append(
                    {
                        "id": f"{doc_id}_{page_num}_{chunk_id}",
                        "doc_id": doc_id,
                        "pdf_name": pdf_path.name,
                        "page": page_num,
                        "chunk_id": chunk_id,
                        "text": chunk,
                        "page_image": str(page_image).replace("\\", "/"),
                    }
                )

    items_path = out_dir / "items.jsonl"

    with open(items_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[OK] text chunks: {len(items)}")

    model = SentenceTransformer(args.model)
    texts = [item["text"] for item in items]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    np.save(out_dir / "embeddings.npy", np.asarray(embeddings, dtype=np.float32))

    meta = {
        "pdf_dir": str(pdf_dir),
        "pdf_count": len(pdfs),
        "pdfs": [p.name for p in pdfs],
        "chunks": len(items),
    }

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[DONE]")
    print(f"PDF files: {len(pdfs)}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
