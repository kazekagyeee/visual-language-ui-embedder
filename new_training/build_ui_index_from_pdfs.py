from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import fitz
import numpy as np
from PIL import Image
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parent
PDF_DIR = ROOT_DIR / "pdf"
OUT_DIR = ROOT_DIR / "generated" / "ui_index"


def normalize(text: str) -> str:
    text = str(text).lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    text = str(text).replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]], mode: str = "w") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_reader(gpu: bool = False) -> Any:
    try:
        import easyocr
    except ImportError as exc:
        raise RuntimeError(
            "EasyOCR is required for OCR. Install training dependencies:\n"
            r".\.venv\Scripts\python.exe -m pip install -r new_training\requirements.txt"
        ) from exc

    return easyocr.Reader(["ru", "en"], gpu=gpu)


def resolve_ocr_gpu(args: argparse.Namespace) -> bool:
    if args.ocr_device == "cpu":
        return False
    if args.ocr_device == "cuda" or args.gpu:
        return True

    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def get_image_blocks(page: fitz.Page, max_blocks: int) -> list[fitz.Rect]:
    info = page.get_text("dict")
    blocks: list[fitz.Rect] = []

    for block in info.get("blocks", []):
        if block.get("type") != 1:
            continue

        x0, y0, x1, y1 = block["bbox"]
        rect = fitz.Rect(x0, y0, x1, y1)

        if rect.width < 120 or rect.height < 80:
            continue
        if rect.width > 1800 or rect.height > 1800:
            continue

        blocks.append(rect)

    blocks.sort(key=lambda rect: (rect.y0, rect.x0))
    return blocks[:max_blocks]


def render_crop(page: fitz.Page, rect: fitz.Rect, scale: float) -> Image.Image:
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=rect, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def render_page(page: fitz.Page, scale: float) -> Image.Image:
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def bbox_from_easyocr(points: list[list[float]]) -> list[int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def bbox_area(bbox: list[int]) -> int:
    return max(1, bbox[2] - bbox[0]) * max(1, bbox[3] - bbox[1])


def guess_ui_type(text: str, bbox: list[int]) -> str:
    normalized = normalize(text)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    if any(
        token in normalized
        for token in (
            "создать",
            "добавить",
            "ок",
            "далее",
            "подключить",
            "записать",
            "сохранить",
            "выбрать",
            "открыть",
            "сформировать",
        )
    ):
        return "button"

    if any(
        token in normalized
        for token in (
            "монитор",
            "ссылка",
            "интернет-поддержка",
            "заявки",
            "контроль",
            "арм",
            "отчет",
            "справочник",
        )
    ):
        return "menu_item"

    if width > 120 and height < 60:
        return "menu_item"

    return "text"


def is_bad_text(text: str) -> bool:
    normalized = normalize(text)

    if len(normalized) < 3:
        return True
    if normalized in {
        "и",
        "в",
        "на",
        "по",
        "из",
        "как",
        "что",
        "для",
        "или",
        "стр",
        "рис",
        "ок",
    }:
        return True
    if normalized.isdigit():
        return True
    if len(normalized.split()) > 14:
        return True

    return False


def make_item_id(pdf_name: str, page_num: int, screenshot_idx: int, item_idx: int) -> str:
    stem = Path(pdf_name).stem
    return f"{stem}_p{page_num:04d}_s{screenshot_idx:02d}_e{item_idx:04d}"


def process_screenshot(
    *,
    reader: Any,
    img: Image.Image,
    screenshot_path: Path,
    pdf_path: Path,
    page_num: int,
    screenshot_idx: int,
    page_text: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(screenshot_path)

    try:
        ocr = reader.readtext(np.asarray(img))
    except Exception as exc:
        print(f"  OCR error p{page_num} s{screenshot_idx}: {exc}")
        return []

    rows: list[dict[str, Any]] = []
    item_idx = 0

    for points, text, conf in ocr:
        text = clean_text(text)

        if conf < args.min_conf:
            continue
        if is_bad_text(text):
            continue

        bbox = bbox_from_easyocr(points)
        if bbox_area(bbox) < args.min_area:
            continue

        item_idx += 1
        normalized = normalize(text)
        rows.append(
            {
                "id": make_item_id(pdf_path.name, page_num, screenshot_idx, item_idx),
                "pdf_name": pdf_path.name,
                "page": page_num,
                "screenshot_idx": screenshot_idx,
                "screenshot_image": str(screenshot_path.resolve()).replace("\\", "/"),
                "text": text,
                "normalized_text": normalized,
                "bbox": bbox,
                "ui_type": guess_ui_type(text, bbox),
                "confidence": float(conf),
                "context_text": clean_text(
                    f"PDF: {pdf_path.name}. Page: {page_num}. UI text: {text}. "
                    f"Page context: {page_text[: args.max_context_chars]}"
                ),
            }
        )

    return rows


def build_index(args: argparse.Namespace) -> None:
    pdf_dir = Path(args.pdf_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    screenshots_dir = out_dir / "screenshots"
    items_path = out_dir / "ui_items.jsonl"

    out_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    existing_items = load_jsonl(items_path) if args.resume else []
    if not args.resume and items_path.exists():
        items_path.unlink()

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
        pdfs = [path for path in pdfs if args.pdf_name.lower() in path.name.lower()]

    if not pdfs:
        raise FileNotFoundError(f"PDF files were not found in {pdf_dir}")

    ocr_gpu = resolve_ocr_gpu(args)
    print(f"[OCR] EasyOCR device: {'cuda' if ocr_gpu else 'cpu'}")
    reader = get_reader(gpu=ocr_gpu)
    total_new_items = 0
    total_screenshots = 0

    for pdf_path in tqdm(pdfs, desc="PDF files"):
        print(f"\n[PDF] {pdf_path.name}")
        doc = fitz.open(pdf_path)

        start_page = max(1, args.start_page)
        end_page = len(doc)
        if args.max_pages:
            end_page = min(end_page, start_page + args.max_pages - 1)

        for page_num in range(start_page, end_page + 1):
            page = doc[page_num - 1]
            page_text = clean_text(page.get_text("text"))
            blocks = get_image_blocks(page, max_blocks=args.max_blocks_per_page)

            if not blocks and args.full_page_when_no_images:
                blocks = [page.rect]

            if not blocks:
                continue

            for screenshot_idx, rect in enumerate(blocks, start=1):
                key = (pdf_path.name, page_num, screenshot_idx)
                if key in processed_keys:
                    continue

                if rect == page.rect:
                    img = render_page(page, scale=args.scale)
                else:
                    img = render_crop(page, rect, scale=args.scale)

                screenshot_name = f"{pdf_path.stem}_p{page_num:04d}_s{screenshot_idx:02d}.png"
                screenshot_path = screenshots_dir / screenshot_name

                rows = process_screenshot(
                    reader=reader,
                    img=img,
                    screenshot_path=screenshot_path,
                    pdf_path=pdf_path,
                    page_num=page_num,
                    screenshot_idx=screenshot_idx,
                    page_text=page_text,
                    args=args,
                )

                total_screenshots += 1
                if rows:
                    write_jsonl(items_path, rows, mode="a")
                    total_new_items += len(rows)

                print(f"  p{page_num} s{screenshot_idx}: items={len(rows)}")

    meta = {
        "pdf_dir": str(pdf_dir),
        "out_dir": str(out_dir),
        "items_path": str(items_path),
        "new_items": total_new_items,
        "screenshots": total_screenshots,
        "resume": args.resume,
        "scale": args.scale,
        "max_blocks_per_page": args.max_blocks_per_page,
        "min_conf": args.min_conf,
        "full_page_when_no_images": args.full_page_when_no_images,
    }
    with (out_dir / "ui_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] screenshots processed: {total_screenshots}")
    print(f"[OK] new UI items: {total_new_items}")
    print(f"[OK] saved: {items_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OCR UI index from PDF screenshots.")
    parser.add_argument("--pdf-dir", type=Path, default=PDF_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--pdf-name", default=None)
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-blocks-per-page", type=int, default=3)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--min-conf", type=float, default=0.20)
    parser.add_argument("--min-area", type=int, default=40)
    parser.add_argument("--max-context-chars", type=int, default=1800)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu", action="store_true", help="Compatibility alias for --ocr-device cuda.")
    parser.add_argument(
        "--ocr-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="EasyOCR device. auto uses CUDA when torch.cuda.is_available() is true.",
    )
    parser.add_argument(
        "--full-page-when-no-images",
        action="store_true",
        help="Render a whole PDF page when it has no embedded image blocks.",
    )
    return parser.parse_args()


def main() -> None:
    build_index(parse_args())


if __name__ == "__main__":
    main()
