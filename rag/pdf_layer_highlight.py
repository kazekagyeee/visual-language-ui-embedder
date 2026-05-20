# -*- coding: utf-8 -*-

import re
from pathlib import Path

import fitz


def normalize(text):
    text = str(text).lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def get_image_blocks(page):
    info = page.get_text("dict")
    rects = []

    for block in info.get("blocks", []):
        if block.get("type") == 1:
            rect = fitz.Rect(block["bbox"])
            if rect.width > 100 and rect.height > 60:
                rects.append(rect)

    return rects


def rect_intersects_any(rect, containers, margin=12):
    for c in containers:
        expanded = fitz.Rect(c.x0 - margin, c.y0 - margin, c.x1 + margin, c.y1 + margin)
        if rect.intersects(expanded):
            return True
    return False


def phrase_variants(target):
    t = str(target).strip()
    variants = [t]

    n = normalize(t)

    if "монитор" in n and "интернет" in n:
        variants.extend([
            "Монитор Интернет-поддержки",
            "Монитор интернет-поддержки",
            "Монитор Интернет поддержки",
            "Монитор интернет поддержки",
        ])

    if "заполнить" in n:
        variants.extend(["Заполнить"])

    if "начните отсюда" in n:
        variants.extend(["Начните отсюда"])

    return list(dict.fromkeys(variants))


def highlight_pdf_layer_targets(pdf_dir, response, targets, out_dir="temp/pdf_layer_highlight"):
    pdf_name = response.get("pdf_name")
    page_num = int(response.get("page") or 1)

    if not pdf_name:
        return []

    pdf_path = Path(pdf_dir) / pdf_name

    if not pdf_path.exists():
        return []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)

    outputs = []

    # Проверяем найденную страницу и 2 следующие: часто текстовый фрагмент указывает на соседний скрин.
    for p in range(page_num, min(page_num + 3, len(doc)) + 1):
        page = doc[p - 1]
        image_blocks = get_image_blocks(page)

        if not image_blocks:
            continue

        matched = []

        for target in targets:
            for phrase in phrase_variants(target):
                rects = page.search_for(phrase)

                for rect in rects:
                    if not rect_intersects_any(rect, image_blocks):
                        continue

                    annot = page.add_rect_annot(rect)
                    annot.set_colors(stroke=(0, 0.75, 0))
                    annot.set_border(width=2.0)
                    annot.update()
                    matched.append(target)

        if not matched:
            continue

        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        out_path = out_dir / f"{Path(pdf_name).stem}_p{p:04d}_pdf_layer_marked.png"
        pix.save(out_path)

        outputs.append(
            {
                "path": str(out_path),
                "targets": sorted(set(matched)),
                "page": p,
            }
        )

    return outputs
