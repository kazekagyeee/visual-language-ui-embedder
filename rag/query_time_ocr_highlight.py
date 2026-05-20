# -*- coding: utf-8 -*-

import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from rag.ocr_cleanup import cleanup_ocr_text


def normalize(text):
    text = cleanup_ocr_text(text)
    text = str(text).lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def get_reader():
    import easyocr
    return easyocr.Reader(["ru", "en"], gpu=False)


def target_tokens(target):
    bad = {"и", "в", "на", "по", "для", "ссылка", "найдите"}
    return [x for x in normalize(target).split() if len(x) > 2 and x not in bad]


def match_target(ocr_text, target):
    o = normalize(ocr_text)
    t = normalize(target)

    if not o or not t:
        return False

    if t in o or o in t:
        return True

    tt = set(target_tokens(target))
    oo = set(o.split())

    if not tt:
        return False

    # Для "Монитор Интернет-поддержки" обязательно должно быть слово монитор.
    if "монитор" in tt and "монитор" not in oo:
        return False

    overlap = len(tt & oo)

    return overlap >= max(1, min(2, len(tt)))


def preprocess_variants(img):
    arr = np.array(img.convert("RGB"))

    variants = []

    variants.append(arr)

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Контраст для мелкого синего текста/ссылок.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    variants.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB))

    # Увеличение.
    big = cv2.resize(arr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    variants.append(big)

    return variants


def draw_matches(original_img, matches, scale_factor=1.0):
    img = original_img.convert("RGB")
    draw = ImageDraw.Draw(img)

    found = []

    for match in matches:
        bbox = match["bbox"]
        target = match["target"]

        x0, y0, x1, y1 = bbox

        x0 = int(x0 / scale_factor)
        y0 = int(y0 / scale_factor)
        x1 = int(x1 / scale_factor)
        y1 = int(y1 / scale_factor)

        pad = 8

        draw.rectangle(
            [x0 - pad, y0 - pad, x1 + pad, y1 + pad],
            outline=(0, 180, 0),
            width=6,
        )

        found.append(target)

    return img, sorted(set(found))


def highlight_targets_on_screenshots(screenshot_paths, targets, out_dir="temp/query_time_ocr"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = get_reader()
    outputs = []

    for screenshot_path in screenshot_paths:
        path = Path(screenshot_path)

        if not path.exists():
            continue

        original = Image.open(path).convert("RGB")

        all_matches = []

        variants = preprocess_variants(original)

        for variant_idx, arr in enumerate(variants):
            scale_factor = 2.0 if variant_idx == 2 else 1.0

            try:
                ocr = reader.readtext(arr, detail=1, paragraph=False)
            except Exception:
                continue

            for bbox, text, conf in ocr:
                if conf < 0.10:
                    continue

                for target in targets:
                    if not match_target(text, target):
                        continue

                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]

                    all_matches.append(
                        {
                            "bbox": [min(xs), min(ys), max(xs), max(ys)],
                            "target": target,
                            "text": text,
                            "conf": float(conf),
                            "scale_factor": scale_factor,
                        }
                    )

        if not all_matches:
            continue

        # Рисуем по первому найденному scale-варианту.
        scale_factor = all_matches[0]["scale_factor"]
        matches = [m for m in all_matches if m["scale_factor"] == scale_factor]

        marked, found = draw_matches(original, matches, scale_factor=scale_factor)

        out_path = out_dir / f"{path.stem}_query_ocr_marked.png"
        marked.save(out_path)

        outputs.append(
            {
                "path": str(out_path),
                "targets": found,
            }
        )

    return outputs
