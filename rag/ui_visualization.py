# -*- coding: utf-8 -*-

from pathlib import Path
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont


def safe_font(size=24):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_ui_results(ui_results, out_dir="temp/ui_result_images"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)

    for global_idx, result in enumerate(ui_results, start=1):
        result = dict(result)
        result["global_idx"] = global_idx
        item = result["item"]
        grouped[item["screenshot_image"]].append(result)

    outputs = []

    for screenshot_path, results in grouped.items():
        path = Path(screenshot_path)

        if not path.exists():
            continue

        img = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(img)
        font = safe_font(26)

        for result in results:
            item = result["item"]
            bbox = item.get("bbox")

            if not bbox:
                continue

            x0, y0, x1, y1 = bbox
            pad = 8

            draw.rectangle(
                [x0 - pad, y0 - pad, x1 + pad, y1 + pad],
                outline=(0, 180, 0),
                width=6,
            )

            label = str(result["global_idx"])

            draw.rectangle(
                [x0 - pad, y0 - 38, x0 + 34, y0 - 4],
                fill=(0, 180, 0),
            )

            draw.text(
                [x0 + 3, y0 - 36],
                label,
                fill=(255, 255, 255),
                font=font,
            )

        out_path = out_dir / f"{path.stem}_marked.png"
        img.save(out_path)

        outputs.append(
            {
                "path": str(out_path),
                "items": [r["item"] for r in results],
                "marked": True,
            }
        )

    return outputs


def show_page_screenshots_from_ui_index(ui_searcher, pdf_name, page, out_dir="temp/ui_page_screenshots"):
    screenshots = []

    for item in ui_searcher.items:
        if item.get("pdf_name") != pdf_name:
            continue

        if int(item.get("page", -1)) != int(page):
            continue

        path = item.get("screenshot_image")

        if path and path not in screenshots:
            screenshots.append(path)

    outputs = []

    for screenshot in screenshots[:4]:
        path = Path(screenshot)

        if not path.exists():
            continue

        outputs.append(
            {
                "path": str(path),
                "items": [],
                "marked": False,
            }
        )

    return outputs
