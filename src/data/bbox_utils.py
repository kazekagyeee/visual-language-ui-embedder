from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import math

import torch


@dataclass(frozen=True)
class PixelBBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)


def is_valid_normalized_bbox(bbox: Iterable[float], epsilon: float = 1e-3) -> bool:
    values = list(bbox)
    if len(values) != 4:
        return False
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        return False
    return all(-epsilon <= v <= 1.0 + epsilon for v in values)


def clamp_bbox_norm(bbox: Iterable[float]) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return [
        min(max(x1, 0.0), 1.0),
        min(max(y1, 0.0), 1.0),
        min(max(x2, 0.0), 1.0),
        min(max(y2, 0.0), 1.0),
    ]


def denormalize_bbox(bbox: Iterable[float], image_width: int, image_height: int) -> PixelBBox:
    x1, y1, x2, y2 = clamp_bbox_norm(bbox)
    left = int(math.floor(x1 * image_width))
    top = int(math.floor(y1 * image_height))
    right = int(math.ceil(x2 * image_width))
    bottom = int(math.ceil(y2 * image_height))
    left = min(max(left, 0), image_width)
    top = min(max(top, 0), image_height)
    right = min(max(right, 0), image_width)
    bottom = min(max(bottom, 0), image_height)
    return PixelBBox(left, top, right, bottom)


def safe_expand_pixel_bbox(
    bbox: PixelBBox,
    image_width: int,
    image_height: int,
    min_size_px: int,
) -> PixelBBox:
    width = max(bbox.width, 1)
    height = max(bbox.height, 1)
    target_w = max(width, min_size_px)
    target_h = max(height, min_size_px)
    cx = (bbox.x1 + bbox.x2) / 2.0
    cy = (bbox.y1 + bbox.y2) / 2.0

    left = int(round(cx - target_w / 2.0))
    top = int(round(cy - target_h / 2.0))
    right = left + target_w
    bottom = top + target_h

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > image_width:
        shift = right - image_width
        left = max(0, left - shift)
        right = image_width
    if bottom > image_height:
        shift = bottom - image_height
        top = max(0, top - shift)
        bottom = image_height

    return PixelBBox(left, top, right, bottom)


def pad_bbox_norm(bbox: Iterable[float], pad_ratio: float) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w = x2 - x1
    h = y2 - y1
    padded = [x1 - w * pad_ratio, y1 - h * pad_ratio, x2 + w * pad_ratio, y2 + h * pad_ratio]
    return clamp_bbox_norm(padded)


def compute_bbox_features(bbox: Iterable[float]) -> torch.FloatTensor:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    area = w * h
    return torch.tensor([x1, y1, x2, y2, cx, cy, w, h, area], dtype=torch.float32)
