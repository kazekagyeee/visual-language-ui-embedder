from __future__ import annotations

from pathlib import Path
import hashlib
import json

from PIL import Image
from torch.utils.data import Dataset

from .bbox_utils import (
    clamp_bbox_norm,
    compute_bbox_features,
    denormalize_bbox,
    is_valid_normalized_bbox,
    pad_bbox_norm,
    safe_expand_pixel_bbox,
)
from .image_utils import apply_mild_ui_augmentations, ensure_rgb
from .audit import resolve_image_path
from .splits import load_split_mapping


class UITTripletDataset(Dataset):
    def __init__(
        self,
        json_path,
        split_path,
        processor,
        split: str = "train",
        crop_pad_ratio: float = 0.05,
        min_crop_size_px: int = 4,
        bbox_epsilon: float = 1e-3,
        apply_augmentations: bool = False,
        augmentation_kwargs: dict | None = None,
    ):
        self.json_path = Path(json_path)
        self.split_mapping = load_split_mapping(split_path)
        self.processor = processor
        self.split = split
        self.crop_pad_ratio = crop_pad_ratio
        self.min_crop_size_px = min_crop_size_px
        self.bbox_epsilon = bbox_epsilon
        self.apply_augmentations = apply_augmentations
        self.augmentation_kwargs = augmentation_kwargs or {}
        raw_samples = json.loads(self.json_path.read_text(encoding="utf-8"))
        self.samples = self._filter_samples(raw_samples)

    def _filter_samples(self, samples: list[dict]) -> list[dict]:
        filtered = []
        seen = set()
        for item in samples:
            image_path = str(item.get("image_path", ""))
            if self.split_mapping.get(image_path) != self.split:
                continue
            text = str(item.get("text", "")).strip()
            pos_bbox = item.get("pos_bbox")
            neg_bbox = item.get("neg_bbox")
            if not text:
                continue
            if not is_valid_normalized_bbox(pos_bbox or [], self.bbox_epsilon):
                continue
            if not is_valid_normalized_bbox(neg_bbox or [], self.bbox_epsilon):
                continue
            try:
                resolved_image_path = resolve_image_path(image_path, json_path=self.json_path)
                with Image.open(resolved_image_path) as img:
                    width, height = img.size
            except Exception:
                continue
            pos_px = denormalize_bbox(pos_bbox, width, height)
            neg_px = denormalize_bbox(neg_bbox, width, height)
            if pos_px.width < self.min_crop_size_px or pos_px.height < self.min_crop_size_px:
                continue
            if neg_px.width < self.min_crop_size_px or neg_px.height < self.min_crop_size_px:
                continue

            dedup_key = (image_path, tuple(round(v, 6) for v in pos_bbox), tuple(round(v, 6) for v in neg_bbox), text.strip().lower())
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            filtered.append(
                {
                    "image_path": image_path,
                    "resolved_image_path": str(resolved_image_path),
                    "text": text,
                    "pos_bbox": clamp_bbox_norm(pos_bbox),
                    "neg_bbox": clamp_bbox_norm(neg_bbox),
                }
            )
        return filtered

    def __len__(self):
        return len(self.samples)

    def _make_sample_id(self, image_path: str, text: str, pos_bbox: list[float], neg_bbox: list[float]) -> str:
        payload = json.dumps(
            {"image_path": image_path, "text": text, "pos_bbox": pos_bbox, "neg_bbox": neg_bbox},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _crop_region(self, image: Image.Image, bbox_norm: list[float]) -> Image.Image:
        padded_bbox = pad_bbox_norm(bbox_norm, self.crop_pad_ratio)
        px_bbox = denormalize_bbox(padded_bbox, image.width, image.height)
        px_bbox = safe_expand_pixel_bbox(px_bbox, image.width, image.height, self.min_crop_size_px)
        crop = image.crop((px_bbox.x1, px_bbox.y1, px_bbox.x2, px_bbox.y2))
        crop = ensure_rgb(crop)
        if self.apply_augmentations:
            crop = apply_mild_ui_augmentations(crop, **self.augmentation_kwargs)
        return crop

    def __getitem__(self, idx):
        item = self.samples[idx]
        with Image.open(item["resolved_image_path"]) as img:
            image = ensure_rgb(img).copy()
        pos_bbox = item["pos_bbox"]
        neg_bbox = item["neg_bbox"]
        text = item["text"]
        return {
            "sample_id": self._make_sample_id(item["image_path"], text, pos_bbox, neg_bbox),
            "image_path": item["image_path"],
            "resolved_image_path": item["resolved_image_path"],
            "text": text,
            "query_text": text,
            "full_image": image,
            "pos_image": self._crop_region(image, pos_bbox),
            "neg_image": self._crop_region(image, neg_bbox),
            "pos_bbox_norm": compute_bbox_features(pos_bbox)[:4],
            "neg_bbox_norm": compute_bbox_features(neg_bbox)[:4],
            "pos_bbox": pos_bbox,
            "neg_bbox": neg_bbox,
            "pos_bbox_features": compute_bbox_features(pos_bbox),
            "neg_bbox_features": compute_bbox_features(neg_bbox),
        }
