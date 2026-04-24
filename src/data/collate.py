from __future__ import annotations

import torch


def _move_to_processor_inputs(processed: dict) -> dict:
    return {key: value for key, value in processed.items()}


def build_triplet_collate_fn(processor):
    def collate_fn(samples: list[dict]) -> dict:
        texts = [sample["text"] for sample in samples]
        pos_images = [sample["pos_image"] for sample in samples]
        neg_images = [sample["neg_image"] for sample in samples]

        text_inputs = processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        )
        pos_image_inputs = processor(images=pos_images, return_tensors="pt")
        neg_image_inputs = processor(images=neg_images, return_tensors="pt")

        return {
            "sample_id": [sample["sample_id"] for sample in samples],
            "image_path": [sample["image_path"] for sample in samples],
            "text": texts,
            "query_text": [sample["query_text"] for sample in samples],
            "pos_bbox": [sample["pos_bbox"] for sample in samples],
            "neg_bbox": [sample["neg_bbox"] for sample in samples],
            "text_inputs": _move_to_processor_inputs(text_inputs),
            "pos_image_inputs": _move_to_processor_inputs(pos_image_inputs),
            "neg_image_inputs": _move_to_processor_inputs(neg_image_inputs),
            "pos_bbox_features": torch.stack([sample["pos_bbox_features"] for sample in samples], dim=0),
            "neg_bbox_features": torch.stack([sample["neg_bbox_features"] for sample in samples], dim=0),
        }

    return collate_fn
