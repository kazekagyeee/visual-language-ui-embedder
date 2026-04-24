from __future__ import annotations

import torch


@torch.inference_mode()
def encode_region(model, processor, images, bbox_features: torch.Tensor, device: str) -> torch.Tensor:
    image_inputs = processor(images=images, return_tensors="pt")
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
    bbox_features = bbox_features.to(device)
    return model.encode_image(image_inputs, bbox_features)
