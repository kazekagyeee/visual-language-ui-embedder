from __future__ import annotations

import torch


@torch.inference_mode()
def encode_text(model, processor, texts: list[str], device: str) -> torch.Tensor:
    inputs = processor(
        text=texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=64,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return model.encode_text(inputs)
