from __future__ import annotations

import torch.nn as nn
from transformers import AutoModel

from .siglip2_text_tower import SigLIP2TextTower
from .siglip2_image_tower import SigLIP2ImageTower


def _unfreeze_last_blocks(backbone, branch_name: str, n_blocks: int) -> None:
    if n_blocks <= 0:
        return
    branch = getattr(backbone, branch_name, None)
    if branch is None:
        return
    encoder = getattr(branch, "encoder", None)
    if encoder is None:
        return
    layers = getattr(encoder, "layers", None)
    if layers is None:
        return
    for layer in list(layers)[-n_blocks:]:
        for param in layer.parameters():
            param.requires_grad = True


class UIDualEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        out_dim: int = 256,
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 0,
        cache_dir: str | None = None,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        _unfreeze_last_blocks(self.backbone, "text_model", unfreeze_last_n_blocks)
        _unfreeze_last_blocks(self.backbone, "vision_model", unfreeze_last_n_blocks)

        self.text_tower = SigLIP2TextTower(
            model_name=model_name,
            out_dim=out_dim,
            freeze_backbone=False,
            backbone=self.backbone,
            cache_dir=cache_dir,
        )
        image_hidden = self.backbone.config.vision_config.hidden_size
        self.image_tower = SigLIP2ImageTower(
            model_name=model_name,
            image_feat_dim=image_hidden,
            bbox_feat_dim=64,
            out_dim=out_dim,
            freeze_backbone=False,
            backbone=self.backbone,
            cache_dir=cache_dir,
        )

    def encode_text(self, text_inputs: dict):
        return self.text_tower(text_inputs)

    def encode_image(self, image_inputs: dict, bbox_features):
        return self.image_tower(image_inputs, bbox_features)

    def forward(self, batch: dict) -> dict:
        z_text = self.encode_text(batch["text_inputs"])
        z_pos = self.encode_image(batch["pos_image_inputs"], batch["pos_bbox_features"])
        z_neg = self.encode_image(batch["neg_image_inputs"], batch["neg_bbox_features"])
        return {
            "z_text": z_text,
            "z_pos": z_pos,
            "z_neg": z_neg,
        }
