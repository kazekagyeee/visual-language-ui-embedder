from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ShortSiameseConfig:
    text_input_dim: int = 3584
    image_input_dim: int = 3584
    short_dim: int = 128
    hidden_dim: int = 512
    dropout: float = 0.1


class ProjectionEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x.float()), dim=-1)


class ShortSiameseEncoder(nn.Module):
    """Dual encoder для text/image vectors + classifier head.

    Вход: длинные Qwen/teacher embeddings или признаки другой модели.
    Выход: короткие text/image embeddings и score схожести.
    """

    def __init__(self, config: ShortSiameseConfig):
        super().__init__()
        self.config = config
        self.text_encoder = ProjectionEncoder(
            config.text_input_dim, config.hidden_dim, config.short_dim, config.dropout
        )
        self.image_encoder = ProjectionEncoder(
            config.image_input_dim, config.hidden_dim, config.short_dim, config.dropout
        )
        fusion_dim = config.short_dim * 4
        self.similarity_head = nn.Sequential(
            nn.Linear(fusion_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def encode_text(self, text_vec: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(text_vec)

    def encode_image(self, image_vec: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(image_vec)

    def fuse(self, text_short: torch.Tensor, image_short: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [text_short, image_short, torch.abs(text_short - image_short), text_short * image_short],
            dim=-1,
        )

    def forward(self, text_vec: torch.Tensor, image_vec: torch.Tensor) -> Dict[str, torch.Tensor]:
        text_short = self.encode_text(text_vec)
        image_short = self.encode_image(image_vec)
        logits = self.similarity_head(self.fuse(text_short, image_short)).squeeze(-1)
        cosine = F.cosine_similarity(text_short, image_short, dim=-1)
        return {
            "text_short": text_short,
            "image_short": image_short,
            "logits": logits,
            "score": torch.sigmoid(logits),
            "cosine": cosine,
        }

    def save(self, path: str) -> None:
        torch.save({"config": self.config.__dict__, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location: str | torch.device = "cpu") -> "ShortSiameseEncoder":
        payload = torch.load(path, map_location=map_location)
        model = cls(ShortSiameseConfig(**payload["config"]))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model
