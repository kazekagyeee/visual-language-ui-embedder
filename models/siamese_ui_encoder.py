# -*- coding: utf-8 -*-

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SiameseUIConfig:
    text_dim: int = 384
    image_size: int = 128
    hidden_dim: int = 256
    embedding_dim: int = 64


class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.proj = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        x = self.proj(x)
        return F.normalize(x, dim=-1)


class TextEncoder(nn.Module):
    def __init__(self, text_dim=384, hidden_dim=256, embedding_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x):
        x = self.net(x.float())
        return F.normalize(x, dim=-1)


class SiameseUIEncoder(nn.Module):
    def __init__(self, config: SiameseUIConfig):
        super().__init__()
        self.config = config

        self.text_encoder = TextEncoder(
            text_dim=config.text_dim,
            hidden_dim=config.hidden_dim,
            embedding_dim=config.embedding_dim,
        )

        self.image_encoder = ImageEncoder(
            embedding_dim=config.embedding_dim,
        )

        self.classifier = nn.Sequential(
            nn.Linear(config.embedding_dim * 4, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, 1),
        )

    def encode_text(self, text_vec):
        return self.text_encoder(text_vec)

    def encode_image(self, image):
        return self.image_encoder(image)

    def forward(self, text_vec, image):
        text_emb = self.encode_text(text_vec)
        image_emb = self.encode_image(image)

        features = torch.cat(
            [
                text_emb,
                image_emb,
                torch.abs(text_emb - image_emb),
                text_emb * image_emb,
            ],
            dim=-1,
        )

        logits = self.classifier(features).squeeze(-1)
        similarity = F.cosine_similarity(text_emb, image_emb, dim=-1)

        return {
            "text_emb": text_emb,
            "image_emb": image_emb,
            "logits": logits,
            "similarity": similarity,
        }

    def save(self, path):
        payload = {
            "config": self.config.__dict__,
            "state_dict": self.state_dict(),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        payload = torch.load(path, map_location=map_location)
        config = SiameseUIConfig(**payload["config"])
        model = cls(config)
        model.load_state_dict(payload["state_dict"])
        return model
