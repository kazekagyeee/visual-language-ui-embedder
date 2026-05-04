from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ShortSiameseConfig:
    input_dim: int = 4
    hidden_dim: int = 16
    output_dim: int = 8
    dropout: float = 0.1


class ProjectionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x.float()), dim=-1)


class ShortSiameseEncoder(nn.Module):
    def __init__(self, config: ShortSiameseConfig):
        super().__init__()
        self.config = config
        self.text_encoder = ProjectionEncoder(
            config.input_dim, config.hidden_dim, config.output_dim, config.dropout
        )
        self.image_encoder = ProjectionEncoder(
            config.input_dim, config.hidden_dim, config.output_dim, config.dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.output_dim * 4, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def encode_text(self, text_vec):
        return self.text_encoder(text_vec)

    def encode_image(self, image_vec):
        return self.image_encoder(image_vec)

    def forward(self, text_vec, image_vec):
        text_short = self.encode_text(text_vec)
        image_short = self.encode_image(image_vec)

        features = torch.cat(
            [
                text_short,
                image_short,
                torch.abs(text_short - image_short),
                text_short * image_short,
            ],
            dim=-1,
        )

        logit = self.classifier(features).squeeze(-1)
        score = torch.sigmoid(logit)

        return {
            "text_short": text_short,
            "image_short": image_short,
            "logit": logit,
            "score": score,
        }

    def save(self, path):
        torch.save(
            {
                "config": self.config.__dict__,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path, map_location="cpu"):
        payload = torch.load(path, map_location=map_location)
        config = ShortSiameseConfig(**payload["config"])
        model = cls(config)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model
