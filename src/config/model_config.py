from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_name: str = "google/siglip2-so400m-patch16-naflex"
    embedding_dim: int = 256
    bbox_feature_dim: int = 9
    bbox_hidden_dim: int = 64
    bbox_out_dim: int = 64
    fusion_hidden_dim: int = 512
    fusion_dropout: float = 0.1
    bbox_dropout: float = 0.1
    freeze_backbone: bool = True
    unfreeze_last_n_blocks: int = 0
