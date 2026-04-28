from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass
class TrainConfig:
    model_name: str = "google/siglip2-so400m-patch16-naflex"
    embedding_dim: int = 256
    crop_pad_ratio: float = 0.05
    min_crop_size_px: int = 4
    bbox_epsilon: float = 1e-3

    freeze_backbone: bool = True
    unfreeze_last_n_blocks: int = 0

    temperature: float = 0.07
    triplet_margin: float = 0.2
    lambda_triplet: float = 0.3

    lr_proj: float = 1e-3
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4

    epochs: int = 5
    micro_batch_size: int = 8
    grad_accum_steps: int = 8
    effective_batch_size: int = 64

    num_workers: int = 4
    seed: int = 42
    mixed_precision: str = "bf16"
    early_stopping_metric: str = "recall@1"
    early_stopping_patience: int = 2

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    gaussian_blur_prob: float = 0.05
    compression_noise_prob: float = 0.05
    brightness_jitter_prob: float = 0.05
    brightness_jitter_range: float = 0.05
    jpeg_quality_min: int = 80
    jpeg_quality_max: int = 95

    output_dir: str = "artifacts/dual_encoder"
    cache_dir: str = ".hf_cache"
    log_every_n_steps: int = 10
    max_grad_norm: float = 1.0
    device: str = "cuda"

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)
