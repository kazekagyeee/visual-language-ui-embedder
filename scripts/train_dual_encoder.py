from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.train_config import TrainConfig
from src.data.triplet_dataset import UITTripletDataset
from src.models.ui_dual_encoder import UIDualEncoder
from src.training.trainer import train_model
from src.utils.io import ensure_hf_cache_env
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", required=True)
    parser.add_argument("--split-path", required=True)
    parser.add_argument("--output-dir", default="artifacts/dual_encoder")
    parser.add_argument("--config-path")
    args = parser.parse_args()

    config = TrainConfig.from_json(args.config_path) if args.config_path else TrainConfig(output_dir=args.output_dir)
    config.output_dir = args.output_dir
    set_seed(config.seed)
    ensure_hf_cache_env(config.cache_dir)
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(config.model_name, cache_dir=config.cache_dir)

    train_dataset = UITTripletDataset(
        json_path=args.json_path,
        split_path=args.split_path,
        processor=processor,
        split="train",
        crop_pad_ratio=config.crop_pad_ratio,
        min_crop_size_px=config.min_crop_size_px,
        bbox_epsilon=config.bbox_epsilon,
        apply_augmentations=True,
        augmentation_kwargs={
            "gaussian_blur_prob": config.gaussian_blur_prob,
            "compression_noise_prob": config.compression_noise_prob,
            "brightness_jitter_prob": config.brightness_jitter_prob,
            "brightness_jitter_range": config.brightness_jitter_range,
            "jpeg_quality_min": config.jpeg_quality_min,
            "jpeg_quality_max": config.jpeg_quality_max,
        },
    )
    val_dataset = UITTripletDataset(
        json_path=args.json_path,
        split_path=args.split_path,
        processor=processor,
        split="val",
        crop_pad_ratio=config.crop_pad_ratio,
        min_crop_size_px=config.min_crop_size_px,
        bbox_epsilon=config.bbox_epsilon,
        apply_augmentations=False,
    )
    model = UIDualEncoder(
        model_name=config.model_name,
        out_dim=config.embedding_dim,
        freeze_backbone=config.freeze_backbone,
        unfreeze_last_n_blocks=config.unfreeze_last_n_blocks,
        cache_dir=config.cache_dir,
    )
    result = train_model(model, processor, train_dataset, val_dataset, config)
    config.save_json(Path(config.output_dir) / "train_config.json")
    print(result)


if __name__ == "__main__":
    main()
