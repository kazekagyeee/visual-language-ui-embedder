from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.train_config import TrainConfig
from src.data.triplet_dataset import UITTripletDataset
from src.inference.export_embeddings import export_positive_embeddings
from src.models.ui_dual_encoder import UIDualEncoder
from src.training.checkpointing import load_checkpoint
from src.utils.io import ensure_hf_cache_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", required=True)
    parser.add_argument("--split-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()

    config = TrainConfig.from_json(args.config_path)
    ensure_hf_cache_env(config.cache_dir)
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(config.model_name, cache_dir=config.cache_dir)
    dataset = UITTripletDataset(
        json_path=args.json_path,
        split_path=args.split_path,
        processor=processor,
        split=args.split,
        crop_pad_ratio=config.crop_pad_ratio,
        min_crop_size_px=config.min_crop_size_px,
        bbox_epsilon=config.bbox_epsilon,
        apply_augmentations=False,
    )
    model = UIDualEncoder(
        config.model_name,
        out_dim=config.embedding_dim,
        freeze_backbone=config.freeze_backbone,
        unfreeze_last_n_blocks=config.unfreeze_last_n_blocks,
        cache_dir=config.cache_dir,
    )
    load_checkpoint(args.checkpoint_path, model, map_location=config.device)
    model.to(config.device)
    report = export_positive_embeddings(model, dataset, processor, config.device, args.output_prefix, batch_size=config.micro_batch_size)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
