from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader

from src.config.train_config import TrainConfig
from src.data.collate import build_triplet_collate_fn
from src.data.triplet_dataset import UITTripletDataset
from src.models.ui_dual_encoder import UIDualEncoder
from src.training.checkpointing import load_checkpoint
from src.training.trainer import _build_qualitative_rows, evaluate_model
from src.evaluation.qualitative_report import save_qualitative_report
from src.utils.io import ensure_hf_cache_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", required=True)
    parser.add_argument("--split-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
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
    loader = DataLoader(
        dataset,
        batch_size=config.micro_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=build_triplet_collate_fn(processor),
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
    metrics, rows = evaluate_model(
        model,
        loader,
        device=config.device,
        lambda_triplet=config.lambda_triplet,
        temperature=config.temperature,
        margin=config.triplet_margin,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    output_path = Path(config.output_dir) / f"{args.split}_qualitative.json"
    save_qualitative_report(output_path, _build_qualitative_rows(rows))


if __name__ == "__main__":
    main()
