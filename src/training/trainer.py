from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

from src.data.collate import build_triplet_collate_fn
from src.evaluation.pairwise_metrics import compute_margin_stats, compute_pos_neg_accuracy
from src.evaluation.qualitative_report import save_qualitative_report
from src.evaluation.retrieval_metrics import compute_median_rank, compute_mrr, compute_recall_at_k
from src.training.checkpointing import append_metrics_history, save_checkpoint
from src.training.losses import total_loss
from src.training.optim import build_optimizer
from src.training.scheduler import build_scheduler
from src.utils.io import ensure_dir
from src.utils.logging import get_logger


def _prepare_batch(batch: dict, device: str) -> dict:
    out = dict(batch)
    for key in ("text_inputs", "pos_image_inputs", "neg_image_inputs"):
        out[key] = {k: v.to(device) for k, v in batch[key].items()}
    out["pos_bbox_features"] = batch["pos_bbox_features"].to(device)
    out["neg_bbox_features"] = batch["neg_bbox_features"].to(device)
    return out


def _autocast_context(device: str, mixed_precision: str):
    if not device.startswith("cuda"):
        return nullcontext()
    dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


@torch.inference_mode()
def evaluate_model(model, dataloader, device: str, lambda_triplet: float, temperature: float, margin: float) -> tuple[dict, list[dict]]:
    model.eval()
    z_text_all = []
    z_pos_all = []
    z_neg_all = []
    rows = []
    total_losses = {"loss": 0.0, "contrastive_loss": 0.0, "triplet_loss": 0.0}
    total_batches = 0

    for batch in dataloader:
        batch = _prepare_batch(batch, device)
        outputs = model(batch)
        losses = total_loss(
            outputs["z_text"],
            outputs["z_pos"],
            outputs["z_neg"],
            lambda_triplet=lambda_triplet,
            temperature=temperature,
            margin=margin,
        )
        for key in total_losses:
            total_losses[key] += float(losses[key].item())
        total_batches += 1

        z_text_all.append(outputs["z_text"].detach().cpu())
        z_pos_all.append(outputs["z_pos"].detach().cpu())
        z_neg_all.append(outputs["z_neg"].detach().cpu())

        pos_scores = torch.sum(outputs["z_text"] * outputs["z_pos"], dim=-1).detach().cpu().tolist()
        neg_scores = torch.sum(outputs["z_text"] * outputs["z_neg"], dim=-1).detach().cpu().tolist()
        for idx in range(len(batch["sample_id"])):
            rows.append(
                {
                    "sample_id": batch["sample_id"][idx],
                    "image_path": batch["image_path"][idx],
                    "text": batch["text"][idx],
                    "pos_bbox": batch["pos_bbox"][idx],
                    "neg_bbox": batch["neg_bbox"][idx],
                    "cos_text_pos": pos_scores[idx],
                    "cos_text_neg": neg_scores[idx],
                }
            )

    if not rows:
        return {"loss": 0.0}, []

    z_text = torch.cat(z_text_all, dim=0)
    z_pos = torch.cat(z_pos_all, dim=0)
    z_neg = torch.cat(z_neg_all, dim=0)
    gt_indices = list(range(z_pos.size(0)))
    score_matrix = z_text @ z_pos.T
    topk = min(5, z_pos.size(0))
    top_indices = torch.topk(score_matrix, k=topk, dim=1).indices.cpu().tolist()
    for idx, candidates in enumerate(top_indices):
        rows[idx]["top_5_retrieved_samples"] = [
            {
                "sample_id": rows[candidate]["sample_id"],
                "image_path": rows[candidate]["image_path"],
                "pos_bbox": rows[candidate]["pos_bbox"],
                "score": float(score_matrix[idx, candidate].item()),
            }
            for candidate in candidates
        ]

    metrics = {
        "loss": total_losses["loss"] / max(1, total_batches),
        "contrastive_loss": total_losses["contrastive_loss"] / max(1, total_batches),
        "triplet_loss": total_losses["triplet_loss"] / max(1, total_batches),
        "pos_vs_neg_accuracy": compute_pos_neg_accuracy(z_text, z_pos, z_neg),
        "mean_cosine_text_pos": float(torch.sum(z_text * z_pos, dim=-1).mean().item()),
        "mean_cosine_text_neg": float(torch.sum(z_text * z_neg, dim=-1).mean().item()),
        "embedding_norm_pre_l2": 1.0,
        "outlier_count_cos_neg_gt_pos": int(sum(1 for row in rows if row["cos_text_neg"] > row["cos_text_pos"])),
    }
    metrics.update(compute_margin_stats(z_text, z_pos, z_neg))
    metrics.update(compute_recall_at_k(z_text, z_pos, gt_indices))
    metrics["mrr"] = compute_mrr(z_text, z_pos, gt_indices)
    metrics["median_rank"] = compute_median_rank(z_text, z_pos, gt_indices)
    return metrics, rows


def _build_qualitative_rows(rows: list[dict], top_k: int = 20) -> list[dict]:
    sorted_rows = sorted(rows, key=lambda row: row["cos_text_pos"] - row["cos_text_neg"])
    false_negative = [row for row in sorted_rows if row["cos_text_neg"] > row["cos_text_pos"]][:top_k]
    best = sorted(rows, key=lambda row: row["cos_text_pos"] - row["cos_text_neg"], reverse=True)[:top_k]
    worst = sorted_rows[:top_k]
    false_positive = sorted(rows, key=lambda row: row["cos_text_neg"], reverse=True)[:top_k]
    return [
        {"bucket": "best", **row} for row in best
    ] + [
        {"bucket": "worst", **row} for row in worst
    ] + [
        {"bucket": "false_positive", **row} for row in false_positive
    ] + [
        {"bucket": "false_negative", **row} for row in false_negative
    ]


def train_model(model, processor, train_dataset, val_dataset, config):
    logger = get_logger()
    output_dir = ensure_dir(config.output_dir)
    collate_fn = build_triplet_collate_fn(processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.micro_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.micro_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )

    total_steps = max(1, (len(train_loader) * config.epochs) // max(1, config.grad_accum_steps))
    optimizer = build_optimizer(model, config.lr_proj, config.lr_backbone, config.weight_decay)
    scheduler = build_scheduler(optimizer, total_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=config.device.startswith("cuda") and config.mixed_precision == "fp16")
    best_metric = float("-inf")
    patience = 0
    history_path = Path(output_dir) / "metrics_history.jsonl"

    model.to(config.device)
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = {"loss": 0.0, "contrastive_loss": 0.0, "triplet_loss": 0.0}

        for step, batch in enumerate(train_loader):
            batch = _prepare_batch(batch, config.device)
            with _autocast_context(config.device, config.mixed_precision):
                outputs = model(batch)
                losses = total_loss(
                    outputs["z_text"],
                    outputs["z_pos"],
                    outputs["z_neg"],
                    lambda_triplet=config.lambda_triplet,
                    temperature=config.temperature,
                    margin=config.triplet_margin,
                )
                loss = losses["loss"] / config.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % config.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            for key in running:
                running[key] += float(losses[key].item())

            if (step + 1) % config.log_every_n_steps == 0:
                logger.info(
                    "epoch=%s step=%s/%s train_loss=%.4f contrastive=%.4f triplet=%.4f lr=%.2e",
                    epoch + 1,
                    step + 1,
                    len(train_loader),
                    running["loss"] / (step + 1),
                    running["contrastive_loss"] / (step + 1),
                    running["triplet_loss"] / (step + 1),
                    optimizer.param_groups[0]["lr"],
                )

        val_metrics, val_rows = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=config.device,
            lambda_triplet=config.lambda_triplet,
            temperature=config.temperature,
            margin=config.triplet_margin,
        )
        train_metrics = {f"train_{k}": v / max(1, len(train_loader)) for k, v in running.items()}
        epoch_metrics = {"epoch": epoch + 1, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
        append_metrics_history(history_path, epoch_metrics)
        logger.info("epoch=%s val_metrics=%s", epoch + 1, epoch_metrics)

        save_checkpoint(Path(output_dir) / "last.ckpt", model, optimizer, scheduler, epoch + 1, best_metric=best_metric)

        metric_name = f"val_{config.early_stopping_metric}"
        score = float(epoch_metrics.get(metric_name, float("-inf")))
        if score > best_metric:
            best_metric = score
            patience = 0
            save_checkpoint(Path(output_dir) / "best_recall_at_1.ckpt", model, optimizer, scheduler, epoch + 1, best_metric=best_metric)
            save_qualitative_report(Path(output_dir) / "best_val_qualitative.json", _build_qualitative_rows(val_rows))
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                logger.info("early stopping triggered at epoch=%s", epoch + 1)
                break

    return {"best_metric": best_metric, "output_dir": str(output_dir)}
