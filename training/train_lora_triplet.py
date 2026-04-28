import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import UIEmbedderConfig
from main import UIEmbedderPipeline, smart_resize


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "training" / "output"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "training" / "triplet_dataset_clean.json"
CHECKPOINT_DIR = str(DEFAULT_OUTPUT_DIR / "checkpoints")
FINAL_ADAPTER_DIR = str(DEFAULT_OUTPUT_DIR / "lora_triplet_adapter")
TRAINING_DTYPE = torch.bfloat16


def force_bfloat16(module):
    module.to(dtype=TRAINING_DTYPE)
    return module


def print_floating_dtype_summary(name, module):
    counts = {}
    for param in module.parameters():
        if param.is_floating_point():
            counts[str(param.dtype)] = counts.get(str(param.dtype), 0) + param.numel()
    for buffer in module.buffers():
        if buffer.is_floating_point():
            counts[str(buffer.dtype)] = counts.get(str(buffer.dtype), 0) + buffer.numel()
    print(f"  [dtype] {name}: {counts}", flush=True)


def parse_args():
    output_dir = Path(os.environ.get("TRAINING_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    dataset_path = Path(os.environ.get("TRIPLET_DATASET_PATH", DEFAULT_DATASET_PATH))
    data_root_env = os.environ.get("TRIPLET_DATA_ROOT")

    parser = argparse.ArgumentParser(description="Train LoRA adapter with triplet loss.")
    parser.add_argument("--dataset-path", type=Path, default=dataset_path)
    parser.add_argument("--data-root", type=Path, default=Path(data_root_env) if data_root_env else None)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--final-adapter-dir", type=Path, default=None)
    parser.add_argument("--model-size", default=os.environ.get("MODEL_SIZE", "2B"))
    parser.add_argument("--device", default=os.environ.get("TRAINING_DEVICE", "cuda"))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "3")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("LR", "2e-5")))
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "4")),
    )
    parser.add_argument("--triplet-margin", type=float, default=float(os.environ.get("TRIPLET_MARGIN", "0.2")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "0")))
    parser.add_argument("--log-every", type=int, default=int(os.environ.get("LOG_EVERY", "25")))
    parser.add_argument(
        "--progress",
        choices=("auto", "tqdm", "plain", "off"),
        default=os.environ.get("PROGRESS", "auto"),
        help="Progress output mode. auto uses tqdm only for interactive terminals.",
    )
    return parser.parse_args()


class TripletUIDataset(Dataset):
    def __init__(self, data_path_or_list, data_root=None, load_images=True):
        self.data_root = Path(data_root).resolve() if data_root else None
        self.dataset_dir = None
        self.load_images = load_images

        if isinstance(data_path_or_list, (str, os.PathLike, Path)):
            dataset_path = Path(data_path_or_list).resolve()
            self.dataset_dir = dataset_path.parent
            with open(dataset_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        elif isinstance(data_path_or_list, list):
            self.data = data_path_or_list
        else:
            self.data = []

    def _resolve_image_path(self, image_path):
        raw_path = Path(image_path)
        candidates = []

        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            if self.data_root is not None:
                candidates.append(self.data_root / raw_path)
                parts = raw_path.parts
                if parts and parts[0] == "training":
                    candidates.append(self.data_root / Path(*parts[1:]))
            if self.dataset_dir is not None:
                candidates.append(self.dataset_dir / raw_path)
            candidates.append(PROJECT_ROOT / raw_path)
            candidates.append(raw_path)

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0] if candidates else raw_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if not self.load_images:
            image = None
        else:
            try:
                image = Image.open(self._resolve_image_path(item["image_path"])).convert("RGB")
            except Exception:
                image = Image.new("RGB", (224, 224), (0, 0, 0))

        return {
            "image": image,
            "anchor_text": item["text"],
            "pos_bbox": item["pos_bbox"],
            "neg_bbox": item["neg_bbox"],
            "sample_idx": idx,
        }


def prepare_sequences_for_llm(pipeline, image, text_content, bbox, device):
    image = smart_resize(image, patch_size=pipeline.config.patch_size_resize)
    img_tensor = pipeline.transform(image).unsqueeze(0).to(device=device, dtype=TRAINING_DTYPE)
    if bbox is None:
        boxes_tensor = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]], device=device, dtype=TRAINING_DTYPE)
    else:
        boxes_tensor = torch.tensor([[bbox]], device=device, dtype=TRAINING_DTYPE)

    with torch.no_grad():
        g_seq, b_seqs = pipeline.box_encoder(img_tensor, boxes_tensor)
        b_seq = b_seqs[0]
        g_summary = g_seq.mean(dim=1, keepdim=True) if pipeline.config.use_global_summary else None
        text_emb_box = pipeline._prepare_text_embeddings(text_content, bbox=bbox)
        s_prefix = 1 if g_summary is not None else 0
        combined_seq = (
            torch.cat([g_summary, b_seq, text_emb_box], dim=1)
            if g_summary is not None
            else torch.cat([b_seq, text_emb_box], dim=1)
        )
        box_start = s_prefix
        box_end = s_prefix + b_seq.shape[1]

    return [combined_seq], s_prefix, [box_start], [box_end]


def prepare_triplet_sequences_for_llm(pipeline, image, text_content, pos_bbox, neg_bbox, device):
    image = smart_resize(image, patch_size=pipeline.config.patch_size_resize)
    img_tensor = pipeline.transform(image).unsqueeze(0).to(device=device, dtype=TRAINING_DTYPE)
    triplet_boxes = [[0.0, 0.0, 1.0, 1.0], pos_bbox, neg_bbox]
    boxes_tensor = torch.tensor([triplet_boxes], device=device, dtype=TRAINING_DTYPE)

    with torch.no_grad():
        g_seq, b_seqs = pipeline.box_encoder(img_tensor, boxes_tensor)
        g_summary = g_seq.mean(dim=1, keepdim=True) if pipeline.config.use_global_summary else None
        s_prefix = 1 if g_summary is not None else 0
        pipeline._build_text_token_cache(text_content)

        seqs_list = []
        s_box_starts = []
        s_box_ends = []
        text_bboxes = [None, pos_bbox, neg_bbox]

        for b_seq, text_bbox in zip(b_seqs, text_bboxes):
            text_emb_box = pipeline._prepare_text_embeddings(text_content, bbox=text_bbox)
            box_start = s_prefix
            box_end = s_prefix + b_seq.shape[1]
            combined_seq = (
                torch.cat([g_summary, b_seq, text_emb_box], dim=1)
                if g_summary is not None
                else torch.cat([b_seq, text_emb_box], dim=1)
            )
            seqs_list.append(combined_seq)
            s_box_starts.append(box_start)
            s_box_ends.append(box_end)

    return seqs_list, s_prefix, s_box_starts, s_box_ends


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, checkpoint_dir=CHECKPOINT_DIR):
    epoch_dir = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}")
    os.makedirs(epoch_dir, exist_ok=True)
    model.save_pretrained(os.path.join(epoch_dir, "lora_adapter"))
    torch.save(
        {
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics,
        },
        os.path.join(epoch_dir, "training_state.pt"),
    )
    print(f"  [CHECKPOINT] Saved epoch {epoch + 1} checkpoint to: {epoch_dir}", flush=True)


def find_latest_checkpoint(checkpoint_dir=CHECKPOINT_DIR):
    if not os.path.exists(checkpoint_dir):
        return None
    epoch_dirs = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*")))
    if not epoch_dirs:
        return None
    latest = epoch_dirs[-1]
    state_path = os.path.join(latest, "training_state.pt")
    adapter_path = os.path.join(latest, "lora_adapter")
    return latest if os.path.exists(state_path) and os.path.exists(adapter_path) else None


def load_checkpoint(checkpoint_path, base_model, optimizer, scheduler, device):
    adapter_path = os.path.join(checkpoint_path, "lora_adapter")
    state_path = os.path.join(checkpoint_path, "training_state.pt")
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
    force_bfloat16(model)
    model.train()

    training_state = torch.load(state_path, map_location=device, weights_only=False)
    new_optimizer = AdamW(
        model.parameters(),
        lr=optimizer.defaults["lr"],
        weight_decay=optimizer.defaults["weight_decay"],
    )
    new_optimizer.load_state_dict(training_state["optimizer_state_dict"])

    if scheduler is not None and training_state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(training_state["scheduler_state_dict"])

    start_epoch = training_state["epoch"] + 1
    prev_metrics = training_state.get("metrics", {})
    return model, new_optimizer, start_epoch, prev_metrics


class TripletMetrics:
    def __init__(self, margin=0.2):
        self.margin = margin
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.total_cos_sim_pos = 0.0
        self.total_cos_sim_neg = 0.0
        self.total_correct = 0
        self.total_margin_violations = 0
        self.count = 0

    @torch.no_grad()
    def update(self, anchor, positive, negative, loss_value):
        cos_pos = F.cosine_similarity(anchor, positive, dim=-1).mean().item()
        cos_neg = F.cosine_similarity(anchor, negative, dim=-1).mean().item()
        dist_pos = 1.0 - cos_pos
        dist_neg = 1.0 - cos_neg

        self.total_cos_sim_pos += cos_pos
        self.total_cos_sim_neg += cos_neg
        self.total_loss += loss_value
        if dist_pos < dist_neg:
            self.total_correct += 1
        if dist_pos - dist_neg + self.margin > 0:
            self.total_margin_violations += 1
        self.count += 1

    def compute(self):
        if self.count == 0:
            return {}
        return {
            "avg_loss": self.total_loss / self.count,
            "avg_cos_sim_positive": self.total_cos_sim_pos / self.count,
            "avg_cos_sim_negative": self.total_cos_sim_neg / self.count,
            "cos_sim_gap": (self.total_cos_sim_pos - self.total_cos_sim_neg) / self.count,
            "triplet_accuracy": self.total_correct / self.count,
            "margin_violation_rate": self.total_margin_violations / self.count,
            "num_samples": self.count,
        }

    @staticmethod
    def print_metrics(metrics, epoch, total_epochs, elapsed_sec):
        print(f"\n{'=' * 70}")
        print(f"  EPOCH {epoch}/{total_epochs} RESULTS")
        print(f"{'=' * 70}")
        print(f"  Avg loss:          {metrics['avg_loss']:.6f}")
        print(f"  Cos sim anchor-pos:{metrics['avg_cos_sim_positive']:.4f}")
        print(f"  Cos sim anchor-neg:{metrics['avg_cos_sim_negative']:.4f}")
        print(f"  Cos sim gap:       {metrics['cos_sim_gap']:+.4f}")
        print(f"  Triplet accuracy:  {metrics['triplet_accuracy'] * 100:.1f}%")
        print(f"  Margin violations: {metrics['margin_violation_rate'] * 100:.1f}%")
        print(f"  Samples processed: {metrics['num_samples']}")
        print(f"  Epoch time:        {elapsed_sec:.0f}s ({elapsed_sec / 60:.1f}min)")
        print(f"{'=' * 70}\n", flush=True)


def _enable_gradient_checkpointing(model):
    if hasattr(model, "model") and hasattr(model.model, "gradient_checkpointing_enable"):
        model.model.gradient_checkpointing_enable()
        return True
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        return True
    return False


def train_lora_triplet(args=None):
    args = args or parse_args()
    output_dir = args.output_dir.resolve()
    checkpoint_dir = (args.checkpoint_dir or (output_dir / "checkpoints")).resolve()
    final_adapter_dir = (args.final_adapter_dir or (output_dir / "lora_triplet_adapter")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Initializing config and pipeline.", flush=True)
    config = UIEmbedderConfig.from_model_name(args.model_size, device=args.device)
    pipeline = UIEmbedderPipeline(config)
    device = pipeline.device

    pipeline.box_encoder.eval()
    for param in pipeline.box_encoder.parameters():
        param.requires_grad = False
    pipeline.token_embedding.eval()
    for param in pipeline.token_embedding.parameters():
        param.requires_grad = False

    print("[*] Configuring LoRA for HeadlessQwen2_5.", flush=True)
    if str(device).startswith("cuda") and torch.cuda.is_bf16_supported():
        pipeline.headless_llm = force_bfloat16(pipeline.headless_llm)
        print("  [+] Headless LLM converted to bfloat16", flush=True)
    else:
        pipeline.headless_llm = force_bfloat16(pipeline.headless_llm)
        print("  [+] Headless LLM forced to bfloat16", flush=True)
    pipeline.box_encoder = force_bfloat16(pipeline.box_encoder)
    pipeline.token_embedding = force_bfloat16(pipeline.token_embedding)
    print_floating_dtype_summary("box_encoder", pipeline.box_encoder)
    print_floating_dtype_summary("token_embedding", pipeline.token_embedding)

    pipeline.headless_llm.train()
    if _enable_gradient_checkpointing(pipeline.headless_llm):
        print("  [+] Gradient checkpointing enabled", flush=True)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
    )

    epochs = args.epochs
    lr = args.lr
    gradient_accumulation_steps = args.gradient_accumulation_steps
    triplet_margin = args.triplet_margin

    dataset = TripletUIDataset(data_path_or_list=args.dataset_path, data_root=args.data_root)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda x: x,
        num_workers=args.num_workers,
        pin_memory=str(device).startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )

    start_epoch = 0
    latest_ckpt = find_latest_checkpoint(str(checkpoint_dir))
    if latest_ckpt is not None:
        print(f"[*] Found checkpoint: {latest_ckpt}", flush=True)
        tmp_optimizer = AdamW(pipeline.headless_llm.parameters(), lr=lr, weight_decay=1e-4)
        tmp_scheduler = CosineAnnealingLR(tmp_optimizer, T_max=epochs * len(dataloader))
        pipeline.headless_llm, optimizer, start_epoch, prev_metrics = load_checkpoint(
            latest_ckpt, pipeline.headless_llm, tmp_optimizer, tmp_scheduler, device
        )
        print_floating_dtype_summary("headless_llm", pipeline.headless_llm)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))
        state_path = os.path.join(latest_ckpt, "training_state.pt")
        training_state = torch.load(state_path, map_location=device, weights_only=False)
        if training_state.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(training_state["scheduler_state_dict"])
        if prev_metrics:
            print(f"  [i] Previous loss: {prev_metrics.get('avg_loss', 'N/A')}", flush=True)
        if start_epoch >= epochs:
            print(f"[!] All {epochs} epochs are already complete.", flush=True)
            return
    elif os.path.exists(os.path.join(final_adapter_dir, "adapter_model.safetensors")):
        print(f"[*] Found final adapter, warm-starting: {final_adapter_dir}", flush=True)
        pipeline.headless_llm = PeftModel.from_pretrained(
            pipeline.headless_llm,
            str(final_adapter_dir),
            is_trainable=True,
        )
        pipeline.headless_llm = force_bfloat16(pipeline.headless_llm)
        pipeline.headless_llm.train()
        print_floating_dtype_summary("headless_llm", pipeline.headless_llm)
        optimizer = AdamW(pipeline.headless_llm.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))
    else:
        print("[*] No checkpoint/adapter found. Starting from base weights.", flush=True)
        pipeline.headless_llm = get_peft_model(pipeline.headless_llm, lora_config)
        pipeline.headless_llm = force_bfloat16(pipeline.headless_llm)
        pipeline.headless_llm.print_trainable_parameters()
        print_floating_dtype_summary("headless_llm", pipeline.headless_llm)
        optimizer = AdamW(pipeline.headless_llm.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))

    criterion = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y, dim=-1),
        margin=triplet_margin,
    )

    use_tqdm = args.progress == "tqdm" or (args.progress == "auto" and sys.stderr.isatty())
    use_plain_progress = args.progress in ("auto", "plain")
    progress_mode = "tqdm" if use_tqdm else ("plain" if use_plain_progress else "off")
    print(
        f"[*] Training: epochs={start_epoch + 1}..{epochs}, samples={len(dataset)}, "
        f"grad_accum={gradient_accumulation_steps}, workers={args.num_workers}, progress={progress_mode}",
        flush=True,
    )

    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        metrics = TripletMetrics(margin=triplet_margin)
        optimizer.zero_grad()

        iterator = (
            tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="sample", dynamic_ncols=True)
            if use_tqdm
            else dataloader
        )

        for step, batch in enumerate(iterator):
            item = batch[0]
            seqs, s_prefix, s_box_starts, s_box_ends = prepare_triplet_sequences_for_llm(
                pipeline,
                item["image"],
                item["anchor_text"],
                item["pos_bbox"],
                item["neg_bbox"],
                device,
            )
            triplet_outputs = pipeline.headless_llm(
                seqs,
                s_prefix=s_prefix,
                s_box_starts=s_box_starts,
                s_box_ends=s_box_ends,
            )
            out_anchor = triplet_outputs[:, 0, :]
            out_positive = triplet_outputs[:, 1, :]
            out_negative = triplet_outputs[:, 2, :]

            loss = criterion(out_anchor, out_positive, out_negative) / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step_loss = loss.item() * gradient_accumulation_steps
            metrics.update(out_anchor.detach(), out_positive.detach(), out_negative.detach(), step_loss)

            if metrics.count > 0:
                m = metrics.compute()
                if use_tqdm:
                    iterator.set_postfix(
                        {
                            "loss": f"{m['avg_loss']:.4f}",
                            "acc": f"{m['triplet_accuracy'] * 100:.0f}%",
                            "gap": f"{m['cos_sim_gap']:+.3f}",
                        }
                    )
                elif use_plain_progress and args.log_every > 0 and (
                    (step + 1) % args.log_every == 0 or (step + 1) == len(dataloader)
                ):
                    elapsed = time.time() - epoch_start_time
                    samples_per_sec = (step + 1) / max(elapsed, 1e-9)
                    eta_sec = (len(dataloader) - step - 1) / max(samples_per_sec, 1e-9)
                    print(
                        f"Epoch {epoch + 1}/{epochs} step {step + 1}/{len(dataloader)} "
                        f"loss={m['avg_loss']:.4f} acc={m['triplet_accuracy'] * 100:.0f}% "
                        f"gap={m['cos_sim_gap']:+.3f} {samples_per_sec:.3f} sample/s "
                        f"eta={eta_sec / 3600:.1f}h",
                        flush=True,
                    )

        epoch_elapsed = time.time() - epoch_start_time
        epoch_metrics = metrics.compute()
        TripletMetrics.print_metrics(epoch_metrics, epoch + 1, epochs, epoch_elapsed)
        save_checkpoint(
            pipeline.headless_llm,
            optimizer,
            scheduler,
            epoch,
            epoch_metrics,
            checkpoint_dir=str(checkpoint_dir),
        )

    os.makedirs(final_adapter_dir, exist_ok=True)
    pipeline.headless_llm.save_pretrained(str(final_adapter_dir))
    print(f"[SUCCESS] LoRA adapter saved to: {final_adapter_dir}", flush=True)


if __name__ == "__main__":
    train_lora_triplet()
