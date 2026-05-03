import argparse
import contextlib
import glob
import io
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
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

# Maximum image side length during training.
# ScreenSpot-v2 images reach 2360×1640 → ~19 000 ViT patches → OOM.
# 672px → ~(48×48)/4 = 576 merged tokens per sequence, manageable on ≥8 GB VRAM.
# Override with --max-image-size or env var MAX_TRAINING_IMAGE_SIZE.
MAX_TRAINING_IMAGE_SIZE: int = int(os.environ.get("MAX_TRAINING_IMAGE_SIZE", "1280"))

# ScreenSpot-v2: path to the FiftyOne samples.json in the HF cache
_HF_CACHE_BASE = Path.home() / ".cache" / "huggingface" / "hub"
_SS_SNAPSHOT_BASE = _HF_CACHE_BASE / "datasets--Voxel51--ScreenSpot-v2" / "snapshots"


def _find_screenspot_snapshot() -> Path:
    """Return the latest cached ScreenSpot-v2 snapshot directory."""
    if _SS_SNAPSHOT_BASE.exists():
        snaps = sorted(_SS_SNAPSHOT_BASE.iterdir())
        if snaps:
            return snaps[-1]
    raise FileNotFoundError(
        f"ScreenSpot-v2 snapshot not found in {_SS_SNAPSHOT_BASE}.\n"
        "Run: python training/download_screenspot.py"
    )


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


@contextlib.contextmanager
def suppress_stdout():
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        yield


def optimizer_steps_per_epoch(num_batches, grad_accum_steps):
    return max(1, math.ceil(num_batches / max(1, grad_accum_steps)))


def build_training_scheduler(optimizer, epochs, num_batches, grad_accum_steps):
    total_optimizer_steps = optimizer_steps_per_epoch(num_batches, grad_accum_steps) * max(1, epochs)
    return CosineAnnealingLR(optimizer, T_max=max(1, total_optimizer_steps))


def parse_args():
    output_dir = Path(os.environ.get("TRAINING_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    dataset_path = Path(os.environ.get("TRIPLET_DATASET_PATH", DEFAULT_DATASET_PATH))
    data_root_env = os.environ.get("TRIPLET_DATA_ROOT")

    parser = argparse.ArgumentParser(description="Train LoRA adapter with triplet loss.")
    parser.add_argument(
        "--dataset-type",
        choices=("json", "screenspot"),
        default=os.environ.get("DATASET_TYPE", "screenspot"),
        help="Dataset format: 'json' (legacy triplet_dataset.json) or 'screenspot' (ScreenSpot-v2).",
    )
    parser.add_argument("--dataset-path", type=Path, default=dataset_path)
    parser.add_argument("--data-root", type=Path, default=Path(data_root_env) if data_root_env else None)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--final-adapter-dir", type=Path, default=None)
    parser.add_argument("--model-size", default=os.environ.get("MODEL_SIZE", "2B"))
    parser.add_argument("--device", default=os.environ.get("TRAINING_DEVICE", "cuda"))
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "10")))
    parser.add_argument("--lr", type=float, default=float(os.environ.get("LR", "2e-5")))
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", "4")),
    )
    parser.add_argument("--triplet-margin", type=float, default=float(os.environ.get("TRIPLET_MARGIN", "0.2")))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "0")))
    parser.add_argument("--log-every", type=int, default=int(os.environ.get("LOG_EVERY", "1")))
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=int(os.environ.get("MAX_TRAINING_IMAGE_SIZE", str(MAX_TRAINING_IMAGE_SIZE))),
        help="Cap the longest image side to this many pixels before processing. "
             "Prevents OOM on large screenshots (default: 672).",
    )
    parser.add_argument(
        "--progress",
        choices=("auto", "tqdm", "plain", "off"),
        default=os.environ.get("PROGRESS", "auto"),
        help="Progress output mode. auto uses tqdm only for interactive terminals.",
    )
    parser.add_argument(
        "--screenspot-snapshot",
        type=Path,
        default=None,
        help="Path to ScreenSpot-v2 snapshot dir (auto-detected from HF cache if not set).",
    )
    parser.add_argument(
        "--screenspot-seed",
        type=int,
        default=int(os.environ.get("SCREENSPOT_SEED", "42")),
        help="Random seed for ScreenSpot-v2 negative sampling.",
    )
    return parser.parse_args()


class TripletUIDataset(Dataset):
    """Legacy JSON-based triplet dataset (triplet_dataset.json)."""

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


class ScreenSpotV2Dataset(Dataset):
    """Dataset adapter for ScreenSpot-v2 (Voxel51/ScreenSpot-v2 on HuggingFace).

    The dataset is stored as a FiftyOne export:
      <snapshot>/samples.json  – all annotations
      <snapshot>/data/*.png    – screenshot images

    Triplet construction:
      anchor   = instruction text
      positive = annotated bounding box (action_detection.bounding_box)
      negative = a bbox from a *different* annotation on the same screen
                 (same scene-UUID prefix), falling back to a random bbox
                 from a different image if only one annotation exists per scene.

    Bounding boxes are returned in absolute pixel coords [x1, y1, x2, y2].
    """

    def __init__(self, snapshot_dir=None, seed: int = 42, load_images: bool = True):
        self.load_images = load_images
        self.rng = random.Random(seed)

        # Locate snapshot
        if snapshot_dir is None:
            snap = _find_screenspot_snapshot()
        else:
            snap = Path(snapshot_dir).resolve()
        self.snap = snap
        self.data_dir = snap / "data"

        # Also check for local copy in training/screenspot_v2/data/
        local_data = PROJECT_ROOT / "training" / "screenspot_v2" / "data"
        if local_data.exists() and any(local_data.iterdir()):
            self.data_dir = local_data
            print(f"  [ScreenSpot] Using local image copy: {self.data_dir}", flush=True)

        print(f"  [ScreenSpot] Loading samples.json from {snap} …", flush=True)
        with open(snap / "samples.json", "r", encoding="utf-8") as f:
            raw = json.load(f)
        all_samples = raw.get("samples", [])

        # Parse into records
        records = []
        skipped = 0
        for s in all_samples:
            filepath = s.get("filepath", "")
            instr    = s.get("instruction", "").strip()
            ad       = s.get("action_detection", {})
            meta     = s.get("metadata", {})
            width    = meta.get("width", 0)
            height   = meta.get("height", 0)
            bbox_rel = ad.get("bounding_box")  # [x, y, w, h] in 0..1

            if not instr or not bbox_rel or not filepath or width == 0 or height == 0:
                skipped += 1
                continue

            # Convert to absolute pixel xyxy
            x1 = bbox_rel[0] * width
            y1 = bbox_rel[1] * height
            x2 = (bbox_rel[0] + bbox_rel[2]) * width
            y2 = (bbox_rel[1] + bbox_rel[3]) * height
            bbox_abs = [x1, y1, x2, y2]

            img_name = Path(filepath).name  # e.g. mobile_uuid_0.png
            # scene key = everything before the trailing _<digit(s)>
            stem = img_name.replace(".png", "")
            parts = stem.rsplit("_", 1)
            scene_id = parts[0] if (len(parts) == 2 and parts[1].isdigit()) else stem

            records.append({
                "img_name": img_name,
                "scene_id": scene_id,
                "text":     instr,
                "bbox_abs": bbox_abs,
                "width":    width,
                "height":   height,
            })

        print(f"  [ScreenSpot] Parsed {len(records)} valid records, skipped {skipped}", flush=True)

        # Build scene → [record_index] map for negative sampling
        scene_to_indices: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(records):
            scene_to_indices[r["scene_id"]].append(i)

        # Pre-compute negatives (stable across epochs thanks to fixed seed)
        self.samples = []
        for i, rec in enumerate(records):
            same = [j for j in scene_to_indices[rec["scene_id"]] if j != i]
            if same:
                neg_idx = self.rng.choice(same)
            else:
                other_keys = [k for k in scene_to_indices if k != rec["scene_id"]]
                if not other_keys:
                    continue  # edge case
                neg_scene = self.rng.choice(other_keys)
                neg_idx   = self.rng.choice(scene_to_indices[neg_scene])

            self.samples.append({
                "img_name":    rec["img_name"],
                "anchor_text": rec["text"],
                "pos_bbox":    rec["bbox_abs"],
                "neg_bbox":    records[neg_idx]["bbox_abs"],
            })

        print(f"  [ScreenSpot] Built {len(self.samples)} triplets", flush=True)

    # ------------------------------------------------------------------
    def _load_image(self, img_name: str) -> Image.Image:
        candidates = [
            self.data_dir / img_name,
            self.snap / "data" / img_name,
        ]
        for p in candidates:
            if p.exists():
                try:
                    return Image.open(p).convert("RGB")
                except Exception:
                    pass
        # Fallback blank image
        return Image.new("RGB", (224, 224), (128, 128, 128))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        image = self._load_image(s["img_name"]) if self.load_images else None
        return {
            "image":       image,
            "anchor_text": s["anchor_text"],
            "pos_bbox":    s["pos_bbox"],
            "neg_bbox":    s["neg_bbox"],
            "sample_idx":  idx,
        }


def prepare_triplet_sequences_for_llm(
    pipeline, image, text_content, pos_bbox, neg_bbox, device,
    max_img_size: int = MAX_TRAINING_IMAGE_SIZE,
):
    """Build anchor / visual sequences for the triplet forward pass.

    Args:
        max_img_size: Cap the longest image side to this many pixels *before*
            patch-aligning.  Prevents OOM on large screenshots.
            Bounding boxes are rescaled proportionally.
    """
    # ── 1. Resize image to training budget ──────────────────────────────
    orig_w, orig_h = image.size
    longest = max(orig_w, orig_h)
    if longest > max_img_size:
        scale  = max_img_size / longest
        new_w  = max(int(orig_w * scale), 1)
        new_h  = max(int(orig_h * scale), 1)
        image  = image.resize((new_w, new_h), Image.BICUBIC)
        # Rescale absolute-pixel bboxes to match the resized image
        pos_bbox = [c * scale for c in pos_bbox]
        neg_bbox = [c * scale for c in neg_bbox]
    else:
        scale = 1.0

    with suppress_stdout():
        image = smart_resize(image, patch_size=pipeline.config.patch_size_resize)

    resized_w, resized_h = image.size

    img_tensor   = pipeline.transform(image).unsqueeze(0).to(device=device, dtype=TRAINING_DTYPE)
    visual_boxes = [pos_bbox, neg_bbox]
    boxes_tensor = torch.tensor([visual_boxes], device=device, dtype=TRAINING_DTYPE)

    with torch.no_grad():
        with suppress_stdout():
            g_seq, b_seqs = pipeline.box_encoder(img_tensor, boxes_tensor)
        g_summary      = g_seq.mean(dim=1, keepdim=True) if pipeline.config.use_global_summary else None
        visual_s_prefix = 1 if g_summary is not None else 0

    # Free raw image tensor immediately — not needed further
    del img_tensor, boxes_tensor

    with torch.no_grad():
        with suppress_stdout():
            anchor_text_emb = pipeline._prepare_text_embeddings(text_content, bbox=None)
    anchor_seq = anchor_text_emb.detach().requires_grad_(True)
    del anchor_text_emb

    visual_seqs       = []
    visual_box_starts = []
    visual_box_ends   = []
    for b_seq, text_bbox in zip(b_seqs, visual_boxes):
        with torch.no_grad():
            with suppress_stdout():
                text_emb_box = pipeline._prepare_text_embeddings(text_content, bbox=text_bbox)
        box_start = visual_s_prefix
        box_end   = visual_s_prefix + b_seq.shape[1]
        if g_summary is not None:
            combined_seq = torch.cat(
                [g_summary.detach(), b_seq.detach(), text_emb_box.detach()], dim=1,
            )
        else:
            combined_seq = torch.cat([b_seq.detach(), text_emb_box.detach()], dim=1)
        del text_emb_box
        visual_seqs.append(combined_seq.detach().requires_grad_(True))
        visual_box_starts.append(box_start)
        visual_box_ends.append(box_end)

    del g_seq, b_seqs, g_summary

    return anchor_seq, visual_seqs, visual_s_prefix, visual_box_starts, visual_box_ends


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


def load_checkpoint(checkpoint_path, base_model, lr, weight_decay, scheduler_t_max, device):
    adapter_path = os.path.join(checkpoint_path, "lora_adapter")
    state_path = os.path.join(checkpoint_path, "training_state.pt")
    model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=True)
    force_bfloat16(model)
    model.train()

    training_state = torch.load(state_path, map_location=device, weights_only=False)
    new_optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    new_optimizer.load_state_dict(training_state["optimizer_state_dict"])

    new_scheduler = CosineAnnealingLR(new_optimizer, T_max=max(1, scheduler_t_max))
    if training_state.get("scheduler_state_dict") is not None:
        new_scheduler.load_state_dict(training_state["scheduler_state_dict"])

    start_epoch = training_state["epoch"] + 1
    prev_metrics = training_state.get("metrics", {})
    return model, new_optimizer, new_scheduler, start_epoch, prev_metrics


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
    weight_decay = 1e-4

    # -----------------------------------------------------------------------
    # Dataset selection
    # -----------------------------------------------------------------------
    if args.dataset_type == "screenspot":
        print("[*] Using ScreenSpot-v2 dataset.", flush=True)
        snapshot_dir = args.screenspot_snapshot or None
        dataset = ScreenSpotV2Dataset(
            snapshot_dir=snapshot_dir,
            seed=args.screenspot_seed,
        )
    else:
        print(f"[*] Using JSON triplet dataset: {args.dataset_path}", flush=True)
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
    scheduler_t_max = optimizer_steps_per_epoch(len(dataloader), gradient_accumulation_steps) * max(1, epochs)
    latest_ckpt = find_latest_checkpoint(str(checkpoint_dir))
    if latest_ckpt is not None:
        print(f"[*] Found checkpoint: {latest_ckpt}", flush=True)
        pipeline.headless_llm, optimizer, scheduler, start_epoch, prev_metrics = load_checkpoint(
            latest_ckpt,
            pipeline.headless_llm,
            lr,
            weight_decay,
            scheduler_t_max,
            device,
        )
        print_floating_dtype_summary("headless_llm", pipeline.headless_llm)
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
        optimizer = AdamW(pipeline.headless_llm.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = build_training_scheduler(optimizer, epochs, len(dataloader), gradient_accumulation_steps)
    else:
        print("[*] No checkpoint/adapter found. Starting from base weights.", flush=True)
        pipeline.headless_llm = get_peft_model(pipeline.headless_llm, lora_config)
        pipeline.headless_llm = force_bfloat16(pipeline.headless_llm)
        pipeline.headless_llm.print_trainable_parameters()
        print_floating_dtype_summary("headless_llm", pipeline.headless_llm)
        optimizer = AdamW(pipeline.headless_llm.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = build_training_scheduler(optimizer, epochs, len(dataloader), gradient_accumulation_steps)

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
        step_times: list = []
        metrics = TripletMetrics(margin=triplet_margin)
        optimizer.zero_grad()

        print(f"\n[*] Epoch {epoch + 1}/{epochs} — {len(dataloader)} steps", flush=True)

        iterator = (
            tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="sample", dynamic_ncols=True)
            if use_tqdm
            else dataloader
        )

        for step, batch in enumerate(iterator):
            step_t0 = time.time()
            item = batch[0]

            # ── per-step header (plain/off modes only) ───────────────────
            if not use_tqdm and args.progress != "off":
                elapsed_so_far = step_t0 - epoch_start_time
                avg_step_prev  = (elapsed_so_far / step) if step > 0 else 0.0
                eta_sec_prev   = avg_step_prev * (len(dataloader) - step)
                eta_str        = f"eta={eta_sec_prev/3600:.2f}h" if step > 0 else "eta=?"
                instr_preview  = item["anchor_text"][:45].replace("\n", " ")
                print(
                    f"\n  >> step {step + 1}/{len(dataloader)} "
                    f"(elapsed={elapsed_so_far:.0f}s {eta_str})"
                    f"\n     text: \"{instr_preview}\"",
                    flush=True,
                )

            # ── prepare sequences ────────────────────────────────────────
            t0 = time.time()
            anchor_seq, visual_seqs, visual_s_prefix, visual_box_starts, visual_box_ends = prepare_triplet_sequences_for_llm(
                pipeline,
                item["image"],
                item["anchor_text"],
                item["pos_bbox"],
                item["neg_bbox"],
                device,
                max_img_size=args.max_image_size,
            )
            t_prepare = time.time() - t0
            if not use_tqdm and args.progress != "off":
                n_vis_tok = sum(s.shape[1] for s in visual_seqs)
                print(f"     prepare={t_prepare:.2f}s  vis_tok={n_vis_tok}", end="", flush=True)

            # ── anchor forward ───────────────────────────────────────────
            t0 = time.time()
            anchor_outputs = pipeline.headless_llm([anchor_seq], s_prefix=0)
            t_anchor = time.time() - t0
            if not use_tqdm and args.progress != "off":
                print(f"  anchor_fwd={t_anchor:.2f}s", end="", flush=True)

            # ── visual forward ───────────────────────────────────────────
            t0 = time.time()
            visual_outputs = pipeline.headless_llm(
                visual_seqs,
                s_prefix=visual_s_prefix,
                s_box_starts=visual_box_starts,
                s_box_ends=visual_box_ends,
            )
            t_visual = time.time() - t0
            if not use_tqdm and args.progress != "off":
                print(f"  visual_fwd={t_visual:.2f}s", end="", flush=True)

            out_anchor   = F.normalize(anchor_outputs[:, 0, :], dim=-1)
            out_positive = F.normalize(visual_outputs[:, 0, :], dim=-1)
            out_negative = F.normalize(visual_outputs[:, 1, :], dim=-1)

            # Free large activation tensors before backward
            del anchor_outputs, visual_outputs
            del anchor_seq, visual_seqs

            # ── loss + backward ──────────────────────────────────────────
            t0 = time.time()
            loss = criterion(out_anchor, out_positive, out_negative) / gradient_accumulation_steps
            loss.backward()
            t_bwd = time.time() - t0
            if not use_tqdm and args.progress != "off":
                print(f"  bwd={t_bwd:.2f}s", flush=True)

            # ── optimizer step ───────────────────────────────────────────
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            step_loss = loss.item() * gradient_accumulation_steps
            del loss  # free computation graph immediately

            step_dt = time.time() - step_t0
            # Cap history to last 50 steps to bound memory usage
            step_times.append(step_dt)
            if len(step_times) > 50:
                step_times.pop(0)

            metrics.update(out_anchor.detach(), out_positive.detach(), out_negative.detach(), step_loss)
            del out_anchor, out_positive, out_negative

            # ── explicit CUDA memory release ─────────────────────────────
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

            if metrics.count > 0:
                m = metrics.compute()
                avg_step_t = sum(step_times) / len(step_times)
                if use_tqdm:
                    mem_str = ""
                    if str(device).startswith("cuda"):
                        used = torch.cuda.memory_reserved() / 1024 ** 3
                        mem_str = f"{used:.1f}GB"
                    iterator.set_postfix(
                        {
                            "loss":  f"{m['avg_loss']:.4f}",
                            "acc":   f"{m['triplet_accuracy'] * 100:.0f}%",
                            "gap":   f"{m['cos_sim_gap']:+.3f}",
                            "s/it":  f"{avg_step_t:.1f}s",
                            "vram":  mem_str,
                        }
                    )
                elif use_plain_progress and args.log_every > 0 and (
                    (step + 1) % args.log_every == 0 or (step + 1) == len(dataloader)
                ):
                    elapsed = time.time() - epoch_start_time
                    eta_sec = avg_step_t * (len(dataloader) - step - 1)
                    cos_p   = m["avg_cos_sim_positive"]
                    cos_n   = m["avg_cos_sim_negative"]
                    mem_str = ""
                    if str(device).startswith("cuda"):
                        used_gb  = torch.cuda.memory_reserved() / 1024 ** 3
                        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                        mem_str  = f"  vram={used_gb:.1f}/{total_gb:.1f}GB"
                    print(
                        f"  ┌ Ep {epoch+1}/{epochs}  step {step+1}/{len(dataloader)}{mem_str}\n"
                        f"  │ loss={m['avg_loss']:.5f}  acc={m['triplet_accuracy']*100:.1f}%  "
                        f"gap={m['cos_sim_gap']:+.4f}  viol={m['margin_violation_rate']*100:.1f}%\n"
                        f"  │ cos(+)={cos_p:.4f}  cos(-)={cos_n:.4f}\n"
                        f"  └ {avg_step_t:.1f}s/step  elapsed={elapsed:.0f}s  eta={eta_sec/3600:.2f}h",
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
