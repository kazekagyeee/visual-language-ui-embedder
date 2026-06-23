from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
TRIPLETS_PATH = ROOT_DIR / "generated" / "triplets.jsonl"
OUTPUT_DIR = ROOT_DIR / "output" / "projection_adapter_3b"
VISUAL_CACHE_PATH = ROOT_DIR / "generated" / "qwen3b_visual_cache.pt"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapter_model import VectorProjectionAdapter
from config import UIEmbedderConfig
from main import UIEmbedderPipeline


@dataclass(frozen=True)
class VisualSpec:
    image_path: str
    bbox: tuple[float, float, float, float]
    context: str


class VectorTripletDataset(Dataset):
    def __init__(self, query_vectors: torch.Tensor, pos_vectors: torch.Tensor, neg_vectors: torch.Tensor) -> None:
        self.query_vectors = query_vectors.float()
        self.pos_vectors = pos_vectors.float()
        self.neg_vectors = neg_vectors.float()

    def __len__(self) -> int:
        return self.query_vectors.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.query_vectors[idx], self.pos_vectors[idx], self.neg_vectors[idx]


@contextlib.contextmanager
def suppress_stdout(enabled: bool = True):
    if not enabled:
        yield
        return

    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        yield


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_existing_path(path_value: str | None) -> Path:
    if not path_value:
        raise ValueError("Empty image path in triplet row")

    raw = Path(path_value)
    candidates = [raw] if raw.is_absolute() else []
    if not raw.is_absolute():
        candidates.extend(
            [
                ROOT_DIR / raw,
                PROJECT_ROOT / raw,
                Path.cwd() / raw,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (PROJECT_ROOT / raw).resolve() if not raw.is_absolute() else raw


def normalize_bbox_for_image(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
    min_size: float = 1e-4,
) -> tuple[float, float, float, float]:
    width, height = image_size
    x1, y1, x2, y2 = [float(value) for value in bbox]

    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.5:
        x1 /= max(1, width)
        x2 /= max(1, width)
        y1 /= max(1, height)
        y2 /= max(1, height)

    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))

    if x2 - x1 < min_size:
        x2 = min(1.0, x1 + min_size)
    if y2 - y1 < min_size:
        y2 = min(1.0, y1 + min_size)

    return (x1, y1, x2, y2)


def resize_to_max_side(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return image

    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image

    scale = max_side / longest
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.BICUBIC)


def stable_cache_key(spec: VisualSpec, *, model_size: str, max_image_side: int) -> str:
    payload = {
        "image_path": spec.image_path,
        "bbox": [round(value, 6) for value in spec.bbox],
        "context_sha1": hashlib.sha1(spec.context.encode("utf-8")).hexdigest(),
        "model_size": model_size,
        "max_image_side": max_image_side,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def triplet_to_specs(row: dict[str, Any]) -> tuple[VisualSpec, VisualSpec]:
    pos_path = resolve_existing_path(row.get("pos_image_path"))
    neg_path = resolve_existing_path(row.get("neg_image_path"))
    pos_bbox = tuple(float(value) for value in row.get("pos_bbox", []))
    neg_bbox = tuple(float(value) for value in row.get("neg_bbox", []))

    if len(pos_bbox) != 4 or len(neg_bbox) != 4:
        raise ValueError(f"Bad bbox in row: {row.get('query')}")

    pos_context = str(row.get("pos_context") or row.get("pos_text") or "")
    neg_context = str(row.get("neg_context") or row.get("neg_text") or "")

    return (
        VisualSpec(str(pos_path), pos_bbox, pos_context),
        VisualSpec(str(neg_path), neg_bbox, neg_context),
    )


def load_visual_cache(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        return {}

    data = torch.load(path, map_location="cpu", weights_only=False)
    vectors = data.get("vectors", data) if isinstance(data, dict) else data
    return {str(key): value.cpu().float() for key, value in dict(vectors).items()}


def save_visual_cache(path: Path, vectors: dict[str, torch.Tensor], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "meta": meta,
            "vectors": {key: value.cpu().float() for key, value in vectors.items()},
        },
        path,
    )


def encode_visual_specs(
    *,
    pipeline: UIEmbedderPipeline,
    specs: list[VisualSpec],
    cache: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    missing_by_group: dict[tuple[str, str], list[tuple[str, VisualSpec]]] = defaultdict(list)

    for spec in specs:
        key = stable_cache_key(spec, model_size=args.model_size, max_image_side=args.max_image_side)
        if args.refresh_visual_cache or key not in cache:
            missing_by_group[(spec.image_path, spec.context)].append((key, spec))

    if not missing_by_group:
        return cache

    iterator = tqdm(missing_by_group.items(), desc="Encoding visual vectors")
    for (image_path, context), entries in iterator:
        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"Image not found: {image_file}")

        with Image.open(image_file) as img:
            image = img.convert("RGB")
            original_size = image.size
            normalized_bboxes = [
                normalize_bbox_for_image(entry_spec.bbox, original_size)
                for _, entry_spec in entries
            ]
            image = resize_to_max_side(image, args.max_image_side)

            with torch.no_grad(), suppress_stdout(args.suppress_encoder_stdout):
                encoded = pipeline.process(
                    image=image,
                    text_content=context,
                    bboxes=[list(bbox) for bbox in normalized_bboxes],
                )

        if not isinstance(encoded, dict):
            raise RuntimeError(f"Encoder returned unexpected type for {image_file}: {type(encoded)!r}")

        by_bbox = {
            tuple(round(float(coord), 6) for coord in bbox): torch.as_tensor(vector, dtype=torch.float32)
            for bbox, vector in encoded.items()
        }

        for (key, spec), normalized_bbox in zip(entries, normalized_bboxes):
            rounded_bbox = tuple(round(float(coord), 6) for coord in normalized_bbox)
            vector = by_bbox.get(rounded_bbox)
            if vector is None:
                available = list(by_bbox.keys())[:3]
                raise RuntimeError(
                    f"Encoder did not return bbox {rounded_bbox} for {spec.image_path}. "
                    f"Available examples: {available}"
                )
            cache[key] = F.normalize(vector.cpu().float(), dim=0)

        if args.save_cache_every > 0 and len(cache) % args.save_cache_every < len(entries):
            save_visual_cache(
                args.visual_cache,
                cache,
                {
                    "model_size": args.model_size,
                    "max_image_side": args.max_image_side,
                },
            )

    return cache


def build_vector_dataset(
    *,
    triplets: list[dict[str, Any]],
    visual_cache: dict[str, torch.Tensor],
    query_vectors: np.ndarray,
    args: argparse.Namespace,
) -> VectorTripletDataset:
    query_tensors: list[torch.Tensor] = []
    pos_tensors: list[torch.Tensor] = []
    neg_tensors: list[torch.Tensor] = []

    for row, query_vector in zip(triplets, query_vectors):
        pos_spec, neg_spec = triplet_to_specs(row)
        pos_key = stable_cache_key(pos_spec, model_size=args.model_size, max_image_side=args.max_image_side)
        neg_key = stable_cache_key(neg_spec, model_size=args.model_size, max_image_side=args.max_image_side)

        query_tensors.append(torch.as_tensor(query_vector, dtype=torch.float32))
        pos_tensors.append(visual_cache[pos_key])
        neg_tensors.append(visual_cache[neg_key])

    return VectorTripletDataset(
        F.normalize(torch.stack(query_tensors), dim=-1),
        F.normalize(torch.stack(pos_tensors), dim=-1),
        F.normalize(torch.stack(neg_tensors), dim=-1),
    )


def split_dataset(dataset: Dataset, val_ratio: float, seed: int) -> tuple[Dataset, Dataset | None]:
    if val_ratio <= 0 or len(dataset) < 10:
        return dataset, None

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    val_size = max(1, int(round(len(indices) * val_ratio)))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def cosine_triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    margin: float,
    positive_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    cos_pos = F.cosine_similarity(anchor, positive, dim=-1)
    cos_neg = F.cosine_similarity(anchor, negative, dim=-1)
    triplet = F.relu(margin + cos_neg - cos_pos).mean()
    positive_alignment = (1.0 - cos_pos).mean()
    loss = triplet + positive_weight * positive_alignment

    metrics = {
        "loss": float(loss.detach().cpu()),
        "triplet_loss": float(triplet.detach().cpu()),
        "positive_loss": float(positive_alignment.detach().cpu()),
        "acc": float((cos_pos > cos_neg).float().mean().detach().cpu()),
        "cos_pos": float(cos_pos.mean().detach().cpu()),
        "cos_neg": float(cos_neg.mean().detach().cpu()),
        "gap": float((cos_pos - cos_neg).mean().detach().cpu()),
    }
    return loss, metrics


def run_epoch(
    *,
    adapter: VectorProjectionAdapter,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    margin: float,
    positive_weight: float,
) -> dict[str, float]:
    is_train = optimizer is not None
    adapter.train(is_train)

    totals: dict[str, float] = defaultdict(float)
    n_batches = 0

    for query, pos, neg in loader:
        query = query.to(device)
        pos = pos.to(device)
        neg = neg.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            pos_projected = adapter(pos)
            neg_projected = adapter(neg)
            loss, metrics = cosine_triplet_loss(
                query,
                pos_projected,
                neg_projected,
                margin=margin,
                positive_weight=positive_weight,
            )

            if is_train:
                loss.backward()
                optimizer.step()

        for key, value in metrics.items():
            totals[key] += value
        n_batches += 1

    return {key: value / max(1, n_batches) for key, value in totals.items()}


def save_adapter_checkpoint(
    *,
    output_dir: Path,
    filename: str,
    adapter: VectorProjectionAdapter,
    args: argparse.Namespace,
    input_dim: int,
    output_dim: int,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter_state_dict": adapter.state_dict(),
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "model_size": args.model_size,
            "bert_model": args.bert_model,
            "epoch": epoch,
            "metrics": metrics,
        },
        output_dir / filename,
    )


def load_sentence_transformer(model_name: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for the BERT query encoder. Install:\n"
            r".\.venv\Scripts\python.exe -m pip install -r new_training\requirements.txt"
        ) from exc

    return SentenceTransformer(model_name, device=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 3B UI encoder projection adapter into BERT vector space.")
    parser.add_argument("--triplets", type=Path, default=TRIPLETS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--visual-cache", type=Path, default=VISUAL_CACHE_PATH)
    parser.add_argument("--model-size", default="3B")
    parser.add_argument("--bert-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--device", default="auto", help="auto, cuda, cpu, or cuda:0")
    parser.add_argument("--bert-device", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--positive-weight", type=float, default=0.25)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-image-side", type=int, default=1280)
    parser.add_argument("--max-token-length", type=int, default=512)
    parser.add_argument("--refresh-visual-cache", action="store_true")
    parser.add_argument("--save-cache-every", type=int, default=100)
    parser.add_argument("--suppress-encoder-stdout", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device(value)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    bert_device = args.bert_device or "cpu"

    triplets = load_jsonl(args.triplets)
    if args.max_samples:
        triplets = triplets[: args.max_samples]
    if not triplets:
        raise RuntimeError(f"No triplets found: {args.triplets}")

    print(f"[DATA] triplets: {len(triplets)}")
    print(f"[MODEL] UI encoder: {args.model_size} on {device}")
    print(f"[MODEL] BERT query encoder: {args.bert_model} on {bert_device}")

    query_encoder = load_sentence_transformer(args.bert_model, bert_device)
    query_dim = int(query_encoder.get_sentence_embedding_dimension())

    config = UIEmbedderConfig.from_model_name(args.model_size, device=str(device))
    config.debug_decode_embeddings = False
    config.max_token_length = args.max_token_length

    pipeline = UIEmbedderPipeline(config)
    pipeline.box_encoder.eval()
    pipeline.headless_llm.eval()
    pipeline.token_embedding.eval()
    for module in (pipeline.box_encoder, pipeline.headless_llm, pipeline.token_embedding):
        for param in module.parameters():
            param.requires_grad_(False)

    visual_specs: list[VisualSpec] = []
    for row in triplets:
        pos_spec, neg_spec = triplet_to_specs(row)
        visual_specs.extend([pos_spec, neg_spec])
    visual_specs = list(dict.fromkeys(visual_specs))

    visual_cache = {} if args.refresh_visual_cache else load_visual_cache(args.visual_cache)
    visual_cache = encode_visual_specs(
        pipeline=pipeline,
        specs=visual_specs,
        cache=visual_cache,
        args=args,
    )
    save_visual_cache(
        args.visual_cache,
        visual_cache,
        {
            "model_size": args.model_size,
            "max_image_side": args.max_image_side,
            "unique_visual_specs": len(visual_specs),
        },
    )

    queries = [str(row.get("query", "")) for row in triplets]
    query_vectors = query_encoder.encode(
        queries,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    query_vectors = np.asarray(query_vectors, dtype=np.float32)

    dataset = build_vector_dataset(
        triplets=triplets,
        visual_cache=visual_cache,
        query_vectors=query_vectors,
        args=args,
    )
    input_dim = int(dataset.pos_vectors.shape[-1])
    output_dim = query_dim
    print(f"[DIM] adapter: {input_dim} -> {output_dim}")

    train_dataset, val_dataset = split_dataset(dataset, args.val_ratio, args.seed)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)
        if val_dataset is not None
        else None
    )

    adapter = VectorProjectionAdapter(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, Any]] = []
    best_score = -math.inf

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            adapter=adapter,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            margin=args.margin,
            positive_weight=args.positive_weight,
        )
        val_metrics = None
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(
                    adapter=adapter,
                    loader=val_loader,
                    optimizer=None,
                    device=device,
                    margin=args.margin,
                    positive_weight=args.positive_weight,
                )

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_record)

        score_metrics = val_metrics or train_metrics
        score = float(score_metrics.get("acc", 0.0)) + float(score_metrics.get("gap", 0.0))
        if score > best_score:
            best_score = score
            save_adapter_checkpoint(
                output_dir=args.output_dir,
                filename="best_adapter.pt",
                adapter=adapter,
                args=args,
                input_dim=input_dim,
                output_dim=output_dim,
                epoch=epoch,
                metrics=epoch_record,
            )

        val_part = ""
        if val_metrics:
            val_part = (
                f" | val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['acc']:.4f} val_gap={val_metrics['gap']:.4f}"
            )
        print(
            f"[EPOCH {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['acc']:.4f} "
            f"train_gap={train_metrics['gap']:.4f}"
            f"{val_part}"
        )

    save_adapter_checkpoint(
        output_dir=args.output_dir,
        filename="adapter.pt",
        adapter=adapter,
        args=args,
        input_dim=input_dim,
        output_dim=output_dim,
        epoch=args.epochs,
        metrics=history[-1],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with (args.output_dir / "training_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2, default=str)

    print(f"[OK] saved final adapter: {args.output_dir / 'adapter.pt'}")
    print(f"[OK] saved best adapter: {args.output_dir / 'best_adapter.pt'}")


if __name__ == "__main__":
    main()
