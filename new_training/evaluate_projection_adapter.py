from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from adapter_model import VectorProjectionAdapter
from build_triplet_dataset import load_user_test_cases, text_match
from train_3b_projection_adapter import (
    OUTPUT_DIR,
    UIEmbedderConfig,
    UIEmbedderPipeline,
    VISUAL_CACHE_PATH,
    VisualSpec,
    encode_visual_specs,
    load_jsonl,
    load_sentence_transformer,
    load_visual_cache,
    resolve_device,
    resolve_existing_path,
    save_visual_cache,
    stable_cache_key,
)


UI_ITEMS_PATH = ROOT_DIR / "generated" / "ui_index" / "ui_items.jsonl"
SCENARIO_FILE = PROJECT_ROOT.parent / "visual-language-two-tower-kristina" / "evaluation" / "user_scenario_eval.py"
DEFAULT_ADAPTER_PATH = OUTPUT_DIR / "best_adapter.pt"
DEFAULT_REPORT_PATH = ROOT_DIR / "output" / "projection_adapter_eval.json"


def item_to_spec(item: dict[str, Any]) -> VisualSpec:
    image_path = resolve_existing_path(item.get("screenshot_image"))
    bbox = tuple(float(value) for value in item.get("bbox", []))
    if len(bbox) != 4:
        raise ValueError(f"Bad bbox for item {item.get('id')}")
    context = str(item.get("context_text") or item.get("text") or "")
    return VisualSpec(str(image_path), bbox, context)


def item_is_relevant(case: dict[str, Any], item: dict[str, Any]) -> bool:
    expected_pdf = case.get("expected_pdf")
    if expected_pdf and str(item.get("pdf_name", "")).lower() != Path(expected_pdf).name.lower():
        return False
    expected = [str(value) for value in case.get("expected", [])]
    text = str(item.get("normalized_text") or item.get("text") or "")
    return any(text_match(pattern, text) for pattern in expected)


def load_adapter(path: Path, device: torch.device) -> tuple[VectorProjectionAdapter, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    adapter = VectorProjectionAdapter(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        hidden_dim=checkpoint.get("hidden_dim"),
        dropout=float(checkpoint.get("dropout", 0.0)),
    ).to(device)
    adapter.load_state_dict(checkpoint["adapter_state_dict"])
    adapter.eval()
    return adapter, checkpoint


def project_candidates(
    *,
    adapter: VectorProjectionAdapter,
    candidate_vectors: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dataset = TensorDataset(candidate_vectors.float())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    projected: list[torch.Tensor] = []

    with torch.no_grad():
        for (batch,) in tqdm(loader, desc="Projecting candidates"):
            batch = batch.to(device)
            projected.append(adapter(batch).cpu())

    return F.normalize(torch.cat(projected, dim=0), dim=-1).numpy()


def reciprocal_rank(order: np.ndarray, relevant: set[int]) -> float:
    for rank, idx in enumerate(order, start=1):
        if int(idx) in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_cases(
    *,
    cases: list[dict[str, Any]],
    items: list[dict[str, Any]],
    candidate_matrix: np.ndarray,
    query_matrix: np.ndarray,
    top_k: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []

    for case, query_vector in zip(cases, query_matrix):
        scores = candidate_matrix @ query_vector
        order = np.argsort(-scores)
        top = order[:top_k]
        relevant = {idx for idx, item in enumerate(items) if item_is_relevant(case, item)}

        if relevant:
            hits = [idx for idx in top if int(idx) in relevant]
            precision = len(hits) / max(1, top_k)
            recall = len(hits) / max(1, len(relevant))
            f1 = 2 * precision * recall / max(1e-12, precision + recall)
        else:
            hits = []
            precision = 0.0
            recall = 0.0
            f1 = 0.0

        top1 = items[int(order[0])]
        expected_pdf = case.get("expected_pdf")
        pdf_ok = 1.0 if expected_pdf and top1.get("pdf_name") == expected_pdf else 0.0

        details.append(
            {
                "query": case.get("query"),
                "expected": case.get("expected"),
                "expected_pdf": expected_pdf,
                "relevant_total": len(relevant),
                "success": 1.0 if hits else 0.0,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mrr": reciprocal_rank(order, relevant),
                "hit@1": 1.0 if int(order[0]) in relevant else 0.0,
                "hit@3": 1.0 if any(int(idx) in relevant for idx in order[:3]) else 0.0,
                "hit@5": 1.0 if any(int(idx) in relevant for idx in order[:5]) else 0.0,
                "pdf_accuracy": pdf_ok,
                "top": [
                    {
                        "rank": rank,
                        "score": float(scores[int(idx)]),
                        "id": items[int(idx)].get("id"),
                        "text": items[int(idx)].get("text"),
                        "pdf_name": items[int(idx)].get("pdf_name"),
                        "page": items[int(idx)].get("page"),
                    }
                    for rank, idx in enumerate(top, start=1)
                ],
            }
        )

    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in details])) if details else 0.0

    summary = {
        "Success Rate": mean("success"),
        "Precision": mean("precision"),
        "Recall": mean("recall"),
        "F1-score": mean("f1"),
        "MRR": mean("mrr"),
        "Hit@1": mean("hit@1"),
        "Hit@3": mean("hit@3"),
        "Hit@5": mean("hit@5"),
        "PDF Accuracy": mean("pdf_accuracy"),
    }
    return summary, details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval with the trained projection adapter.")
    parser.add_argument("--ui-items", type=Path, default=UI_ITEMS_PATH)
    parser.add_argument("--scenario-file", type=Path, default=SCENARIO_FILE)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--visual-cache", type=Path, default=VISUAL_CACHE_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--model-size", default="3B")
    parser.add_argument("--bert-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bert-device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-image-side", type=int, default=1280)
    parser.add_argument("--max-token-length", type=int, default=512)
    parser.add_argument("--refresh-visual-cache", action="store_true")
    parser.add_argument("--save-cache-every", type=int, default=100)
    parser.add_argument("--suppress-encoder-stdout", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    bert_device = args.bert_device or "cpu"

    items = load_jsonl(args.ui_items)
    cases = load_user_test_cases(args.scenario_file)
    adapter, checkpoint = load_adapter(args.adapter, device)

    print(f"[DATA] candidates: {len(items)}")
    print(f"[DATA] cases: {len(cases)}")
    print(f"[ADAPTER] {args.adapter}")

    config = UIEmbedderConfig.from_model_name(args.model_size, device=str(device))
    config.debug_decode_embeddings = False
    config.max_token_length = args.max_token_length
    pipeline = UIEmbedderPipeline(config)

    specs = [item_to_spec(item) for item in items]
    visual_cache = {} if args.refresh_visual_cache else load_visual_cache(args.visual_cache)
    visual_cache = encode_visual_specs(
        pipeline=pipeline,
        specs=specs,
        cache=visual_cache,
        args=args,
    )
    save_visual_cache(
        args.visual_cache,
        visual_cache,
        {
            "model_size": args.model_size,
            "max_image_side": args.max_image_side,
            "candidate_specs": len(specs),
        },
    )

    vectors = []
    for spec in specs:
        key = stable_cache_key(spec, model_size=args.model_size, max_image_side=args.max_image_side)
        vectors.append(visual_cache[key])
    candidate_vectors = torch.stack(vectors)
    candidate_matrix = project_candidates(
        adapter=adapter,
        candidate_vectors=candidate_vectors,
        batch_size=args.batch_size,
        device=device,
    )

    query_encoder = load_sentence_transformer(
        checkpoint.get("bert_model", args.bert_model),
        bert_device,
    )
    query_matrix = query_encoder.encode(
        [str(case.get("query", "")) for case in cases],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    query_matrix = np.asarray(query_matrix, dtype=np.float32)

    summary, details = evaluate_cases(
        cases=cases,
        items=items,
        candidate_matrix=candidate_matrix,
        query_matrix=query_matrix,
        top_k=args.top_k,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": details}, f, ensure_ascii=False, indent=2)

    print("\nRetrieval metrics")
    print("| Metric | Value |")
    print("|---|---:|")
    for key, value in summary.items():
        print(f"| {key} | {value:.4f} |")
    print(f"\n[OK] report saved: {args.out}")


if __name__ == "__main__":
    main()
