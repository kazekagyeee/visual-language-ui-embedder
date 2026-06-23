from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Main PDF input. Change this variable when you want to evaluate another
# folder with several PDF files. By default it points to the project that
# contains user_scenario_eval.py and its two manual PDFs.
SCENARIO_PROJECT = (PROJECT_ROOT.parent / "visual-language-two-tower-kristina").resolve()
PDF_FOLDER = (SCENARIO_PROJECT / "data_source").resolve()

# Query labels for PDF mode. The file can be a list or {"queries": [...]}.
# Minimal query item:
#   {"query": "...", "expected_pdf": "manual_1.pdf"}
# Better component-level item:
#   {"query": "...", "expected_pdf": "manual_1.pdf", "expected_page": 12,
#    "expected_bbox": [0.1, 0.2, 0.4, 0.3]}
CONTROL_QUERIES_PATH = PROJECT_ROOT / "output" / "pdf_control_queries.json"
USER_SCENARIO_EVAL_PATH = SCENARIO_PROJECT / "evaluation" / "user_scenario_eval.py"

# Extracted PDF images and sidecar context files are written here.
PDF_EXTRACT_DIR = PROJECT_ROOT / "output" / "pdf_metric_extracted"

NEIGHBOR_PROJECT = (PROJECT_ROOT.parent / "fffaffafaf").resolve()
DEFAULT_DATASET = PROJECT_ROOT / "training" / "synthetic_dataset" / "triplet_dataset_clean.json"
DEFAULT_CACHE = PROJECT_ROOT / "output" / "encoder_retrieval_cache.npz"
DEFAULT_JSON_REPORT = PROJECT_ROOT / "output" / "encoder_retrieval_eval.json"
DEFAULT_CSV_REPORT = PROJECT_ROOT / "output" / "encoder_retrieval_eval_queries.csv"

DEFAULT_IMAGE_CONTEXT_CHARS = 1800


@dataclass(frozen=True)
class Candidate:
    id: str
    image_path: str
    bbox: tuple[float, float, float, float]
    pdf_id: str | None = None
    text: str | None = None
    page_num: int | None = None
    image_index: int | None = None
    source_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryCase:
    id: str
    text: str
    relevant_ids: set[str]
    expected_pdf: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalData:
    queries: list[QueryCase]
    candidates: list[Candidate]


@dataclass(frozen=True)
class PdfImage:
    image_path: str
    text_path: str
    context_text: str
    nearby_text: str
    pdf_id: str
    source_name: str
    page_num: int
    page_index: int
    image_index: int
    xref: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate UI encoder retrieval metrics. Default mode extracts GUI images "
            "from PDF files using the PDF-processing logic adapted from ../fffaffafaf."
        )
    )
    parser.add_argument("--mode", choices=("pdf", "dataset"), default="pdf")

    # PDF mode.
    parser.add_argument("--pdf-dir", type=Path, default=PDF_FOLDER)
    parser.add_argument("--queries", type=Path, default=CONTROL_QUERIES_PATH)
    parser.add_argument("--scenario-file", type=Path, default=USER_SCENARIO_EVAL_PATH)
    parser.add_argument("--pdf-extract-dir", type=Path, default=PDF_EXTRACT_DIR)
    parser.add_argument("--skip-first-images", type=int, default=0)
    parser.add_argument("--skip-last-images", type=int, default=0)
    parser.add_argument("--image-context-chars", type=int, default=DEFAULT_IMAGE_CONTEXT_CHARS)
    parser.add_argument("--limit-pdf-images", type=int, default=None)
    parser.add_argument("--bbox-iou-threshold", type=float, default=0.30)

    # Dataset mode, kept for the synthetic/triplet corpus.
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split-path", type=Path, default=None)
    parser.add_argument("--split", default=None, help="Optional split name from a json map, e.g. test.")
    parser.add_argument(
        "--candidate-source",
        choices=("positives", "positives-and-negatives"),
        default="positives-and-negatives",
    )
    parser.add_argument(
        "--component-text-source",
        choices=("sidecar-or-candidate", "sidecar", "candidate", "empty"),
        default="sidecar-or-candidate",
    )

    # Encoder and scoring.
    parser.add_argument("--model-size", default="2B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-retrieval-prompt", action="store_true")
    parser.add_argument("--max-token-length", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--hit-k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--scenario-found-k", type=int, default=8)
    parser.add_argument("--success-k", type=int, default=None)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--limit-candidates", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # Output/cache.
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV_REPORT)
    parser.add_argument("--detail-top-k", type=int, default=5)
    parser.add_argument("--no-details", action="store_true")
    parser.add_argument("--skip-missing-images", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only parse inputs and print counts.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def cleanup_ocr_text(text: Any) -> str:
    if not text:
        return ""
    text = str(text)
    for ch in "{}|[]":
        text = text.replace(ch, "")
    return normalize_whitespace(text)


def normalize_ui_text(text: Any) -> str:
    text = cleanup_ocr_text(text).lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s\-]+", " ", text)
    return normalize_whitespace(text)


def is_match(expected: str, found: str) -> bool:
    expected_norm = normalize_ui_text(expected)
    found_norm = normalize_ui_text(found)

    if not expected_norm or not found_norm:
        return False
    if expected_norm == found_norm:
        return True
    if expected_norm in found_norm or found_norm in expected_norm:
        return True

    expected_words = set(expected_norm.split())
    found_words = set(found_norm.split())
    if not expected_words or not found_words:
        return False
    return len(expected_words & found_words) / max(1, len(expected_words)) >= 0.65


def precision_recall_f1(expected: list[str], found: list[str]) -> tuple[float, float, float]:
    if not found:
        precision = 0.0
    else:
        precision = sum(any(is_match(exp, item) for exp in expected) for item in found) / len(found)

    if not expected:
        recall = 0.0
    else:
        recall = sum(any(is_match(exp, item) for item in found) for exp in expected) / len(expected)

    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2.0 * precision * recall / (precision + recall)


def mrr_for_expected(expected: list[str], found: list[str]) -> float:
    for idx, item in enumerate(found, start=1):
        if any(is_match(exp, item) for exp in expected):
            return 1.0 / idx
    return 0.0


def hit_at_k_for_expected(expected: list[str], found: list[str], k: int) -> bool:
    top = found[:k]
    return any(any(is_match(exp, item) for item in top) for exp in expected)


def clip_text(text: str, max_chars: int) -> str:
    text = normalize_whitespace(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip()


def pdf_key(value: str | Path | None) -> str | None:
    if value in (None, ""):
        return None
    raw = str(value).replace("\\", "/")
    name = Path(raw).name
    if name.lower().endswith(".pdf"):
        name = Path(name).stem
    return re.sub(r"[^0-9a-zA-Z._-]+", "_", name).strip("._").lower() or None


def safe_stem(path: Path) -> str:
    stem = re.sub(r"[^0-9a-zA-Z._-]+", "_", path.stem).strip("._")
    return stem or "document"


def as_bbox(value: Any) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"Expected bbox with 4 values, got: {value!r}")
    return tuple(float(x) for x in value)


def bbox_key(bbox: Iterable[float], ndigits: int = 6) -> str:
    return ",".join(f"{float(x):.{ndigits}f}" for x in bbox)


def infer_pdf_id(image_path: str, explicit: Any = None) -> str | None:
    explicit_key = pdf_key(explicit)
    if explicit_key is not None:
        return explicit_key
    normalized = str(image_path).replace("\\", "/")
    match = re.search(r"(pdf[_-]?\d+)", normalized, flags=re.IGNORECASE)
    if match:
        return match.group(1).replace("-", "_").lower()
    stem = Path(normalized).stem
    if "_image_" in stem:
        return stem.split("_image_", 1)[0]
    return pdf_key(stem)


def make_candidate_id(image_path: str, bbox: Iterable[float], explicit: Any = None) -> str:
    if explicit not in (None, ""):
        return str(explicit)
    normalized = str(image_path).replace("\\", "/")
    return f"{normalized}#{bbox_key(bbox)}"


def normalize_bbox_for_image(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    if max(abs(v) for v in bbox) <= 1.0:
        return bbox
    width, height = image_size
    if width <= 0 or height <= 0:
        return bbox
    x1, y1, x2, y2 = bbox
    return (x1 / width, y1 / height, x2 / width, y2 / height)


def bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


# This block is adapted from ../fffaffafaf/rag_core.py. It intentionally keeps
# only PDF extraction/context logic and avoids Streamlit/PostgreSQL dependencies.
def text_blocks_for_page(page: Any) -> list[tuple[float, float, float, float, str]]:
    blocks = []
    for block in page.get_text("blocks"):
        if len(block) >= 7 and block[6] == 0:
            text = normalize_whitespace(block[4])
            if text:
                blocks.append((block[0], block[1], block[2], block[3], text))
    return blocks


def nearby_text_for_rect(
    text_blocks: list[tuple[float, float, float, float, str]],
    rect: Any,
    max_blocks: int = 6,
) -> str:
    if not text_blocks:
        return ""

    img_cx = (rect.x0 + rect.x1) / 2
    img_cy = (rect.y0 + rect.y1) / 2
    ranked = []
    for x0, y0, x1, y1, text in text_blocks:
        tb_cx = (x0 + x1) / 2
        tb_cy = (y0 + y1) / 2
        ranked.append((math.hypot(img_cx - tb_cx, img_cy - tb_cy), text))

    ranked.sort(key=lambda item: item[0])
    return " ".join(text for _, text in ranked[:max_blocks])


def image_context(page_num: int, page_text: str, nearby_text: str, max_chars: int) -> str:
    parts = [f"Page {page_num}."]
    if nearby_text:
        parts.append(f"Nearest text to image: {nearby_text}")
    if page_text:
        parts.append(f"Page context: {clip_text(page_text, max_chars)}")
    return normalize_whitespace(" ".join(parts))


def extract_pdf_images(
    pdf_path: Path,
    out_dir: Path,
    skip_first_images: int = 0,
    skip_last_images: int = 0,
    image_context_chars: int = DEFAULT_IMAGE_CONTEXT_CHARS,
) -> list[PdfImage]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for PDF mode. Install it in the project venv: "
            ".\\.venv\\Scripts\\python.exe -m pip install PyMuPDF"
        ) from exc

    img_dir = out_dir / "images"
    txt_dir = out_dir / "texts"
    img_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    source_name = pdf_path.name
    source_pdf_id = pdf_key(pdf_path.name) or safe_stem(pdf_path)
    images_info: list[dict[str, Any]] = []

    with fitz.open(pdf_path) as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_num = page_index + 1
            page_text = normalize_whitespace(page.get_text("text"))
            page_blocks = text_blocks_for_page(page)

            for img_info in page.get_images(full=True):
                xref = img_info[0]
                rects = page.get_image_rects(xref)
                if not rects:
                    continue

                rect = rects[0]
                base_image = doc.extract_image(xref)
                nearby = nearby_text_for_rect(page_blocks, rect)
                context_text = image_context(page_num, page_text, nearby, image_context_chars)
                images_info.append(
                    {
                        "page_num": page_num,
                        "page_index": page_index,
                        "xref": xref,
                        "ext": base_image["ext"],
                        "bytes": base_image["image"],
                        "context_text": context_text,
                        "nearby_text": nearby,
                    }
                )

    total_images = len(images_info)
    start_idx = max(0, int(skip_first_images or 0))
    end_idx = total_images - max(0, int(skip_last_images or 0))
    filtered_images = images_info[start_idx:end_idx] if start_idx < end_idx else []

    pairs: list[PdfImage] = []
    for image_index, image_data in enumerate(filtered_images):
        img_filename = f"{safe_stem(pdf_path)}_image_{image_index}.{image_data['ext']}"
        txt_filename = f"{safe_stem(pdf_path)}_image_{image_index}.txt"
        img_path = img_dir / img_filename
        txt_path = txt_dir / txt_filename

        img_path.write_bytes(image_data["bytes"])
        txt_path.write_text(image_data["context_text"], encoding="utf-8")

        pairs.append(
            PdfImage(
                image_path=str(img_path),
                text_path=str(txt_path),
                context_text=image_data["context_text"],
                nearby_text=image_data["nearby_text"],
                pdf_id=source_pdf_id,
                source_name=source_name,
                page_num=int(image_data["page_num"]),
                page_index=int(image_data["page_index"]),
                image_index=image_index,
                xref=int(image_data["xref"]),
            )
        )

    return pairs


def extract_pdf_folder(args: argparse.Namespace) -> list[PdfImage]:
    pdf_dir = args.pdf_dir.resolve()
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No PDF files found in: {pdf_dir}")

    all_images: list[PdfImage] = []
    for pdf_path in tqdm(pdf_paths, desc="Extracting PDF images"):
        out_dir = args.pdf_extract_dir.resolve() / safe_stem(pdf_path)
        all_images.extend(
            extract_pdf_images(
                pdf_path=pdf_path,
                out_dir=out_dir,
                skip_first_images=args.skip_first_images,
                skip_last_images=args.skip_last_images,
                image_context_chars=args.image_context_chars,
            )
        )
        if args.limit_pdf_images is not None and len(all_images) >= args.limit_pdf_images:
            all_images = all_images[: args.limit_pdf_images]
            break

    return all_images


def load_user_scenario_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    spec = importlib.util.spec_from_file_location("external_user_scenario_eval", path)
    if spec is None or spec.loader is None:
        return []
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = getattr(module, "USER_TEST_CASES", None)
    if not cases:
        return []
    return [dict(item) for item in cases]


def load_pdf_queries(args: argparse.Namespace) -> list[QueryCase]:
    if args.queries.exists():
        raw = load_json(args.queries)
        raw_queries = raw.get("queries", raw.get("eval", [])) if isinstance(raw, dict) else raw
        source = str(args.queries)
    else:
        raw_queries = load_user_scenario_cases(args.scenario_file)
        source = str(args.scenario_file)

    raw_queries = apply_limits(list(raw_queries), args.limit_queries, args.shuffle, args.seed)

    queries: list[QueryCase] = []
    for idx, item in enumerate(raw_queries):
        query_text = str(item.get("query") or item.get("query_text") or item.get("text") or "").strip()
        if not query_text:
            continue
        relevant_ids = {str(value) for value in (item.get("relevant_ids") or [])}
        expected_pdf = item.get("expected_pdf") or item.get("pdf_id") or item.get("source_pdf")
        meta = dict(item)
        meta["query_source"] = source
        queries.append(
            QueryCase(
                id=str(item.get("id") or item.get("query_id") or f"q{idx}"),
                text=query_text,
                relevant_ids=relevant_ids,
                expected_pdf=pdf_key(expected_pdf),
                meta=meta,
            )
        )
    return queries


def candidate_to_dict(candidate: Candidate) -> dict[str, Any]:
    payload = asdict(candidate)
    payload["bbox"] = list(candidate.bbox)
    return payload


def candidate_from_dict(payload: dict[str, Any]) -> Candidate:
    return Candidate(
        id=str(payload["id"]),
        image_path=str(payload["image_path"]),
        bbox=as_bbox(payload["bbox"]),
        pdf_id=payload.get("pdf_id"),
        text=payload.get("text"),
        page_num=payload.get("page_num"),
        image_index=payload.get("image_index"),
        source_name=payload.get("source_name"),
        metadata=payload.get("metadata") or {},
    )


def maybe_filter_by_split(records: list[dict[str, Any]], split_path: Path | None, split: str | None) -> list[dict[str, Any]]:
    if not split_path or not split:
        return records
    split_map = load_json(split_path)

    def record_split(record: dict[str, Any]) -> str | None:
        raw = str(record.get("image_path", ""))
        variants = {raw, raw.replace("\\", "/"), Path(raw).name}
        for key in variants:
            if key in split_map:
                return split_map[key]
        return None

    return [record for record in records if record_split(record) == split]


def apply_limits(items: list[Any], limit: int | None, shuffle: bool, seed: int) -> list[Any]:
    if not shuffle and limit is None:
        return items
    result = list(items)
    if shuffle:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(result))
        result = [result[int(i)] for i in order]
    if limit is not None:
        result = result[:limit]
    return result


def add_candidate(
    candidates: dict[str, Candidate],
    image_path: str,
    bbox: tuple[float, float, float, float],
    pdf_id: str | None = None,
    text: str | None = None,
    explicit_id: Any = None,
    page_num: int | None = None,
    image_index: int | None = None,
    source_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Candidate:
    candidate_id = make_candidate_id(image_path, bbox, explicit=explicit_id)
    candidate = Candidate(
        id=candidate_id,
        image_path=str(image_path),
        bbox=bbox,
        pdf_id=infer_pdf_id(str(image_path), explicit=pdf_id),
        text=text,
        page_num=page_num,
        image_index=image_index,
        source_name=source_name,
        metadata=metadata or {},
    )
    candidates.setdefault(candidate.id, candidate)
    return candidates[candidate.id]


def parse_triplet_dataset(raw: list[dict[str, Any]], args: argparse.Namespace) -> EvalData:
    records = maybe_filter_by_split(raw, args.split_path, args.split)
    query_records = apply_limits(records, args.limit_queries, args.shuffle, args.seed)

    candidates: dict[str, Candidate] = {}
    queries: list[QueryCase] = []

    for item in records:
        image_path = str(item["image_path"])
        add_candidate(
            candidates,
            image_path=image_path,
            bbox=as_bbox(item["pos_bbox"]),
            pdf_id=item.get("pdf_id") or item.get("expected_pdf"),
        )
        if args.candidate_source == "positives-and-negatives" and item.get("neg_bbox") is not None:
            add_candidate(
                candidates,
                image_path=image_path,
                bbox=as_bbox(item["neg_bbox"]),
                pdf_id=item.get("pdf_id") or item.get("expected_pdf"),
            )

    for idx, item in enumerate(query_records):
        image_path = str(item["image_path"])
        query_text = str(item.get("query") or item.get("text") or "").strip()
        if not query_text:
            continue

        pos_id = make_candidate_id(image_path, as_bbox(item["pos_bbox"]))
        pos = candidates[pos_id]
        queries.append(
            QueryCase(
                id=str(item.get("id") or item.get("query_id") or f"q{idx}"),
                text=query_text,
                relevant_ids={pos.id},
                expected_pdf=pos.pdf_id,
                meta={"image_path": image_path, "pos_bbox": list(pos.bbox)},
            )
        )

    candidate_list = list(candidates.values())
    candidate_list = apply_limits(candidate_list, args.limit_candidates, False, args.seed)
    if args.limit_candidates is not None:
        kept = {candidate.id for candidate in candidate_list}
        for query in queries:
            query.relevant_ids.intersection_update(kept)
        queries = [query for query in queries if query.relevant_ids]
    return EvalData(queries=queries, candidates=candidate_list)


def candidate_from_obj(obj: Any, candidates: dict[str, Candidate]) -> Candidate:
    if isinstance(obj, str):
        candidate = candidates.get(obj)
        if candidate is None:
            raise ValueError(f"Relevant candidate id is not present in candidates: {obj}")
        return candidate
    image_path = str(obj.get("image_path") or obj.get("path") or "")
    bbox = as_bbox(obj.get("bbox") or obj.get("pos_bbox"))
    return add_candidate(
        candidates,
        image_path=image_path,
        bbox=bbox,
        pdf_id=obj.get("pdf_id") or obj.get("expected_pdf"),
        text=obj.get("text") or obj.get("context"),
        explicit_id=obj.get("id") or obj.get("candidate_id") or obj.get("component_id"),
        page_num=obj.get("page_num") or obj.get("expected_page"),
        image_index=obj.get("image_index"),
        source_name=obj.get("source_name"),
    )


def parse_explicit_dataset(raw: dict[str, Any] | list[dict[str, Any]], args: argparse.Namespace) -> EvalData:
    if isinstance(raw, dict):
        raw_candidates = raw.get("candidates", [])
        raw_queries = raw.get("queries", raw.get("eval", []))
    else:
        raw_candidates = []
        raw_queries = raw

    candidates: dict[str, Candidate] = {}
    for obj in raw_candidates:
        candidate_from_obj(obj, candidates)

    raw_queries = apply_limits(list(raw_queries), args.limit_queries, args.shuffle, args.seed)
    queries: list[QueryCase] = []

    for idx, item in enumerate(raw_queries):
        query_text = str(item.get("query") or item.get("query_text") or item.get("text") or "").strip()
        if not query_text:
            continue

        relevant_ids: set[str] = set()
        for key in ("relevant", "expected", "positive", "positives"):
            value = item.get(key)
            if value is None:
                continue
            values = value if isinstance(value, list) else [value]
            for obj in values:
                relevant_ids.add(candidate_from_obj(obj, candidates).id)

        for rel_id in item.get("relevant_ids", []) or []:
            relevant_ids.add(str(rel_id))

        if item.get("image_path") and item.get("bbox"):
            relevant_ids.add(candidate_from_obj(item, candidates).id)

        for key in ("candidates", "negatives", "distractors"):
            for obj in item.get(key, []) or []:
                candidate_from_obj(obj, candidates)

        if not relevant_ids:
            raise ValueError(f"Query {idx} has no relevant candidates: {item}")

        expected_pdf = item.get("expected_pdf") or item.get("pdf_id") or item.get("source_pdf")
        queries.append(
            QueryCase(
                id=str(item.get("id") or item.get("query_id") or f"q{idx}"),
                text=query_text,
                relevant_ids={str(x) for x in relevant_ids},
                expected_pdf=infer_pdf_id("", explicit=expected_pdf),
                meta={k: v for k, v in item.items() if k not in {"query", "query_text", "text"}},
            )
        )

    candidate_list = list(candidates.values())
    candidate_list = apply_limits(candidate_list, args.limit_candidates, False, args.seed)
    if args.limit_candidates is not None:
        kept = {candidate.id for candidate in candidate_list}
        for query in queries:
            query.relevant_ids.intersection_update(kept)
        queries = [query for query in queries if query.relevant_ids]
    return EvalData(queries=queries, candidates=candidate_list)


def load_dataset_eval_data(args: argparse.Namespace) -> EvalData:
    raw = load_json(args.dataset)
    if isinstance(raw, list) and raw and {"image_path", "pos_bbox"}.issubset(raw[0].keys()):
        return parse_triplet_dataset(raw, args)
    return parse_explicit_dataset(raw, args)


def resolve_image_path(image_path: str, dataset_path: Path, data_root: Path | None) -> Path:
    raw = Path(image_path)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        roots = [p for p in (data_root, dataset_path.parent, PROJECT_ROOT, Path.cwd()) if p is not None]
        for root in roots:
            candidates.append(root / raw)
            if "dataset_images" in raw.parts:
                replaced = Path(*("data" if part == "dataset_images" else part for part in raw.parts))
                candidates.append(root / replaced)
            candidates.append(root / "data" / raw.name)
            candidates.append(root / "dataset_images" / raw.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else raw.resolve()


def filter_missing_candidates(data: EvalData, args: argparse.Namespace) -> EvalData:
    if not args.skip_missing_images:
        return data

    kept: list[Candidate] = []
    missing_ids: set[str] = set()
    for candidate in data.candidates:
        image_path = resolve_image_path(candidate.image_path, args.dataset, args.data_root)
        if image_path.exists():
            kept.append(candidate)
        else:
            missing_ids.add(candidate.id)

    if not missing_ids:
        return data

    kept_ids = {candidate.id for candidate in kept}
    filtered_queries: list[QueryCase] = []
    for query in data.queries:
        query.relevant_ids.intersection_update(kept_ids)
        if query.relevant_ids:
            filtered_queries.append(query)

    print(
        f"[WARN] Skipped {len(missing_ids)} candidates with missing images; "
        f"{len(data.queries) - len(filtered_queries)} queries lost all relevant candidates."
    )
    return EvalData(queries=filtered_queries, candidates=kept)


def resolve_sidecar_text(image_path: Path, original_image_path: str, dataset_path: Path, data_root: Path | None) -> Path | None:
    raw = Path(original_image_path)
    candidates = [
        image_path.with_suffix(".txt"),
        dataset_path.parent / "data" / f"{image_path.stem}.txt",
        dataset_path.parent / "dataset_images" / f"{image_path.stem}.txt",
        PROJECT_ROOT / "training" / "synthetic_dataset" / "data" / f"{image_path.stem}.txt",
    ]
    if data_root is not None:
        candidates.extend(
            [
                data_root / raw.with_suffix(".txt"),
                data_root / "data" / f"{image_path.stem}.txt",
                data_root / "dataset_images" / f"{image_path.stem}.txt",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def read_component_text(
    candidate: Candidate,
    image_path: Path,
    dataset_path: Path,
    data_root: Path | None,
    source: str,
) -> str:
    if source == "empty":
        return ""
    if source == "candidate":
        return candidate.text or ""

    sidecar_path = resolve_sidecar_text(image_path, candidate.image_path, dataset_path, data_root)
    sidecar_text = sidecar_path.read_text(encoding="utf-8", errors="replace").strip() if sidecar_path else ""
    if source == "sidecar":
        return sidecar_text
    return sidecar_text or (candidate.text or "")


def choose_component_text(candidate: Candidate, sidecar_text: str, source: str) -> str:
    if source == "empty":
        return ""
    if source == "candidate":
        return candidate.text or ""
    if source == "sidecar":
        return sidecar_text
    return sidecar_text or (candidate.text or "")


@contextlib.contextmanager
def maybe_suppress_stdout(enabled: bool = True):
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def common_fingerprint_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "mode": args.mode,
        "model_size": args.model_size,
        "device": args.device,
        "use_retrieval_prompt": args.use_retrieval_prompt,
        "max_token_length": args.max_token_length,
        "component_text_source": args.component_text_source,
        "top_k": args.top_k,
    }


def dataset_fingerprint(args: argparse.Namespace, data: EvalData) -> str:
    payload = common_fingerprint_payload(args)
    payload.update(
        {
            "dataset": str(args.dataset.resolve()),
            "query_ids": [q.id for q in data.queries],
            "candidate_ids": [c.id for c in data.candidates],
        }
    )
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def pdf_fingerprint(args: argparse.Namespace, queries: list[QueryCase], pdf_images: list[PdfImage]) -> str:
    pdf_stats = []
    for pdf_path in sorted(args.pdf_dir.resolve().glob("*.pdf")):
        stat = pdf_path.stat()
        pdf_stats.append([pdf_path.name, stat.st_size, stat.st_mtime_ns])
    payload = common_fingerprint_payload(args)
    payload.update(
        {
            "pdf_dir": str(args.pdf_dir.resolve()),
            "pdf_stats": pdf_stats,
            "queries_path": str(args.queries.resolve()) if args.queries.exists() else None,
            "queries": [
                {
                    "id": q.id,
                    "text": q.text,
                    "expected_pdf": q.expected_pdf,
                    "relevant_ids": sorted(q.relevant_ids),
                    "meta": q.meta,
                }
                for q in queries
            ],
            "pdf_images": [asdict(item) for item in pdf_images],
            "bbox_iou_threshold": args.bbox_iou_threshold,
        }
    )
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def load_cache(
    cache_path: Path,
    fingerprint: str,
    with_candidates: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[Candidate] | None] | None:
    if not cache_path.exists():
        return None
    try:
        cached = np.load(cache_path, allow_pickle=False)
        meta = json.loads(str(cached["meta"].item()))
        if meta.get("fingerprint") != fingerprint:
            return None
        candidates = None
        if with_candidates:
            candidates_payload = json.loads(str(cached["candidates_json"].item()))
            candidates = [candidate_from_dict(item) for item in candidates_payload]
        return (
            np.asarray(cached["query_embeddings"], dtype=np.float32),
            np.asarray(cached["candidate_embeddings"], dtype=np.float32),
            candidates,
        )
    except Exception:
        return None


def save_cache(
    cache_path: Path,
    fingerprint: str,
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidates: list[Candidate] | None = None,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta = json.dumps({"fingerprint": fingerprint}, ensure_ascii=False)
    payload: dict[str, Any] = {
        "meta": np.array(meta),
        "query_embeddings": query_embeddings.astype(np.float32),
        "candidate_embeddings": candidate_embeddings.astype(np.float32),
    }
    if candidates is not None:
        payload["candidates_json"] = np.array(
            json.dumps([candidate_to_dict(candidate) for candidate in candidates], ensure_ascii=False)
        )
    np.savez_compressed(cache_path, **payload)


def build_pipeline(args: argparse.Namespace) -> Any:
    from config import UIEmbedderConfig
    from main import UIEmbedderPipeline

    config = UIEmbedderConfig.from_model_name(args.model_size, device=args.device)
    config.debug_decode_embeddings = False
    config.use_retrieval_prompt = bool(args.use_retrieval_prompt)
    if args.max_token_length is not None:
        config.max_token_length = args.max_token_length
    return UIEmbedderPipeline(config)


def encode_queries(pipeline: Any, queries: list[QueryCase]) -> np.ndarray:
    texts = [query.text for query in queries]
    with maybe_suppress_stdout(True):
        embeddings = pipeline.process(image=None, text_content=texts)
    return l2_normalize(embeddings)


def encode_dataset_candidates(
    pipeline: Any,
    candidates: list[Candidate],
    args: argparse.Namespace,
) -> np.ndarray:
    id_to_index = {candidate.id: idx for idx, candidate in enumerate(candidates)}
    embeddings: list[np.ndarray | None] = [None] * len(candidates)

    groups: dict[tuple[str, str], list[Candidate]] = {}
    image_cache: dict[str, tuple[Path, tuple[int, int], str]] = {}

    for candidate in candidates:
        image_path = resolve_image_path(candidate.image_path, args.dataset, args.data_root)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for candidate {candidate.id}: {image_path}")

        cache_key = str(image_path)
        if cache_key not in image_cache:
            with Image.open(image_path) as image:
                size = image.size
            sidecar_text = read_component_text(
                candidate=candidate,
                image_path=image_path,
                dataset_path=args.dataset,
                data_root=args.data_root,
                source="sidecar",
            )
            image_cache[cache_key] = (image_path, size, sidecar_text)
        else:
            _, size, sidecar_text = image_cache[cache_key]

        text = choose_component_text(candidate, sidecar_text, args.component_text_source)
        groups.setdefault((str(image_path), text), []).append(candidate)

    for (image_path_str, text_context), group in tqdm(groups.items(), desc="Encoding GUI candidates"):
        image_path = Path(image_path_str)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            size = image.size
            normalized_bboxes = [normalize_bbox_for_image(candidate.bbox, size) for candidate in group]
            unique_bboxes = list(dict.fromkeys(normalized_bboxes))
            with maybe_suppress_stdout(True):
                encoded = pipeline.process(image=image, text_content=text_context, bboxes=[list(b) for b in unique_bboxes])

        by_bbox = {bbox_key(key): np.asarray(value, dtype=np.float32) for key, value in encoded.items()}
        for candidate, normalized_bbox in zip(group, normalized_bboxes):
            vector = by_bbox.get(bbox_key(normalized_bbox))
            if vector is None:
                raise RuntimeError(f"Encoder did not return bbox {normalized_bbox} for {candidate.id}")
            embeddings[id_to_index[candidate.id]] = vector

    return l2_normalize(np.vstack([value for value in embeddings if value is not None]))


def encode_pdf_candidates(
    pipeline: Any,
    pdf_images: list[PdfImage],
    args: argparse.Namespace,
) -> tuple[list[Candidate], np.ndarray]:
    candidates: list[Candidate] = []
    embeddings: list[np.ndarray] = []

    for item in tqdm(pdf_images, desc="Encoding PDF GUI candidates"):
        image_path = Path(item.image_path)
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                width, height = image.size
                with maybe_suppress_stdout(True):
                    encoded = pipeline.process(image=image, text_content=item.context_text, bboxes=None)
        except Exception as exc:
            print(f"[WARN] Skipping PDF image {image_path}: {exc}")
            continue

        for bbox, emb in encoded.items():
            bbox_tuple = as_bbox(list(bbox))
            candidate_id = (
                f"{item.pdf_id}::page_{item.page_num}::image_{item.image_index}"
                f"::bbox_{bbox_key(bbox_tuple)}"
            )
            candidates.append(
                Candidate(
                    id=candidate_id,
                    image_path=str(image_path),
                    bbox=bbox_tuple,
                    pdf_id=item.pdf_id,
                    text=item.context_text,
                    page_num=item.page_num,
                    image_index=item.image_index,
                    source_name=item.source_name,
                    metadata={
                        "page_index": item.page_index,
                        "text_path": item.text_path,
                        "nearby_text": item.nearby_text,
                        "xref": item.xref,
                        "width": width,
                        "height": height,
                    },
                )
            )
            embeddings.append(np.asarray(emb, dtype=np.float32))
            if args.limit_candidates is not None and len(candidates) >= args.limit_candidates:
                return candidates, l2_normalize(np.vstack(embeddings))

    if not candidates:
        raise RuntimeError("No GUI candidates were produced from PDF images.")
    return candidates, l2_normalize(np.vstack(embeddings))


def expected_int(meta: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if meta.get(key) not in (None, ""):
            return int(meta[key])
    return None


def expected_bboxes(meta: dict[str, Any]) -> list[tuple[float, float, float, float]]:
    values = []
    for key in ("expected_bbox", "bbox", "relevant_bbox"):
        if meta.get(key) is not None:
            values.append(meta[key])
    for key in ("expected_bboxes", "bboxes", "relevant_bboxes"):
        if meta.get(key):
            values.extend(meta[key])
    return [as_bbox(value) for value in values]


def expected_terms(query: QueryCase) -> list[str]:
    values = query.meta.get("expected") or query.meta.get("targets") or query.meta.get("expected_terms") or []
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values if str(value).strip()]


def candidate_display_text(candidate: Candidate) -> str:
    text = candidate_metric_text(candidate)
    if len(text) > 240:
        return text[:240].rsplit(" ", 1)[0].strip()
    return text


def candidate_metric_text(candidate: Candidate) -> str:
    for key in ("ui_text", "label", "title", "ocr_text"):
        value = candidate.metadata.get(key)
        if value:
            return cleanup_ocr_text(value)
    return cleanup_ocr_text(candidate.text)


def query_matches_candidate(
    query: QueryCase,
    candidate: Candidate,
    iou_threshold: float,
) -> bool:
    meta = query.meta
    expected_pdf = query.expected_pdf or pdf_key(meta.get("expected_pdf") or meta.get("pdf_id") or meta.get("source_pdf"))
    if expected_pdf and candidate.pdf_id != expected_pdf:
        return False

    expected_page = expected_int(meta, "expected_page", "page_num", "page")
    if expected_page is not None and candidate.page_num != expected_page:
        return False

    expected_image_index = expected_int(meta, "expected_image_index", "image_index")
    if expected_image_index is not None and candidate.image_index != expected_image_index:
        return False

    expected_image_path = meta.get("expected_image_path") or meta.get("image_path")
    if expected_image_path:
        if Path(str(expected_image_path)).name != Path(candidate.image_path).name:
            return False

    bboxes = expected_bboxes(meta)
    if not bboxes:
        return any([expected_pdf, expected_page is not None, expected_image_index is not None, expected_image_path])

    image_size = (
        int(candidate.metadata.get("width") or 0),
        int(candidate.metadata.get("height") or 0),
    )
    normalized_expected = [normalize_bbox_for_image(bbox, image_size) for bbox in bboxes]
    return any(bbox_iou(candidate.bbox, expected) >= iou_threshold for expected in normalized_expected)


def resolve_pdf_relevance(queries: list[QueryCase], candidates: list[Candidate], iou_threshold: float) -> list[QueryCase]:
    candidate_ids = {candidate.id for candidate in candidates}
    resolved: list[QueryCase] = []
    dropped = 0
    for query in queries:
        relevant = {candidate_id for candidate_id in query.relevant_ids if candidate_id in candidate_ids}
        if not relevant:
            relevant = {
                candidate.id
                for candidate in candidates
                if query_matches_candidate(query, candidate, iou_threshold)
            }
        query.relevant_ids = relevant
        if relevant:
            resolved.append(query)
        else:
            dropped += 1
    if dropped:
        print(f"[WARN] Dropped {dropped} queries without matched relevant PDF candidates.")
    return resolved


def top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[1])
    if k <= 0:
        return np.empty((scores.shape[0], 0), dtype=np.int64)
    unsorted = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    row = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[row, unsorted], axis=1)
    return unsorted[row, order]


def compute_metrics(
    queries: list[QueryCase],
    candidates: list[Candidate],
    query_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    top_k: int,
    hit_ks: list[int],
    scenario_found_k: int,
    success_k: int,
    query_batch_size: int,
    detail_top_k: int,
    save_details: bool,
) -> dict[str, Any]:
    candidate_index = {candidate.id: idx for idx, candidate in enumerate(candidates)}
    candidate_by_index = {idx: candidate for idx, candidate in enumerate(candidates)}
    max_rank_k = max([top_k, success_k, scenario_found_k, detail_top_k] + hit_ks)

    per_query: list[dict[str, Any]] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    reciprocal_ranks: list[float] = []
    hit_counts = {k: 0 for k in hit_ks}
    success_count = 0
    pdf_correct = 0
    pdf_total = 0

    for start in tqdm(range(0, len(queries), query_batch_size), desc="Scoring queries"):
        end = min(start + query_batch_size, len(queries))
        scores = query_embeddings[start:end] @ candidate_embeddings.T
        top = top_indices(scores, max_rank_k)

        for row_idx, query_idx in enumerate(range(start, end)):
            query = queries[query_idx]
            row_scores = scores[row_idx]
            top_for_query = [int(i) for i in top[row_idx]]
            scenario_expected = expected_terms(query)

            if scenario_expected:
                found_indices = top_for_query[:scenario_found_k]
                found_metric = [candidate_metric_text(candidate_by_index[cidx]) for cidx in found_indices]
                found = [candidate_display_text(candidate_by_index[cidx]) for cidx in found_indices]
                precision, recall, f1 = precision_recall_f1(scenario_expected, found_metric)
                rr = mrr_for_expected(scenario_expected, found_metric)

                precision_values.append(precision)
                recall_values.append(recall)
                f1_values.append(f1)
                reciprocal_ranks.append(rr)

                success = recall == 1.0
                if success:
                    success_count += 1
                for k in hit_ks:
                    if hit_at_k_for_expected(scenario_expected, found_metric, k):
                        hit_counts[k] += 1

                top1 = candidate_by_index[top_for_query[0]] if top_for_query else None
                pdf_match = None
                expected_pdf = query.expected_pdf
                if expected_pdf is not None and top1 is not None:
                    pdf_total += 1
                    pdf_match = top1.pdf_id == expected_pdf
                    if pdf_match:
                        pdf_correct += 1

                item = {
                    "query_id": query.id,
                    "query": query.text,
                    "rank": (int(round(1.0 / rr)) if rr > 0 else None),
                    "reciprocal_rank": rr,
                    "precision_at_k": precision,
                    "recall_at_k": recall,
                    "f1_at_k": f1,
                    "expected_pdf": expected_pdf,
                    "top1_pdf": top1.pdf_id if top1 else None,
                    "pdf_match": pdf_match,
                    "best_relevant_score": None,
                    "relevant_count": len(scenario_expected),
                    "expected": scenario_expected,
                    "found": found,
                    "success": success,
                }
                if save_details:
                    item["top_results"] = [
                        {
                            "rank": pos + 1,
                            "candidate_id": candidate_by_index[cidx].id,
                            "score": float(row_scores[cidx]),
                            "pdf_id": candidate_by_index[cidx].pdf_id,
                            "source_name": candidate_by_index[cidx].source_name,
                            "page_num": candidate_by_index[cidx].page_num,
                            "image_index": candidate_by_index[cidx].image_index,
                            "image_path": candidate_by_index[cidx].image_path,
                            "bbox": list(candidate_by_index[cidx].bbox),
                            "text": candidate_display_text(candidate_by_index[cidx]),
                            "is_relevant": any(is_match(exp, candidate_metric_text(candidate_by_index[cidx])) for exp in scenario_expected),
                        }
                        for pos, cidx in enumerate(top_for_query[:detail_top_k])
                    ]
                per_query.append(item)
                continue

            relevant_indices = {candidate_index[cid] for cid in query.relevant_ids if cid in candidate_index}
            if not relevant_indices:
                continue

            relevant_score_values = row_scores[list(relevant_indices)]
            best_relevant_score = float(np.max(relevant_score_values))
            rank = int(np.count_nonzero(row_scores > best_relevant_score) + 1)

            top_k_indices = top_for_query[:top_k]
            found_at_top_k = len(set(top_k_indices) & relevant_indices)

            precision = found_at_top_k / max(1, min(top_k, len(candidates)))
            recall = found_at_top_k / len(relevant_indices)
            f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)
            reciprocal_ranks.append(1.0 / rank)

            if rank <= success_k:
                success_count += 1
            for k in hit_ks:
                if rank <= k:
                    hit_counts[k] += 1

            top1 = candidate_by_index[top_for_query[0]] if top_for_query else None
            expected_pdf = query.expected_pdf
            if expected_pdf is None and relevant_indices:
                expected_pdf = candidate_by_index[next(iter(relevant_indices))].pdf_id
            pdf_match = None
            if expected_pdf is not None and top1 is not None:
                pdf_total += 1
                pdf_match = top1.pdf_id == expected_pdf
                if pdf_match:
                    pdf_correct += 1

            item = {
                "query_id": query.id,
                "rank": rank,
                "reciprocal_rank": 1.0 / rank,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "f1_at_k": f1,
                "expected_pdf": expected_pdf,
                "top1_pdf": top1.pdf_id if top1 else None,
                "pdf_match": pdf_match,
                "best_relevant_score": best_relevant_score,
                "relevant_count": len(relevant_indices),
            }
            if save_details:
                item["top_results"] = [
                    {
                        "rank": pos + 1,
                        "candidate_id": candidate_by_index[cidx].id,
                        "score": float(row_scores[cidx]),
                        "pdf_id": candidate_by_index[cidx].pdf_id,
                        "source_name": candidate_by_index[cidx].source_name,
                        "page_num": candidate_by_index[cidx].page_num,
                        "image_index": candidate_by_index[cidx].image_index,
                        "image_path": candidate_by_index[cidx].image_path,
                        "bbox": list(candidate_by_index[cidx].bbox),
                        "is_relevant": cidx in relevant_indices,
                    }
                    for pos, cidx in enumerate(top_for_query[:detail_top_k])
                ]
            per_query.append(item)

    n = max(1, len(per_query))
    metrics = {
        "Success Rate": success_count / n,
        "Precision": float(np.mean(precision_values)) if precision_values else 0.0,
        "Recall": float(np.mean(recall_values)) if recall_values else 0.0,
        "F1-score": float(np.mean(f1_values)) if f1_values else 0.0,
        "MRR": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "PDF Accuracy": (pdf_correct / pdf_total) if pdf_total else None,
    }
    for k in hit_ks:
        metrics[f"Hit@{k}"] = hit_counts[k] / n

    return {
        "metrics": metrics,
        "per_query": per_query,
        "settings": {
            "top_k": top_k,
            "scenario_found_k": scenario_found_k,
            "success_k": success_k,
            "hit_k": hit_ks,
            "pdf_accuracy_queries": pdf_total,
        },
    }


def print_dataset_summary(data: EvalData, pdf_images: list[PdfImage] | None = None) -> None:
    pdfs = {candidate.pdf_id for candidate in data.candidates if candidate.pdf_id is not None}
    images = {candidate.image_path for candidate in data.candidates}
    print("\nDataset summary")
    print(f"  Queries:        {len(data.queries)}")
    print(f"  GUI candidates: {len(data.candidates)}")
    print(f"  PDFs:           {len(pdfs)}")
    print(f"  Images:         {len(images)}")
    if pdf_images is not None:
        extracted_pdfs = {item.pdf_id for item in pdf_images}
        print(f"  PDF images:     {len(pdf_images)}")
        print(f"  PDF files:      {len(extracted_pdfs)}")


def print_pdf_input_summary(pdf_dir: Path, queries: list[QueryCase], pdf_images: list[PdfImage] | None = None) -> None:
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    print("\nPDF input summary")
    print(f"  PDF folder: {pdf_dir}")
    print(f"  PDF files:  {len(pdf_paths)}")
    print(f"  Queries:    {len(queries)}")
    if pdf_images is not None:
        print(f"  Images:     {len(pdf_images)}")


def print_metrics_table(metrics: dict[str, Any]) -> None:
    order = ["Success Rate", "Precision", "Recall", "F1-score", "MRR", "Hit@1", "Hit@3", "Hit@5", "PDF Accuracy"]
    print("\nRetrieval metrics")
    print("| Metric | Value |")
    print("|---|---:|")
    for key in order:
        if key not in metrics:
            continue
        value = metrics[key]
        rendered = "n/a" if value is None else f"{float(value):.4f}"
        print(f"| {key} | {rendered} |")


def write_reports(
    args: argparse.Namespace,
    data: EvalData,
    result: dict[str, Any],
    pdf_images: list[PdfImage] | None = None,
) -> None:
    payload = {
        "dataset": {
            "mode": args.mode,
            "path": str(args.pdf_dir if args.mode == "pdf" else args.dataset),
            "num_queries": len(data.queries),
            "num_candidates": len(data.candidates),
            "num_pdfs": len({c.pdf_id for c in data.candidates if c.pdf_id is not None}),
            "num_images": len({c.image_path for c in data.candidates}),
            "num_pdf_images": len(pdf_images or []),
        },
        "settings": result["settings"],
        "metrics": result["metrics"],
        "queries": result["per_query"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_id",
                "query",
                "rank",
                "reciprocal_rank",
                "precision_at_k",
                "recall_at_k",
                "f1_at_k",
                "expected_pdf",
                "top1_pdf",
                "pdf_match",
                "expected",
                "found",
                "success",
                "best_relevant_score",
                "relevant_count",
            ],
        )
        writer.writeheader()
        for item in result["per_query"]:
            row = {key: item.get(key) for key in writer.fieldnames}
            if isinstance(row.get("expected"), list):
                row["expected"] = " | ".join(row["expected"])
            if isinstance(row.get("found"), list):
                row["found"] = " | ".join(row["found"])
            writer.writerow(row)

    print(f"\nSaved JSON report: {args.output_json}")
    print(f"Saved CSV report:  {args.output_csv}")


def run_dataset_mode(args: argparse.Namespace) -> None:
    args.dataset = args.dataset.resolve()
    args.data_root = args.data_root.resolve() if args.data_root else args.dataset.parent
    success_k = args.success_k if args.success_k is not None else args.top_k

    data = filter_missing_candidates(load_dataset_eval_data(args), args)
    if not data.queries:
        raise SystemExit("No evaluation queries were loaded.")
    if not data.candidates:
        raise SystemExit("No GUI candidates were loaded.")
    print_dataset_summary(data)

    if args.dry_run:
        return

    fingerprint = dataset_fingerprint(args, data)
    cache = None if args.no_cache or args.refresh_cache else load_cache(args.cache_path, fingerprint)
    if cache is None:
        pipeline = build_pipeline(args)
        candidate_embeddings = encode_dataset_candidates(pipeline, data.candidates, args)
        query_embeddings = encode_queries(pipeline, data.queries)
        if not args.no_cache:
            save_cache(args.cache_path, fingerprint, query_embeddings, candidate_embeddings)
            print(f"Saved embedding cache: {args.cache_path}")
    else:
        query_embeddings, candidate_embeddings, _ = cache
        print(f"Loaded embedding cache: {args.cache_path}")

    result = compute_metrics(
        queries=data.queries,
        candidates=data.candidates,
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
        top_k=args.top_k,
        hit_ks=sorted(set(args.hit_k)),
        scenario_found_k=args.scenario_found_k,
        success_k=success_k,
        query_batch_size=args.query_batch_size,
        detail_top_k=args.detail_top_k,
        save_details=not args.no_details,
    )
    print_metrics_table(result["metrics"])
    write_reports(args, data, result)


def run_pdf_mode(args: argparse.Namespace) -> None:
    args.pdf_dir = args.pdf_dir.resolve()
    args.queries = args.queries.resolve()
    args.pdf_extract_dir = args.pdf_extract_dir.resolve()
    success_k = args.success_k if args.success_k is not None else args.top_k

    queries = load_pdf_queries(args)
    if args.dry_run:
        print_pdf_input_summary(args.pdf_dir, queries)
        if not args.queries.exists():
            print(f"[INFO] Query file not found, using scenarios from: {args.scenario_file}")
        return

    if not queries:
        raise SystemExit(
            f"No control queries loaded from {args.queries}. "
            "Create this JSON before running metrics."
        )

    pdf_images = extract_pdf_folder(args)
    if not pdf_images:
        raise SystemExit(f"No images were extracted from PDF folder: {args.pdf_dir}")

    print_pdf_input_summary(args.pdf_dir, queries, pdf_images)
    fingerprint = pdf_fingerprint(args, queries, pdf_images)
    cache = None if args.no_cache or args.refresh_cache else load_cache(
        args.cache_path,
        fingerprint,
        with_candidates=True,
    )

    if cache is None:
        pipeline = build_pipeline(args)
        candidates, candidate_embeddings = encode_pdf_candidates(pipeline, pdf_images, args)
        queries = resolve_pdf_relevance(queries, candidates, args.bbox_iou_threshold)
        data = EvalData(queries=queries, candidates=candidates)
        if not data.queries:
            raise SystemExit(
                "No queries could be matched to PDF candidates. "
                "Check expected_pdf/expected_page/expected_bbox in the query JSON."
            )
        query_embeddings = encode_queries(pipeline, data.queries)
        if not args.no_cache:
            save_cache(args.cache_path, fingerprint, query_embeddings, candidate_embeddings, data.candidates)
            print(f"Saved embedding cache: {args.cache_path}")
    else:
        query_embeddings, candidate_embeddings, cached_candidates = cache
        if cached_candidates is None:
            raise RuntimeError("PDF cache is missing candidate metadata. Re-run with --refresh-cache.")
        candidates = cached_candidates
        queries = resolve_pdf_relevance(queries, candidates, args.bbox_iou_threshold)
        data = EvalData(queries=queries, candidates=candidates)
        if len(data.queries) != query_embeddings.shape[0]:
            raise RuntimeError("PDF cache query count mismatch. Re-run with --refresh-cache.")
        print(f"Loaded embedding cache: {args.cache_path}")

    if not data.queries:
        raise SystemExit(
            "No queries could be matched to PDF candidates. "
            "Check expected_pdf/expected_page/expected_bbox in the query JSON."
        )
    print_dataset_summary(data, pdf_images=pdf_images)

    result = compute_metrics(
        queries=data.queries,
        candidates=data.candidates,
        query_embeddings=query_embeddings,
        candidate_embeddings=candidate_embeddings,
        top_k=args.top_k,
        hit_ks=sorted(set(args.hit_k)),
        scenario_found_k=args.scenario_found_k,
        success_k=success_k,
        query_batch_size=args.query_batch_size,
        detail_top_k=args.detail_top_k,
        save_details=not args.no_details,
    )
    print_metrics_table(result["metrics"])
    write_reports(args, data, result, pdf_images=pdf_images)


def main() -> None:
    args = parse_args()
    if args.mode == "pdf":
        run_pdf_mode(args)
    else:
        run_dataset_mode(args)


if __name__ == "__main__":
    main()
