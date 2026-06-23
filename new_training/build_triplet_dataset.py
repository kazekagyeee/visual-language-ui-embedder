from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
UI_ITEMS_PATH = ROOT_DIR / "generated" / "ui_index" / "ui_items.jsonl"
OUT_PATH = ROOT_DIR / "generated" / "triplets.jsonl"
SCENARIO_FILE = PROJECT_ROOT.parent / "visual-language-two-tower-kristina" / "evaluation" / "user_scenario_eval.py"


def normalize(text: str) -> str:
    text = str(text).lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_user_test_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"user_scenario_eval.py was not found: {path}")

    spec = importlib.util.spec_from_file_location("user_scenario_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cases = getattr(module, "USER_TEST_CASES", None)
    if not cases:
        raise RuntimeError(f"USER_TEST_CASES is empty or missing in {path}")

    return list(cases)


def text_match(pattern: str, text: str) -> bool:
    pattern_norm = normalize(pattern)
    text_norm = normalize(text)

    if not pattern_norm or not text_norm:
        return False
    if pattern_norm == text_norm:
        return True
    if pattern_norm in text_norm or text_norm in pattern_norm:
        return True

    pattern_words = set(pattern_norm.split())
    text_words = set(text_norm.split())
    if not pattern_words:
        return False

    overlap = len(pattern_words & text_words) / max(1, len(pattern_words))
    return overlap >= 0.65


def lexical_overlap(query: str, text: str) -> float:
    query_words = set(normalize(query).split())
    text_words = set(normalize(text).split())
    if not query_words or not text_words:
        return 0.0
    return len(query_words & text_words) / max(1, len(query_words))


def item_text(item: dict[str, Any]) -> str:
    return str(item.get("normalized_text") or normalize(item.get("text", "")))


def build_items_by_pdf(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item.get("pdf_name", ""))].append(item)
    return grouped


def matches_any(item: dict[str, Any], patterns: list[str]) -> bool:
    text = item_text(item)
    return any(text_match(pattern, text) for pattern in patterns)


def find_positives(
    items: list[dict[str, Any]],
    patterns: list[str],
    expected_pdf: str | None = None,
) -> list[dict[str, Any]]:
    positives: list[dict[str, Any]] = []
    expected_pdf_norm = Path(expected_pdf).name.lower() if expected_pdf else None

    for item in items:
        if expected_pdf_norm and str(item.get("pdf_name", "")).lower() != expected_pdf_norm:
            continue
        text = item_text(item)
        if any(text_match(pattern, text) for pattern in patterns):
            positives.append(item)

    return positives


def is_positive_for_case(item: dict[str, Any], patterns: list[str], expected_pdf: str | None) -> bool:
    expected_pdf_norm = Path(expected_pdf).name.lower() if expected_pdf else None
    if expected_pdf_norm and str(item.get("pdf_name", "")).lower() != expected_pdf_norm:
        return False
    return any(text_match(pattern, item_text(item)) for pattern in patterns)


def choose_negatives(
    *,
    rng: random.Random,
    items: list[dict[str, Any]],
    items_by_pdf: dict[str, list[dict[str, Any]]],
    positive: dict[str, Any],
    query: str,
    expected: list[str],
    negatives_per_positive: int,
) -> list[dict[str, Any]]:
    positive_id = positive.get("id")
    selected: list[dict[str, Any]] = []
    same_pdf_pool = items_by_pdf.get(str(positive.get("pdf_name", "")), [])
    same_pdf: list[dict[str, Any]] = []
    sample_attempts = max(200, negatives_per_positive * 100)

    for _ in range(sample_attempts):
        if not same_pdf_pool:
            break
        item = rng.choice(same_pdf_pool)
        if item.get("id") == positive_id:
            continue
        if matches_any(item, expected):
            continue
        same_pdf.append(item)
        if len(same_pdf) >= 200:
            break

    hard = sorted(
        same_pdf,
        key=lambda item: lexical_overlap(query, item.get("text", "")),
        reverse=True,
    )
    selected.extend(hard[: max(0, negatives_per_positive)])

    selected_ids = {item.get("id") for item in selected}
    max_attempts = max(200, negatives_per_positive * 100)
    attempts = 0

    while len(selected) < negatives_per_positive and attempts < max_attempts:
        attempts += 1
        item = rng.choice(items)
        item_id = item.get("id")
        if item_id == positive_id or item_id in selected_ids:
            continue
        if matches_any(item, expected):
            continue
        selected.append(item)
        selected_ids.add(item_id)

    return selected[:negatives_per_positive]


def choose_negative_for_item(
    *,
    rng: random.Random,
    items: list[dict[str, Any]],
    items_by_pdf: dict[str, list[dict[str, Any]]],
    positive: dict[str, Any],
) -> dict[str, Any] | None:
    positive_text = item_text(positive)
    positive_id = positive.get("id")

    pools = [
        items_by_pdf.get(str(positive.get("pdf_name", "")), []),
        items,
    ]
    for pool in pools:
        if not pool:
            continue
        for _ in range(100):
            item = rng.choice(pool)
            if item.get("id") == positive_id:
                continue
            if text_match(positive_text, item_text(item)):
                continue
            return item

    return None


def make_triplet(
    *,
    query: str,
    positive: dict[str, Any],
    negative: dict[str, Any],
    source: str,
    expected: list[str],
) -> dict[str, Any]:
    return {
        "query": query,
        "source": source,
        "expected": expected,
        "pos_id": positive.get("id"),
        "pos_image_path": positive.get("screenshot_image"),
        "pos_bbox": positive.get("bbox"),
        "pos_text": positive.get("text", ""),
        "pos_context": positive.get("context_text") or positive.get("text", ""),
        "pos_pdf": positive.get("pdf_name"),
        "pos_page": positive.get("page"),
        "pos_ui_type": positive.get("ui_type", ""),
        "neg_id": negative.get("id"),
        "neg_image_path": negative.get("screenshot_image"),
        "neg_bbox": negative.get("bbox"),
        "neg_text": negative.get("text", ""),
        "neg_context": negative.get("context_text") or negative.get("text", ""),
        "neg_pdf": negative.get("pdf_name"),
        "neg_page": negative.get("page"),
        "neg_ui_type": negative.get("ui_type", ""),
    }


def build_scenario_triplets(
    *,
    rng: random.Random,
    items: list[dict[str, Any]],
    items_by_pdf: dict[str, list[dict[str, Any]]],
    cases: list[dict[str, Any]],
    max_positives_per_query: int,
    negatives_per_positive: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    triplets: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for case in tqdm(cases, desc="Scenario triplets"):
        query = str(case.get("query", "")).strip()
        expected = [str(value) for value in case.get("expected", []) if str(value).strip()]
        expected_pdf = case.get("expected_pdf")

        if not query or not expected:
            continue

        expected_pdf_norm = Path(expected_pdf).name if expected_pdf else None
        search_items = items_by_pdf.get(expected_pdf_norm, items) if expected_pdf_norm else items
        positives = find_positives(search_items, expected, expected_pdf=None)
        if not positives:
            missing.append(case)
            continue

        positives = sorted(
            positives,
            key=lambda item: (
                -max(lexical_overlap(pattern, item.get("text", "")) for pattern in expected),
                item.get("id", ""),
            ),
        )[:max_positives_per_query]

        for positive in positives:
            negatives = choose_negatives(
                rng=rng,
                items=items,
                items_by_pdf=items_by_pdf,
                positive=positive,
                query=query,
                expected=expected,
                negatives_per_positive=negatives_per_positive,
            )
            for negative in negatives:
                triplets.append(
                    make_triplet(
                        query=query,
                        positive=positive,
                        negative=negative,
                        source="user_scenario",
                        expected=expected,
                    )
                )

    return triplets, missing


def query_templates(item: dict[str, Any]) -> list[str]:
    text = str(item.get("text", "")).strip()
    ui_type = str(item.get("ui_type", ""))
    if not text:
        return []

    templates = [
        f"где найти {text}",
        f"как открыть {text}",
        f"покажи {text}",
    ]
    if ui_type == "button":
        templates.extend([f"как нажать {text}", f"где кнопка {text}"])
    if ui_type == "menu_item":
        templates.append(f"где находится раздел {text}")
    return templates


def build_synthetic_triplets(
    *,
    rng: random.Random,
    items: list[dict[str, Any]],
    items_by_pdf: dict[str, list[dict[str, Any]]],
    max_synthetic_items: int,
    synthetic_queries_per_item: int,
) -> list[dict[str, Any]]:
    usable_items = [
        item
        for item in items
        if len(item_text(item)) >= 4 and Path(str(item.get("screenshot_image", ""))).exists()
    ]
    rng.shuffle(usable_items)
    if max_synthetic_items > 0:
        usable_items = usable_items[:max_synthetic_items]

    triplets: list[dict[str, Any]] = []
    for positive in tqdm(usable_items, desc="Synthetic triplets"):
        negative = choose_negative_for_item(
            rng=rng,
            items=items,
            items_by_pdf=items_by_pdf,
            positive=positive,
        )
        if not negative:
            continue

        templates = query_templates(positive)
        rng.shuffle(templates)
        for query in templates[:synthetic_queries_per_item]:
            triplets.append(
                make_triplet(
                    query=query,
                    positive=positive,
                    negative=negative,
                    source="ocr_synthetic",
                    expected=[str(positive.get("text", ""))],
                )
            )

    return triplets


def dedupe_triplets(triplets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, Any]] = []

    for triplet in triplets:
        key = (
            normalize(triplet.get("query", "")),
            str(triplet.get("pos_id", "")),
            str(triplet.get("neg_id", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(triplet)

    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build triplet dataset for 3B projection adapter.")
    parser.add_argument("--ui-items", type=Path, default=UI_ITEMS_PATH)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--scenario-file", type=Path, default=SCENARIO_FILE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-positives-per-query", type=int, default=8)
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--max-synthetic-items", type=int, default=3000)
    parser.add_argument("--synthetic-queries-per-item", type=int, default=1)
    parser.add_argument("--no-synthetic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    items = load_jsonl(args.ui_items)
    if not items:
        raise RuntimeError(f"No UI items found: {args.ui_items}")
    items_by_pdf = build_items_by_pdf(items)

    cases = load_user_test_cases(args.scenario_file)
    scenario_triplets, missing = build_scenario_triplets(
        rng=rng,
        items=items,
        items_by_pdf=items_by_pdf,
        cases=cases,
        max_positives_per_query=args.max_positives_per_query,
        negatives_per_positive=args.negatives_per_positive,
    )

    synthetic_triplets: list[dict[str, Any]] = []
    if not args.no_synthetic:
        synthetic_triplets = build_synthetic_triplets(
            rng=rng,
            items=items,
            items_by_pdf=items_by_pdf,
            max_synthetic_items=args.max_synthetic_items,
            synthetic_queries_per_item=args.synthetic_queries_per_item,
        )

    triplets = dedupe_triplets(scenario_triplets + synthetic_triplets)
    write_jsonl(args.out, triplets)

    summary = {
        "ui_items": len(items),
        "scenario_cases": len(cases),
        "scenario_triplets": len(scenario_triplets),
        "synthetic_triplets": len(synthetic_triplets),
        "total_triplets": len(triplets),
        "missing_cases": len(missing),
        "missing_queries": [case.get("query") for case in missing],
    }
    with (args.out.parent / "triplets_meta.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[OK] triplets saved:", args.out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
