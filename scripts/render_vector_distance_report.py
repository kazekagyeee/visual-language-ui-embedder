import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_INPUT = "scripts/debug/db_vector_distance_report.json"


METRIC_DESCRIPTIONS = {
    "rows": "Всего строк в таблице.",
    "non_null_embeddings": "Сколько строк содержит непустой embedding.",
    "null_embeddings": "Сколько строк без embedding.",
    "min_dims": "Минимальная размерность embedding.",
    "max_dims": "Максимальная размерность embedding.",
    "min_norm": "Минимальная L2-норма вектора. Для нормализованных векторов должна быть около 1.",
    "avg_norm": "Средняя L2-норма вектора. Сильное отклонение от 1 мешает интерпретировать cosine-поиск.",
    "max_norm": "Максимальная L2-норма вектора.",
    "pairs": "Количество пар, по которым посчитаны расстояния.",
    "mode": "exact = перебраны все пары; sample = случайная выборка пар.",
    "requested_sample_pairs": "Сколько случайных пар запрашивали для sample-режима.",
    "min_distance": "Минимальное cosine distance. Чем ближе к 0, тем похожее пара.",
    "avg_distance": "Среднее cosine distance по парам.",
    "max_distance": "Максимальное cosine distance. Значения около 1 и выше означают слабую/отрицательную cosine similarity.",
    "min_similarity": "Минимальная cosine similarity. Ниже 0 означает противоположные направления.",
    "avg_similarity": "Средняя cosine similarity по парам. Чем выше, тем плотнее/похожее облако векторов.",
    "max_similarity": "Максимальная cosine similarity. Значения около 1 обычно означают дубли или почти одинаковые объекты.",
    "distinct_image_bbox": "Уникальные пары image_path + bbox.",
    "distinct_content": "Уникальные текстовые content.",
    "distinct_embeddings": "Уникальные embedding по точному текстовому представлению вектора.",
    "duplicate_groups": "Сколько групп ключей встречаются больше одного раза.",
    "rows_in_duplicate_groups": "Сколько строк лежит внутри групп дублей.",
    "extra_duplicate_rows": "Сколько строк является лишними повторами: sum(group_size - 1).",
    "max_group_size": "Размер самой большой группы дублей.",
    "same_image": "true = обе строки из одного image_path; false = из разных изображений.",
    "cosine_distance": "Cosine distance для конкретной пары: 1 - cosine similarity.",
    "cosine_similarity": "Cosine similarity для конкретной пары: ближе к 1 = более похожи.",
    "id_a": "ID первой строки в паре.",
    "id_b": "ID второй строки в паре.",
    "image_path_a": "Путь изображения первой строки.",
    "image_path_b": "Путь изображения второй строки.",
    "bbox_a": "BBox первой строки.",
    "bbox_b": "BBox второй строки.",
    "text_chunk_id_a": "Связанный text_chunk_id первой image-строки.",
    "text_chunk_id_b": "Связанный text_chunk_id второй image-строки.",
    "content_a": "Текст первой строки.",
    "content_b": "Текст второй строки.",
    "content_prefix": "Начало текстового content для группы дублей.",
    "ids": "ID строк в группе.",
    "text_chunk_ids": "Связанные text_chunk_id в группе image-дублей.",
}


SECTION_DESCRIPTIONS = {
    "overview": "Базовая проверка таблицы: количество строк, размерность и нормы векторов.",
    "pairwise": "Распределение cosine distance/similarity между векторами внутри одной таблицы.",
    "duplicates": "Дубли по бизнес-ключу: для image это image_path+bbox, для text это content.",
    "nearest_pairs": "Самые близкие пары из перебора или выборки. Полезно смотреть, не являются ли это дубли.",
    "farthest_pairs": "Самые дальние пары из перебора или выборки. Показывают нижнюю границу похожести.",
    "image_same_path_pairwise": "Сравнение image-векторов внутри одного image_path и между разными image_path.",
    "matched_image_text": "Cosine distance/similarity между image_chunks и связанными text_chunks через text_chunk_id.",
}


def percentile_description(metric: str) -> Optional[str]:
    if not metric.startswith("p") or "_" not in metric:
        return None
    percentile, kind = metric.split("_", 1)
    if len(percentile) != 3 or not percentile[1:].isdigit():
        return None
    p = int(percentile[1:])
    if kind == "distance":
        return f"{p}-й процентиль cosine distance: у {p}% пар distance не выше этого значения."
    if kind == "similarity":
        return f"{p}-й процентиль cosine similarity: у {p}% пар similarity не выше этого значения."
    return None


def metric_description(metric: str) -> str:
    return METRIC_DESCRIPTIONS.get(metric) or percentile_description(metric) or "Метрика из JSON-отчёта."


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    text = str(value)
    return " ".join(text.split())


def truncate(text: str, max_len: int = 120) -> str:
    text = format_value(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def escape_md(text: Any) -> str:
    escaped = format_value(text).replace("\\", "\\\\").replace("|", "\\|")
    return escaped.replace("\n", " ")


def markdown_table(headers: List[str], rows: Iterable[Iterable[Any]]) -> str:
    header_line = "| " + " | ".join(escape_md(h) for h in headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(escape_md(cell) for cell in row) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body])


def metric_table(data: Dict[str, Any], order: Optional[List[str]] = None) -> str:
    keys = order or list(data.keys())
    rows = [
        [key, data.get(key), metric_description(key)]
        for key in keys
        if key in data
    ]
    return markdown_table(["Метрика", "Значение", "Смысл"], rows)


def records_table(records: List[Dict[str, Any]], columns: List[str], max_rows: int) -> str:
    rows = []
    for record in records[:max_rows]:
        rows.append([truncate(record.get(col, ""), 140) for col in columns])
    return markdown_table(columns, rows)


def render_table_section(table_name: str, table_report: Dict[str, Any], max_examples: int) -> List[str]:
    lines = [f"## {table_name}", ""]

    stats = table_report.get("stats", {})
    overview = stats.get("overview", {})
    pairwise = stats.get("pairwise", {})
    if overview:
        lines += ["### Overview", "", SECTION_DESCRIPTIONS["overview"], "", metric_table(overview), ""]
    if pairwise:
        pairwise_order = [
            "mode",
            "pairs",
            "requested_sample_pairs",
            "min_distance",
            "p01_distance",
            "p05_distance",
            "p10_distance",
            "p25_distance",
            "p50_distance",
            "p75_distance",
            "p90_distance",
            "p95_distance",
            "p99_distance",
            "avg_distance",
            "max_distance",
            "min_similarity",
            "p01_similarity",
            "p05_similarity",
            "p10_similarity",
            "p25_similarity",
            "p50_similarity",
            "p75_similarity",
            "p90_similarity",
            "p95_similarity",
            "p99_similarity",
            "avg_similarity",
            "max_similarity",
        ]
        lines += ["### Pairwise", "", SECTION_DESCRIPTIONS["pairwise"], "", metric_table(pairwise, pairwise_order), ""]

    duplicates = table_report.get("duplicates", {})
    duplicate_summary = duplicates.get("summary", {})
    if duplicate_summary:
        lines += ["### Duplicates", "", SECTION_DESCRIPTIONS["duplicates"], "", metric_table(duplicate_summary), ""]
    top_groups = duplicates.get("top_groups", [])
    if top_groups:
        columns = list(top_groups[0].keys())
        lines += ["#### Top Duplicate Groups", "", records_table(top_groups, columns, max_examples), ""]

    nearest = table_report.get("nearest_pairs", [])
    if nearest:
        lines += ["### Nearest Pairs", "", SECTION_DESCRIPTIONS["nearest_pairs"], ""]
        columns = [
            "id_a",
            "id_b",
            "cosine_distance",
            "cosine_similarity",
            "image_path_a",
            "bbox_a",
            "image_path_b",
            "bbox_b",
            "text_chunk_id_a",
            "text_chunk_id_b",
        ]
        if "content_a" in nearest[0]:
            columns = ["id_a", "id_b", "cosine_distance", "cosine_similarity", "content_a", "content_b"]
        lines += [records_table(nearest, columns, max_examples), ""]

    farthest = table_report.get("farthest_pairs", [])
    if farthest:
        lines += ["### Farthest Pairs", "", SECTION_DESCRIPTIONS["farthest_pairs"], ""]
        columns = [
            "id_a",
            "id_b",
            "cosine_distance",
            "cosine_similarity",
            "image_path_a",
            "bbox_a",
            "image_path_b",
            "bbox_b",
            "text_chunk_id_a",
            "text_chunk_id_b",
        ]
        if "content_a" in farthest[0]:
            columns = ["id_a", "id_b", "cosine_distance", "cosine_similarity", "content_a", "content_b"]
        lines += [records_table(farthest, columns, max_examples), ""]

    return lines


def render_report(report: Dict[str, Any], max_examples: int) -> str:
    lines = [
        "# Vector Distance Report",
        "",
        markdown_table(
            ["Параметр", "Значение"],
            [
                ["jdbc", report.get("jdbc", "")],
                ["docker_container", report.get("docker_container", "")],
                ["sample_pairs", report.get("sample_pairs", "")],
                ["exact_threshold", report.get("exact_threshold", "")],
            ],
        ),
        "",
        "## Как читать метрики",
        "",
        markdown_table(
            ["Группа", "Смысл"],
            [
                ["cosine_distance", "1 - cosine_similarity. 0 означает почти одинаковые направления; около 1 означает слабую похожесть; больше 1 возможно при отрицательной similarity."],
                ["cosine_similarity", "Чем ближе к 1, тем похожее векторы. Значения около 0 слабосвязаны, ниже 0 направлены противоположно."],
                ["percentiles", "p50 = медиана, p95/p99 показывают верхний хвост похожести или расстояния."],
                ["mode", "exact перебирает все пары, sample использует случайную выборку и поэтому даёт оценку."],
                ["norm", "L2-норма embedding. Для cosine-поиска обычно ожидается около 1 после нормализации."],
            ],
        ),
        "",
    ]

    tables = report.get("tables", {})
    for table_name, table_report in tables.items():
        lines.extend(render_table_section(table_name, table_report, max_examples))

    same_path = report.get("image_same_path_pairwise", [])
    if same_path:
        lines += ["## Image Same Path Split", "", SECTION_DESCRIPTIONS["image_same_path_pairwise"], ""]
        lines += [
            records_table(
                same_path,
                [
                    "same_image",
                    "pairs",
                    "avg_distance",
                    "p50_distance",
                    "p95_distance",
                    "avg_similarity",
                    "p50_similarity",
                    "p95_similarity",
                    "max_similarity",
                ],
                max_examples,
            ),
            "",
        ]

    matched = report.get("matched_image_text", {})
    if matched:
        lines += ["## Matched Image-Text", "", SECTION_DESCRIPTIONS["matched_image_text"], "", metric_table(matched), ""]

    return "\n".join(lines).rstrip() + "\n"


def default_output_path(input_path: str) -> str:
    root, _ = os.path.splitext(input_path)
    return root + ".md"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render db_vector_distance_report.json as Markdown tables.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to db_vector_distance_report.json.")
    parser.add_argument("--output", default=None, help="Markdown output path. Defaults to input path with .md suffix.")
    parser.add_argument("--max-examples", type=int, default=10, help="Max nearest/farthest/duplicate rows per table.")
    parser.add_argument("--stdout", action="store_true", help="Print Markdown to stdout instead of writing a file.")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        report = json.load(f)

    rendered = render_report(report, max_examples=args.max_examples)
    if args.stdout:
        print(rendered, end="")
        return

    output = args.output or default_output_path(args.input)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8-sig") as f:
        f.write(rendered)
    print(f"Saved Markdown report: {output}")


if __name__ == "__main__":
    main()
