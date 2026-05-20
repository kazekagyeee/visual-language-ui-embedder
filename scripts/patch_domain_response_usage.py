# -*- coding: utf-8 -*-
from pathlib import Path

PATCHES = [
    "scripts/debug_ui_query.py",
    "evaluation/evaluate_ui_retrieval.py",
    "evaluation/user_scenario_eval.py",
    "rag/streamlit_pdf_rag.py",
]

IMPORT_LINE = "from rag.domain_response import enrich_response_with_domain\n"

PATTERNS = [
    "response = AnswerEngine().build_response(query, text_results)",
    "response = answer_engine.build_response(query, text_results)",
    "response = load_answer_engine().build_response(\n            query=query,\n            results=results,\n        )",
    "response = AnswerEngine().build_response(query=query, results=results)",
    "response = AnswerEngine().build_response(query, results)",
]


def patch_file(path: Path):
    if not path.exists():
        print("[MISS]", path)
        return

    text = path.read_text(encoding="utf-8")

    if "enrich_response_with_domain" not in text:
        lines = text.splitlines(True)
        insert_at = 0
        for i, line in enumerate(lines):
            if line.startswith("from ") or line.startswith("import "):
                insert_at = i + 1
        lines.insert(insert_at, IMPORT_LINE)
        text = "".join(lines)

    changed = False

    for pattern in PATTERNS:
        if pattern in text and pattern + "\n    response = enrich_response_with_domain(query, response)" not in text:
            text = text.replace(
                pattern,
                pattern + "\n    response = enrich_response_with_domain(query, response)",
            )
            changed = True

    # streamlit nested indentation case
    pattern = "response = load_answer_engine().build_response(\n            query=query,\n            results=results,\n        )"
    if pattern in text and "response = enrich_response_with_domain(query, response)" not in text:
        text = text.replace(
            pattern,
            pattern + "\n\n        response = enrich_response_with_domain(query, response)",
        )
        changed = True

    path.write_text(text, encoding="utf-8")
    print("[OK]", path, "changed=", changed)


def main():
    for p in PATCHES:
        patch_file(Path(p))


if __name__ == "__main__":
    main()
