# -*- coding: utf-8 -*-
from pathlib import Path

FILES = [
    "scripts/debug_ui_query.py",
    "evaluation/evaluate_ui_retrieval.py",
    "evaluation/user_scenario_eval.py",
    "rag/streamlit_pdf_rag.py",
]

IMPORT_LINE = "from rag.final_ui_filter import final_filter_ui_results\n"


def patch_file(path: Path):
    if not path.exists():
        print("[MISS]", path)
        return

    text = path.read_text(encoding="utf-8")

    if "final_filter_ui_results" not in text:
        lines = text.splitlines(True)
        insert_at = 0
        for i, line in enumerate(lines):
            if line.startswith("from ") or line.startswith("import "):
                insert_at = i + 1
        lines.insert(insert_at, IMPORT_LINE)
        text = "".join(lines)

    patterns = [
        "final = build_ui_semantic_results(\n            query=query,\n            response=response,\n            results=raw,\n            limit=8,\n        )",
        "final = build_ui_semantic_results(\n        query=query,\n        response=response,\n        results=raw,\n        limit=8,\n    )",
        "results = build_ui_semantic_results(\n        query=query,\n        response=response,\n        results=raw_results,\n        limit=8,\n    )",
        "results = build_ui_semantic_results(\n            query=query,\n            response=response,\n            results=raw_results,\n            limit=8,\n        )",
    ]

    changed = False

    for pattern in patterns:
        if pattern in text and "final_filter_ui_results" not in text[text.find(pattern):text.find(pattern)+600]:
            replacement = pattern + "\n\n    final = final_filter_ui_results(\n        final,\n        targets=response.get(\"targets\", []),\n        limit=6,\n    )"
            if pattern.startswith("results ="):
                replacement = pattern + "\n\n    results = final_filter_ui_results(\n        results,\n        targets=response.get(\"targets\", []),\n        limit=6,\n    )"

            text = text.replace(pattern, replacement)
            changed = True

    path.write_text(text, encoding="utf-8")
    print("[OK]", path, "changed=", changed)


def main():
    for file in FILES:
        patch_file(Path(file))


if __name__ == "__main__":
    main()
