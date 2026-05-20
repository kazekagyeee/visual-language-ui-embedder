# -*- coding: utf-8 -*-
from pathlib import Path

FILES = [
    "evaluation/evaluate_ui_retrieval.py",
    "evaluation/user_scenario_eval.py",
    "scripts/debug_ui_query.py",
    "rag/streamlit_pdf_rag.py",
]

for file in FILES:
    path = Path(file)
    if not path.exists():
        print("[MISS]", file)
        continue

    text = path.read_text(encoding="utf-8")

    text = text.replace(
        'targets=response.get("targets", [])',
        'targets=response.get("primary_targets", response.get("targets", []))'
    )

    text = text.replace(
        "targets=response.get('targets', [])",
        "targets=response.get('primary_targets', response.get('targets', []))"
    )

    path.write_text(text, encoding="utf-8")
    print("[OK]", file)
