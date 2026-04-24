from __future__ import annotations

from pathlib import Path
import json


def save_qualitative_report(output_path: str | Path, rows: list[dict]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
