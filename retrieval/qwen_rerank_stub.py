from __future__ import annotations

from typing import Any, Dict, List


def rerank_with_qwen_long_vectors(query_long_vec: list[float], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Заглушка для rerank.

    Сюда можно подключить текущие Qwen long embeddings из output/embeddings.json.
    Логика: для top-k из short index пересчитать similarity по длинным Qwen-векторам
    или заново прогнать Headless Qwen LLM для более точной оценки.
    """
    return candidates
