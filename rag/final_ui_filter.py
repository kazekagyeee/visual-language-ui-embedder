# -*- coding: utf-8 -*-
from __future__ import annotations


def _norm(x: str) -> str:
    return (x or "").lower().replace("ё", "е").strip().rstrip(":").strip()


def _score_match(target: str, text: str, matched: str) -> int:
    nt = _norm(target)
    ntext = _norm(text)
    nmatched = _norm(matched)

    if not nt or not ntext:
        return 0

    # лучший случай: reranker явно сопоставил с нужной целью
    if nmatched == nt and ntext == nt:
        return 100

    if ntext == nt:
        return 90

    if nmatched == nt:
        return 80

    # важно: "арм входной контроль" НЕ должен побеждать target "входной контроль"
    if nt in ntext:
        extra = ntext.replace(nt, "").strip()
        if len(extra) <= 3:
            return 60
        return 30

    if ntext in nt:
        return 50

    return 0


def final_filter_ui_results(results, response=None, limit=8, targets=None, **kwargs):
    if not results:
        return []

    primary = []
    if targets:
        primary = targets
    elif response:
        primary = response.get("primary_targets") or response.get("targets") or []

    selected = []
    used_texts = set()
    used_targets = set()

    # 1. Сначала берем лучший элемент на каждую цель
    for target in primary:
        nt = _norm(target)

        best = None
        best_rank = -1
        best_sem = -999.0

        for r in results:
            item = r.get("item", {})
            text = item.get("text", "")
            matched = r.get("matched_target", "")

            rank = _score_match(target, text, matched)
            if rank <= 0:
                continue

            sem = float(r.get("semantic_score", 0) or r.get("score", 0) or 0)

            if rank > best_rank or (rank == best_rank and sem > best_sem):
                best = r
                best_rank = rank
                best_sem = sem

        if best:
            text_key = _norm(best.get("item", {}).get("text", ""))
            target_key = nt

            if text_key not in used_texts and target_key not in used_targets:
                selected.append(best)
                used_texts.add(text_key)
                used_targets.add(target_key)

    # 2. Потом добираем остальные хорошие результаты
    for r in results:
        if len(selected) >= limit:
            break

        item = r.get("item", {})
        text = item.get("text", "")
        ntext = _norm(text)

        if not ntext or ntext in used_texts:
            continue

        if len(ntext) <= 2:
            continue

        # не добираем огромные merged_text, если они не совпадают с целью
        if ntext.count(" ") > 8 and not any(_norm(t) == ntext for t in primary):
            continue

        selected.append(r)
        used_texts.add(ntext)

    return selected[:limit]

