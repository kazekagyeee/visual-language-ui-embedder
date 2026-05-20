# -*- coding: utf-8 -*-

import re
from difflib import SequenceMatcher

from rag.ocr_cleanup import cleanup_ocr_text


def normalize(text):
    text = cleanup_ocr_text(text)
    text = str(text).lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def words(text):
    return [w for w in normalize(text).split() if len(w) > 2]


def fuzzy_ratio(a, b):
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def word_similarity(a, b):
    aw = words(a)
    bw = words(b)

    if not aw or not bw:
        return 0.0

    scores = []

    for target_word in aw:
        best = 0.0

        for text_word in bw:
            sim = SequenceMatcher(None, target_word, text_word).ratio()
            best = max(best, sim)

        scores.append(best)

    return sum(scores) / max(1, len(scores))


def mandatory_token_ok(target, text):
    target_n = normalize(target)
    text_n = normalize(text)

    # Не требуем буквальное совпадение, а проверяем fuzzy-наличие ключевого слова.
    rules = ["монитор", "заявк", "создат", "заполн", "инн", "контрагент"]

    for key in rules:
        if key in target_n:
            return any(SequenceMatcher(None, key, w).ratio() >= 0.72 for w in words(text_n))

    return True


def is_noise_item(item):
    text = normalize(item.get("text", ""))

    if "заполтенже" in text:
        return True

    if "реквизитов" in text and "контрагента" in text:
        return True

    if len(text) < 2:
        return True

    if text in {"и", "в", "на", "по", "из", "для", "как", "что", "стр", "рис", "ok", "оk", "еще", "закрыть"}:
        return True

    if text.isdigit():
        return True

    if len(text.split()) > 9:
        return True

    if "000000" in text:
        return True

    return False


def match_score(target, text):
    target_n = normalize(target)
    text_n = normalize(text)

    if not target_n or not text_n:
        return 0.0

    if not mandatory_token_ok(target_n, text_n):
        return 0.0

    if target_n == text_n:
        return 1.0

    if target_n in text_n:
        return 0.95

    if text_n in target_n:
        return 0.80

    full = fuzzy_ratio(target_n, text_n)
    word = word_similarity(target_n, text_n)

    return max(full, word)


def extract_targets(response):
    targets = []

    for target in response.get("targets", []):
        targets.append(target)

    for step in response.get("steps", []):
        targets.extend(re.findall(r"«([^»]{2,100})»", step))

    result = []
    seen = set()

    for target in targets:
        key = normalize(target)

        if key and key not in seen:
            seen.add(key)
            result.append(target)

    return result


def chain_order(text):
    text = normalize(text)

    if text == "входной контроль":
        return 10
    if "арм" in text and "входной" in text and "контрол" in text:
        return 20
    if "заявк" in text and "контрол" in text:
        return 30
    if text == "создать":
        return 40
    if "создать" in text:
        return 45
    if "контрагент" in text:
        return 10
    if text == "инн" or "начните отсюда" in text:
        return 20
    if "заполнить" in text:
        return 30
    if "интернет" in text and "поддерж" in text and "пользоват" in text:
        return 10
    if "монитор" in text and "интернет" in text:
        return 20
    if "подключ" in text and "интернет" in text:
        return 30

    return 100


def choose_best_for_target(target, results, used):
    exact = []
    fuzzy = []

    for result in results:
        item = result["item"]

        if is_noise_item(item):
            continue

        text = item.get("text", "")
        text_n = normalize(text)

        key = (
            item.get("screenshot_image"),
            text_n,
        )

        if key in used:
            continue

        m = match_score(target, text)

        if m < 0.50:
            continue

        score = (
            m * 1.7
            + result.get("score", 0) * 0.30
            + result.get("target_score", 0) * 0.25
        )

        candidate = (score, result)

        if normalize(target) == text_n:
            exact.append(candidate)
        else:
            fuzzy.append(candidate)

    candidates = exact or fuzzy

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (
            -x[0],
            x[1]["item"].get("page", 9999),
            x[1]["item"].get("screenshot_idx", 9999),
        )
    )

    best = dict(candidates[0][1])
    best["semantic_score"] = float(candidates[0][0])
    best["matched_target"] = target

    return best


def build_ui_semantic_results(query, response, results, limit=8):
    targets = extract_targets(response)

    if not targets:
        return []

    selected = []
    used = set()

    for target in targets:
        best = choose_best_for_target(
            target=target,
            results=results,
            used=used,
        )

        if best is None:
            continue

        item = best["item"]

        used.add(
            (
                item.get("screenshot_image"),
                normalize(item.get("text", "")),
            )
        )

        selected.append(best)

    selected.sort(
        key=lambda x: (
            chain_order(x["item"].get("text", "")),
            x["item"].get("page", 9999),
            x["item"].get("screenshot_idx", 9999),
        )
    )

    return selected[:limit]
