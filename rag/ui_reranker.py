# -*- coding: utf-8 -*-

import re
from difflib import SequenceMatcher
from rag.ocr_cleanup import cleanup_ocr_text


def normalize(text):
    text = cleanup_ocr_text(text)
    text = str(text).lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text):
    return set(normalize(text).split())


def fuzzy_ratio(a, b):
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def is_noise_item(item):
    text = normalize(item.get("text", ""))

    if len(text) < 2:
        return True

    if text in {
        "и", "в", "на", "по", "из", "для", "как", "что",
        "стр", "рис", "ok", "оk", "еще", "закрыть"
    }:
        return True

    if text.isdigit():
        return True

    if len(text.split()) > 7:
        return True

    if "000000" in text:
        return True

    if "от " in text and any(ch.isdigit() for ch in text):
        return True

    return False


def mandatory_token_ok(target, text):
    target = normalize(target)
    text = normalize(text)

    rules = {
        "монитор": ["монитор"],
        "заявк": ["заявк"],
        "создать": ["создат"],
        "заполнить": ["заполн"],
        "инн": ["инн"],
        "контрагент": ["контрагент"],
    }

    for key, required_parts in rules.items():
        if key in target:
            return any(part in text for part in required_parts)

    return True


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
        return 0.86

    if text_n in target_n:
        return 0.72

    fuzzy = fuzzy_ratio(target_n, text_n)

    target_tokens = tokens(target_n)
    text_tokens = tokens(text_n)

    if not target_tokens or not text_tokens:
        return 0.0

    overlap = len(target_tokens & text_tokens) / max(1, len(target_tokens))

    return max(overlap, fuzzy * 0.60)


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

    if text.startswith("создать "):
        return 45

    if "контрагент" in text:
        return 10

    if text == "инн" or "начните отсюда" in text:
        return 20

    if "заполнить" in text:
        return 30

    if "интернет-поддержка пользователей" in text:
        return 10

    if "монитор интернет-поддержки" in text:
        return 20

    if "подключить интернет-поддержку" in text:
        return 30

    return 100


def choose_best_for_target(target, results, used):
    target_n = normalize(target)

    exact_candidates = []
    soft_candidates = []

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

        score = match_score(target, text)

        if score < 0.55:
            continue

        final = (
            score * 1.5
            + result.get("score", 0) * 0.30
            + result.get("target_score", 0) * 0.30
        )

        candidate = (final, result)

        if text_n == target_n:
            exact_candidates.append(candidate)
        else:
            soft_candidates.append(candidate)

    candidates = exact_candidates or soft_candidates

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


def build_ui_semantic_results(query, response, results, limit=6):
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
