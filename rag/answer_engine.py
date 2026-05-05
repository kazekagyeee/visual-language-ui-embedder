# -*- coding: utf-8 -*-

import re


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[страница\s+\d+,\s*блок\s*\d+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class AnswerEngine:
    def __init__(self, model_name=None):
        pass

    def generate(self, question: str, context: str) -> str:
        blocks = [b.strip() for b in context.split("---") if b.strip()]

        unique_blocks = []
        seen = set()

        for block in blocks:
            key = normalize_text(block)
            if key in seen:
                continue
            seen.add(key)
            unique_blocks.append(block)

        if not unique_blocks:
            return "В найденном контексте нет подходящей информации."

        pages = []
        answer = []

        answer.append("### Короткий ответ")
        answer.append("Нашлись следующие релевантные фрагменты инструкции:")

        answer.append("\n### Найденные шаги / фрагменты")

        for i, block in enumerate(unique_blocks[:4], start=1):
            page_match = re.search(r"Страница\s+(\d+)", block)
            if page_match:
                pages.append(page_match.group(1))

            text = re.sub(r"\[Страница\s+\d+,\s*блок\s*\d+\]", "", block).strip()
            answer.append(f"\n**{i}.** {text}")

        if pages:
            pages = sorted(set(pages), key=lambda x: int(x))
            answer.append("\n### Где смотреть")
            answer.append("Страницы: " + ", ".join(pages))

        return "\n".join(answer)
