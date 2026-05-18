# -*- coding: utf-8 -*-

import re


def clean_pdf_text(text):
    text = str(text)
    text = text.replace("-\n", "")
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^\d+\s+\d+\s+", "", text)
    return text.strip()


def normalize(text):
    text = str(text).lower().replace("ё", "е")
    text = re.sub(r"[^а-яa-z0-9\s\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def quoted_phrases(text):
    return re.findall(r"«([^»]{2,100})»", text)


class AnswerEngine:
    def build_response(self, query, results):
        if not results:
            return {
                "source": "",
                "page": "",
                "pdf_name": "",
                "short_answer": "По загруженным PDF не удалось найти ответ.",
                "steps": [],
                "targets": [],
                "raw_text": "",
            }

        best = results[0]["item"]

        pdf_name = best.get("pdf_name", "PDF")
        page = best.get("page", "?")
        text = clean_pdf_text(best.get("text", ""))

        steps, targets = self.extract_steps_and_targets(query, text)

        if steps:
            short_answer = "Нашла нужный фрагмент инструкции. Выполните шаги ниже."
        else:
            short_answer = self.make_short_answer(text)

        return {
            "source": f"{pdf_name}, страница {page}",
            "page": page,
            "pdf_name": pdf_name,
            "short_answer": short_answer,
            "steps": steps,
            "targets": targets,
            "raw_text": text,
        }

    def make_short_answer(self, text):
        sentences = re.split(r"(?<=[.!?])\s+", text)
        useful = []

        for s in sentences:
            s = clean_pdf_text(s)

            if len(s) >= 30:
                useful.append(s)

            if len(useful) >= 3:
                break

        return " ".join(useful) if useful else text[:900]

    def extract_steps_and_targets(self, query, text):
        q = normalize(query)
        t_norm = normalize(text)
        t = text.replace("ё", "е")

        # 1. Простое "где найти ..." — показываем только нужный раздел.
        if "где" in q and "найти" in q and "создат" not in q and "заявк" not in q:
            if "входной контроль" in q:
                return (
                    ["Найдите раздел «Входной контроль»."],
                    ["Входной контроль"],
                )

            if "монитор" in q and "интернет" in q:
                return (
                    ["Найдите ссылку «Монитор Интернет-поддержки»."],
                    ["Монитор Интернет-поддержки"],
                )

        # 2. Контрагенты / ИНН.
        if "контрагент" in q or "инн" in q or "контрагент" in t_norm:
            steps = []
            targets = []

            if "создат" in q or "нов" in q or "создайте нового контрагента" in t_norm:
                steps.append("Откройте справочник «Контрагенты».")
                steps.append("Нажмите «Создать».")
                targets.extend(["Контрагенты", "Создать"])

            if "инн" in q or "инн" in t_norm or "реквизит" in q:
                steps.append("Введите ИНН в поле «ИНН» или «Начните отсюда».")
                steps.append("Нажмите кнопку «Заполнить».")
                targets.extend(["ИНН", "Начните отсюда", "Заполнить"])

            if not steps:
                steps.append("Откройте справочник «Контрагенты».")
                targets.append("Контрагенты")

            return self.dedupe(steps), self.dedupe(targets)

        # 3. Входной контроль / заявка.
        steps = []
        targets = []

        m = re.search(
            r"На вкладке\s+(.+?)\s+[–-]\s+(.+?),\s*([А-ЯA-Zа-яa-z0-9 ]+)\.?",
            t,
            flags=re.IGNORECASE,
        )

        if m:
            tab = m.group(1).strip()
            section = m.group(2).strip()
            action = m.group(3).strip()

            steps.append(f"Откройте вкладку «{tab}».")
            steps.append(f"Перейдите в раздел «{section}».")

            targets.extend([tab, section])

            if "заявк" in q or "заявк" in t_norm:
                steps.append("Откройте пункт «Заявки на контроль».")
                targets.append("Заявки на контроль")

            steps.append(f"Нажмите «{action}».")
            targets.append(action)

            return self.dedupe(steps), self.dedupe(targets)

        # 4. Интернет-поддержка и общие кавычки.
        for phrase in quoted_phrases(t):
            low = normalize(phrase)

            if "интернет-поддержка пользователей" in low:
                steps.append(f"Откройте пункт «{phrase}».")
                targets.append(phrase)

            elif "монитор интернет-поддержки" in low:
                steps.append(f"Перейдите по ссылке «{phrase}».")
                targets.append(phrase)

            elif "подключить интернет-поддержку" in low:
                steps.append(f"Нажмите кнопку «{phrase}».")
                targets.append(phrase)

            elif "организации" in low:
                steps.append(f"Откройте справочник «{phrase}».")
                targets.append(phrase)

            elif "контрагенты" in low:
                steps.append(f"Откройте справочник «{phrase}».")
                targets.append(phrase)

            elif "создать" in low:
                steps.append(f"Нажмите «{phrase}».")
                targets.append(phrase)

            elif "заполнить" in low:
                steps.append(f"Нажмите «{phrase}».")
                targets.append(phrase)

        return self.dedupe(steps)[:6], self.dedupe(targets)[:8]

    def dedupe(self, items):
        result = []
        seen = set()

        for item in items:
            item = clean_pdf_text(item)
            key = normalize(item)

            if key and key not in seen:
                seen.add(key)
                result.append(item)

        return result
