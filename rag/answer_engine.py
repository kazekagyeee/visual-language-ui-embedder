# -*- coding: utf-8 -*-

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class AnswerEngine:
    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

    def generate(self, question: str, context: str) -> str:
        prompt = f"""
Ты помощник по инструкции 1С/ERP.

Отвечай СТРОГО по контексту.
Не выдумывай.
Если данных не хватает, напиши: "В найденном контексте этого нет".

Формат ответа:
1. Короткий ответ.
2. Шаги.
3. Где смотреть: страницы из контекста.

Вопрос:
{question}

Контекст:
{context}

Ответ:
"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=350,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Ответ:" in full_text:
            return full_text.split("Ответ:", 1)[-1].strip()

        return full_text.strip()
