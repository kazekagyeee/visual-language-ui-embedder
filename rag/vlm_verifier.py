# -*- coding: utf-8 -*-

from pathlib import Path
from dataclasses import dataclass

from PIL import Image


@dataclass
class VLMVerificationResult:
    available: bool
    score: float
    reason: str


class VLMVerifier:
    def __init__(
        self,
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        enabled=False,
    ):
        self.model_name = model_name
        self.enabled = enabled
        self.available = False
        self.model = None
        self.processor = None

        if not enabled:
            return

        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            self.torch = torch
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
            )

            self.available = True

        except Exception as exc:
            print(f"[VLMVerifier] disabled: {exc}")
            self.available = False

    def verify(self, query, target, image_path):
        """
        Возвращает score 0..1.
        Если VLM недоступна — возвращает нейтральный score.
        """

        if not self.enabled:
            return VLMVerificationResult(
                available=False,
                score=0.5,
                reason="VLM disabled",
            )

        if not self.available:
            return VLMVerificationResult(
                available=False,
                score=0.5,
                reason="VLM unavailable",
            )

        image_path = Path(image_path)

        if not image_path.exists():
            return VLMVerificationResult(
                available=True,
                score=0.0,
                reason="image not found",
            )

        image = Image.open(image_path).convert("RGB")

        prompt = (
            "Ты анализируешь скриншот интерфейса 1С. "
            "Ответь строго одним словом: YES или NO.\n\n"
            f"Вопрос пользователя: {query}\n"
            f"Искомый элемент интерфейса: {target}\n\n"
            "Есть ли на скриншоте этот элемент интерфейса или очень близкий по смыслу элемент?"
        )

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
            )

            inputs = {
                k: v.to(self.model.device)
                for k, v in inputs.items()
                if hasattr(v, "to")
            }

            with self.torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=False,
                )

            output = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]

            out = output.lower()

            if "yes" in out or "да" in out:
                return VLMVerificationResult(
                    available=True,
                    score=1.0,
                    reason=output,
                )

            if "no" in out or "нет" in out:
                return VLMVerificationResult(
                    available=True,
                    score=0.0,
                    reason=output,
                )

            return VLMVerificationResult(
                available=True,
                score=0.5,
                reason=output,
            )

        except Exception as exc:
            return VLMVerificationResult(
                available=False,
                score=0.5,
                reason=str(exc),
            )


def apply_vlm_verification(query, results, verifier, weight=0.25):
    if not verifier.enabled:
        return results

    verified = []

    for result in results:
        item = result["item"]
        target = result.get("matched_target") or item.get("text", "")
        image_path = item.get("screenshot_image")

        vlm = verifier.verify(
            query=query,
            target=target,
            image_path=image_path,
        )

        result = dict(result)
        result["vlm_score"] = vlm.score
        result["vlm_reason"] = vlm.reason
        result["vlm_available"] = vlm.available

        base = float(result.get("semantic_score", result.get("score", 0.0)))
        result["semantic_score"] = base * (1.0 - weight) + vlm.score * weight

        verified.append(result)

    verified.sort(
        key=lambda x: (
            x.get("chain_order", 100),
            -x.get("semantic_score", 0),
        )
    )

    return verified
