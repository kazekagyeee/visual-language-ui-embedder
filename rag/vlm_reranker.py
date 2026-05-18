# -*- coding: utf-8 -*-

from dataclasses import dataclass


@dataclass
class VLMRerankerConfig:
    enabled: bool = False


class VLMReranker:

    def __init__(self, config=None):

        self.config = config or VLMRerankerConfig()

        self.available = False

        if not self.config.enabled:
            return

        try:
            from transformers import (
                AutoProcessor,
                Qwen2_5_VLForConditionalGeneration
            )

            self.available = True

        except Exception as exc:
            print(exc)

    def rerank(self, query, candidates):

        # пока возвращаем как есть
        # но архитектурно VLM уже встроен

        return candidates