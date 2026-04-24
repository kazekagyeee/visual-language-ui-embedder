from __future__ import annotations

from transformers import get_cosine_schedule_with_warmup


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float = 0.05):
    warmup_steps = int(total_steps * warmup_ratio)
    return get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
