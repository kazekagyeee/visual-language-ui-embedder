from __future__ import annotations

from io import BytesIO
import random

from PIL import Image, ImageEnhance, ImageFilter


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def apply_mild_ui_augmentations(
    image: Image.Image,
    gaussian_blur_prob: float = 0.05,
    compression_noise_prob: float = 0.05,
    brightness_jitter_prob: float = 0.05,
    brightness_jitter_range: float = 0.05,
    jpeg_quality_min: int = 80,
    jpeg_quality_max: int = 95,
) -> Image.Image:
    image = ensure_rgb(image)

    if random.random() < gaussian_blur_prob:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.35))

    if random.random() < brightness_jitter_prob:
        factor = random.uniform(1.0 - brightness_jitter_range, 1.0 + brightness_jitter_range)
        image = ImageEnhance.Brightness(image).enhance(factor)

    if random.random() < compression_noise_prob:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(jpeg_quality_min, jpeg_quality_max))
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")

    return image
