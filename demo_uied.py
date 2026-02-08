"""
Демо-скрипт использования UIED детектора
Принимает картинку и возвращает bbox'ы компонентов, готовые для Qwen2_5_BoxEncoder
"""

import torch
from PIL import Image
from uied_detector import UIEDDetector


def main():
    # --- 1) Инициализация детектора ---
    detector = UIEDDetector(
        resized_height=800,
        key_params={
            'min-grad': 10,
            'ffl-block': 5,
            'min-ele-area': 50,
            'merge-contained-ele': True,
            'merge-line-to-paragraph': True,
            'remove-bar': True
        }
    )
    
    # --- 2) Загрузка изображения ---
    # Замени путь на свой
    image_path = "input_images/image_26_1.png"
    image = Image.open(image_path)
    
    # --- 3) Детекция bbox'ов ---
    bboxes = detector.detect(image, max_dist=20)
    
    print(f"✅ Найдено {len(bboxes)} UI компонентов")
    print("\n📦 Bbox'ы (нормализованные координаты [x1, y1, x2, y2]):")
    for i, bbox in enumerate(bboxes):
        print(f"  {i+1}. {bbox}")
    
    # --- 4) Подготовка для Qwen2_5_BoxEncoder ---
    # Конвертируем в тензор нужного формата
    # Для одного изображения: (1, n_boxes, 4)
    if len(bboxes) > 0:
        boxes_tensor = torch.tensor([bboxes])  # добавляем batch dimension
        print(f"\n🔢 Тензор для Qwen2_5_BoxEncoder:")
        print(f"   Shape: {boxes_tensor.shape}")  # (1, n_boxes, 4)
        print(f"   Dtype: {boxes_tensor.dtype}")
        print(f"\n   Первые 3 бокса:")
        print(boxes_tensor[0, :3])  # показываем первые 3 бокса
        
        # Пример использования с Qwen2_5_BoxEncoder (закомментировано т.к. нужна модель)
        """
        from box_aware_visual_encoder import Qwen2_5_BoxEncoder
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Инициализация энкодера
        encoder = Qwen2_5_BoxEncoder(
            img_size=224,
            patch_size=14,
            embed_dim=1024,
            depth=12,
            num_heads=16,
            n_boxes=len(bboxes)  # количество боксов
        ).to(device)
        
        # Препроцессинг изображения (resize + нормализация)
        # ... (здесь нужна ваша логика препроцессинга)
        
        # Forward pass
        g_emb, b_embs = encoder(img_tensor, boxes_tensor.to(device))
        print(f"Global embedding: {g_emb.shape}")
        print(f"Box embeddings: {b_embs.shape}")
        """
    else:
        print("⚠️ Компоненты не найдены")


if __name__ == "__main__":
    main()
