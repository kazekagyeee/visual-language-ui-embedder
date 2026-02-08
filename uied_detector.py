"""
UIED Detector - детектор UI-компонентов на основе UIED (CV метод)
Возвращает bbox'ы компонентов в нормализованном формате для Qwen2_5_BoxEncoder
"""

import os
import json
import tempfile
from os.path import join as pjoin
from typing import List
from PIL import Image

from uied_cv import ip_region_proposal as ip


class UIEDDetector:
    """
    Детектор UI-компонентов на основе UIED (CV метод)
    Возвращает bbox'ы в нормализованном формате [x1, y1, x2, y2] где значения от 0 до 1
    """

    def __init__(self, resized_height=800, key_params=None):
        """
        Args:
            resized_height: Высота для ресайза изображения перед обработкой
            key_params: Параметры UIED (dict)
        """
        self.resized_height = resized_height
        self.key_params = key_params or {
            'min-grad': 10,
            'ffl-block': 5,
            'min-ele-area': 50,
            'merge-contained-ele': True,
            'merge-line-to-paragraph': True,
            'remove-bar': True
        }

    def detect(self, image: Image.Image, max_dist=20) -> List[List[float]]:
        """
        Запускает UIED и возвращает список bbox'ов в нормализованном формате
        
        Args:
            image: PIL изображение
            max_dist: Максимальное расстояние для склейки близких боксов
            
        Returns:
            List[List[float]]: Список bbox'ов [[x1, y1, x2, y2], ...] где координаты нормализованы (0-1)
        """
        img_width, img_height = image.size
        
        # --- 1) Сохраняем изображение во временную папку ---
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = pjoin(tmpdir, "input.png")
            output_dir = pjoin(tmpdir, "out")
            os.makedirs(output_dir, exist_ok=True)

            image.save(input_path)

            # --- 2) Запуск UIED ---
            ip.compo_detection(
                input_img_path=input_path,
                output_root=output_dir,
                uied_params=self.key_params,
                classifier=None,
                resize_by_height=None,
                show=False
            )

            # --- 3) UIED сохраняет промежуточные файлы ---
            # основной результат лежит в out/ip/{input_name}.json
            input_name = os.path.splitext(os.path.basename(input_path))[0]
            compo_json_path = pjoin(output_dir, "ip", f"{input_name}.json")

            if not os.path.exists(compo_json_path):
                print("⚠️ UIED did not generate compo.json")
                return []

            # --- 4) Читаем JSON ---
            with open(compo_json_path, "r") as f:
                compo_info = json.load(f)

            raw_boxes = []
            for comp in compo_info.get("compos", []):
                x1, y1 = comp["column_min"], comp["row_min"]
                x2, y2 = comp["column_max"], comp["row_max"]
                raw_boxes.append([x1, y1, x2, y2])

            # --- 5) Объединяем рядом стоящие боксы ---
            merged_boxes = self._merge_boxes(raw_boxes, max_dist=max_dist)

            # --- 6) Нормализуем координаты (0-1) ---
            normalized_boxes = []
            for box in merged_boxes:
                x1, y1, x2, y2 = box
                norm_box = [
                    x1 / img_width,
                    y1 / img_height,
                    x2 / img_width,
                    y2 / img_height
                ]
                normalized_boxes.append(norm_box)

            return normalized_boxes

    def _boxes_close(self, a, b, max_dist):
        """Определяет находятся ли сегменты рядом"""
        # вертикальное перекрытие
        vert_overlap = min(a[3], b[3]) - max(a[1], b[1])
        if vert_overlap <= 0:
            return False

        # горизонтальное расстояние
        dist = min(abs(a[0] - b[2]), abs(b[0] - a[2]))

        return dist < max_dist

    def _merge_boxes(self, boxes, max_dist=20):
        """Объединяет рядом стоящие боксы"""
        merged = True
        while merged:
            merged = False
            new = []
            while boxes:
                a = boxes.pop(0)
                merged_with_a = False

                for i, b in enumerate(boxes):
                    if self._boxes_close(a, b, max_dist=max_dist):
                        nx1 = min(a[0], b[0])
                        ny1 = min(a[1], b[1])
                        nx2 = max(a[2], b[2])
                        ny2 = max(a[3], b[3])
                        new.append([nx1, ny1, nx2, ny2])
                        boxes.pop(i)
                        merged_with_a = True
                        merged = True
                        break

                if not merged_with_a:
                    new.append(a)

            boxes = new

        return boxes
