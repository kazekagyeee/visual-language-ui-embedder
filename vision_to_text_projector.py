import torch
import torch.nn as nn

class VisualToTextEmbeddingProjector(nn.Module):
    def __init__(self, visual_dim=1024, text_dim=4096, target_dim=4096):
        super().__init__()

        # Стандартный проектор Qwen2.5-VL: MLP (Linear -> GELU -> Linear)
        # Он переводит визуальные эмбеддинги в пространство LLM
        self.vision_projector = nn.Sequential(
            nn.Linear(visual_dim, target_dim, bias=True),
            nn.GELU(),
            nn.Linear(target_dim, target_dim, bias=True)
        )

        # Если текстовый эмбеддинг уже в target_dim, используем Identity.
        # Если нет (например, из другой модели) — проецируем.
        if text_dim != target_dim:
            self.text_projector = nn.Linear(text_dim, target_dim, bias=False)
        else:
            self.text_projector = nn.Identity()

    def forward(self, global_emb, box_embs, text_emb):
        """
        Args:
            global_emb: (B, Dim_V) - Глобальный эмбеддинг картинки
            box_embs:   (B, N, Dim_V) - N эмбеддингов из-под bbox
            text_emb:   (B, Dim_T) - Текстовый эмбеддинг (например, описание задачи или класса)

        Returns:
            triples:    (B, N, 3, Target_Dim) - Тензор троек (Box_i, Global, Text)
        """
        B, N, _ = box_embs.shape

        # 1. Проецируем визуальные эмбеддинги в общее пространство
        # Проецируем глобальный (B, 1, Target_Dim)
        v_global = self.vision_projector(global_emb).unsqueeze(1)

        # Проецируем боксы (B, N, Target_Dim)
        v_boxes = self.vision_projector(box_embs)

        # 2. Проецируем текст (B, 1, Target_Dim)
        t_proj = self.text_projector(text_emb).unsqueeze(1)

        # 3. Собираем тройки (Triples)
        # Для каждого из N боксов мы создаем тройку: [Box_i, Global, Text]

        # Расширяем глобальный эмбеддинг и текст до N штук, чтобы составить пары
        v_global_expanded = v_global.expand(-1, N, -1)  # (B, N, Target_Dim)
        t_proj_expanded = t_proj.expand(-1, N, -1)  # (B, N, Target_Dim)

        # Склеиваем в тензор (B, N, 3, Target_Dim)
        # stack по новой размерности (индекс 2)
        triples = torch.stack([v_boxes, v_global_expanded, t_proj_expanded], dim=2)

        return triples
