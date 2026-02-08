from box_aware_visual_encoder import Qwen2_5_BoxEncoder
from headless_qwen_llm import HeadlessQwen2_5
from vision_to_text_projector import VisualToTextEmbeddingProjector

# -----------------------------------------------------------
# Тестовый запуск ViT
# -----------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# Инициализация модели (параметры можно менять под нужный конфиг Qwen)
model = Qwen2_5_BoxEncoder(
    img_size=224,
    patch_size=14,
    embed_dim=1024,
    depth=12,  # Уменьшил глубину для теста
    num_heads=16,
    n_boxes=3
).to(device)

# Данные
dummy_img = torch.randn(2, 3, 224, 224).to(device)
# 3 бокса для каждого изображения в батче [x1, y1, x2, y2]
dummy_boxes = torch.tensor([
    [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0], [0.2, 0.2, 0.8, 0.8]],
    [[0.1, 0.1, 0.3, 0.3], [0.0, 0.0, 1.0, 1.0], [0.7, 0.7, 0.9, 0.9]]
]).to(device)

g_emb, b_embs = model(dummy_img, dummy_boxes)

print(f"Global Embedding shape: {g_emb.shape}")  # (2, 1024)
print(f"Box Embeddings shape:   {b_embs.shape}")  # (2, 3, 1024)
print("Success!")


# -----------------------------------------------------------
# Тестовый пример использования UIED
# -----------------------------------------------------------

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

# -----------------------------------------------------------
# Тестовый пример использования VisualToTextEmbeddingProjector
# -----------------------------------------------------------

# Параметры Qwen2.5-VL-7B
VIS_DIM = 1024
LLM_DIM = 3584  # Примерная размерность Qwen2.5-7B
N_BOXES = 5

projector = VisualToTextEmbeddingProjector(
    visual_dim=VIS_DIM,
    text_dim=LLM_DIM,
    target_dim=LLM_DIM
)

# Имитация входа из энкодера
batch_global = torch.randn(2, VIS_DIM)
batch_boxes = torch.randn(2, N_BOXES, VIS_DIM)
batch_text = torch.randn(2, LLM_DIM)

sd = torch.load("qwen2.5_vl_model.bin")
# Маппинг
projector.vision_projector[0].weight.data = sd['visual.vision_projector.0.weight']
projector.vision_projector[2].weight.data = sd['visual.vision_projector.2.weight']

result_triples = projector(batch_global, batch_boxes, batch_text)

print(f"Input Box Embs: {batch_boxes.shape}")
print(f"Output Triples Shape: {result_triples.shape}")
# Ожидаем (Batch=2, N=5, Triple_Size=3, Dim=3584)

# Пример доступа к первой тройке для первого изображения:
# triple_0 = result_triples[0, 0] # Содержит [Box_0, Global, Text]


# -----------------------------------------------------------
# Тестовый пример использования энкодера
# -----------------------------------------------------------


# 1. Конфигурация (в стиле Qwen2.5-VL)
device = "cuda" if torch.cuda.is_available() else "cpu"
VIS_DIM = 1024
LLM_DIM = 3584  # Qwen2.5-7B
N_BOXES = 3

qwen_config = {
    'hidden_size': LLM_DIM,
    'num_heads': 28,
    'num_key_value_heads': 4,
    'intermediate_size': 18944,
    'num_layers': 4,      # Для примера возьмем 4 слоя, в оригинале 28
    'rms_norm_eps': 1e-6,
    'rope_theta': 1000000.0,
    'use_qkv_bias': True
}

# 2. Инициализация всех компонентов
encoder = Qwen2_5_BoxEncoder(
    img_size=224, patch_size=14, embed_dim=VIS_DIM, n_boxes=N_BOXES
).to(device)

projector = VisualToTextEmbeddingProjector(
    visual_dim=VIS_DIM, text_dim=LLM_DIM, target_dim=LLM_DIM
).to(device)

llm_fuser = HeadlessQwen2_5(qwen_config).to(device)

# 3. Подготовка входных данных
batch_size = 2
dummy_img = torch.randn(batch_size, 3, 224, 224).to(device)
# Нормализованные координаты [x1, y1, x2, y2]
dummy_boxes = torch.tensor([
    [[0.1, 0.1, 0.4, 0.4], [0.5, 0.5, 0.9, 0.9], [0.0, 0.0, 1.0, 1.0]],
    [[0.2, 0.2, 0.3, 0.3], [0.1, 0.6, 0.4, 0.8], [0.5, 0.1, 0.7, 0.3]]
]).to(device)

# Текстовый эмбеддинг (например, извлеченный из замороженной Qwen или CLIP)
# Допустим, это эмбеддинг вопроса "Что это за объект?"
dummy_text_queries = torch.randn(batch_size, LLM_DIM).to(device)

# -----------------------------------------------------------
# ПРОХОД (FORWARD PASS)
# -----------------------------------------------------------

# Шаг 1: Энкодер (извлекает признаки с учетом масок внимания)
# g_emb: (B, VIS_DIM)
# b_embs: (B, N, VIS_DIM)
g_emb, b_embs = encoder(dummy_img, dummy_boxes)

# Шаг 2: Проектор (переводит в пространство LLM и формирует тройки)
# triples: (B, N, 3, LLM_DIM) -> [Box_i, Global, Text]
triples = projector(g_emb, b_embs, dummy_text_queries)

# Шаг 3: Headless LLM (запускает "рассуждение" внутри каждой тройки)
# fused_features: (B, N, LLM_DIM)
fused_features = llm_fuser(triples)

print(f"Final shape: {fused_features.shape}")
# Ожидаем (2, 3, 3584) — по одному вектору на каждый бокс