import json
import random
from pathlib import Path

import streamlit as st
import torch
import yaml

from models.short_siamese_encoder import ShortSiameseEncoder


def fake_embedding(text: str, dim: int = 4):
    random.seed(abs(hash(text)) % (10**8))
    return [round(random.random(), 4) for _ in range(dim)]


def cosine(a, b):
    a = torch.tensor(a, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


@st.cache_resource
def load_all():
    with open("configs/short_siamese.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = ShortSiameseEncoder.load(cfg["paths"]["checkpoint"], map_location="cpu")
    model.eval()

    with open(cfg["paths"]["index"], "r", encoding="utf-8") as f:
        index = json.load(f)

    return cfg, model, index


st.set_page_config(page_title="Short Siamese RAG", layout="wide")
st.title("Short Siamese RAG по PDF-инструкции")

cfg, model, index = load_all()

query = st.text_input("Вопрос", "как создать заявку на контроль")
top_k = st.slider("Количество результатов", 1, 10, 5)

if st.button("Искать"):
    query_vec = fake_embedding("TEXT:" + query, cfg["model"]["input_dim"])

    with torch.no_grad():
        query_tensor = torch.tensor([query_vec], dtype=torch.float32)
        query_short = model.encode_text(query_tensor)[0].tolist()

    scored = []
    for item in index:
        score = cosine(query_short, item["short_vec"])
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)

    context = []

    for score, item in scored[:top_k]:
        with st.expander(f"Страница {item['page']} | score={score:.4f}"):
            st.write(item["text"])
        context.append(item["text"])

    st.subheader("RAG-ответ")
    st.write("Ниже найденный контекст. На следующем шаге сюда можно подключить Qwen/LLM для генерации ответа.")
    st.text_area("Контекст", "\n\n---\n\n".join(context), height=300)
