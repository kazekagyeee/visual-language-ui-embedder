from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import yaml

from models.short_siamese_encoder import ShortSiameseEncoder
from retrieval.search_short_index import load_metadata, search_numpy


@st.cache_resource
def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_model(checkpoint: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShortSiameseEncoder.load(checkpoint, map_location=device).to(device)
    model.eval()
    return model, device


@st.cache_data
def load_index(matrix_path: str, metadata_path: str):
    return np.load(matrix_path), load_metadata(metadata_path)


def parse_vector(raw: str) -> list[float]:
    raw = raw.strip()
    if not raw:
        raise ValueError("Нужен text_vec. Пока приложение принимает готовый вектор запроса.")
    return json.loads(raw)


st.set_page_config(page_title="Short Siamese UI RAG", layout="wide")
st.title("Short Siamese UI RAG")
st.caption("Быстрый поиск UI-компонентов по коротким text/image embeddings. Qwen можно добавить как reranker.")

config_path = st.sidebar.text_input("Config", "configs/short_siamese.yaml")
checkpoint_path = st.sidebar.text_input("Checkpoint", "checkpoints/short_siamese/best.pt")
top_k = st.sidebar.slider("Top-K", 1, 30, 10)

cfg = load_config(config_path)
model, device = load_model(checkpoint_path)
matrix, metadata = load_index(cfg["retrieval"]["numpy_index_path"], cfg["retrieval"]["metadata_path"])

st.subheader("Запрос")
query_text = st.text_input("Текст запроса", "кнопка входа")
st.info("MVP принимает готовый teacher text_vec. Следующий шаг — подключить генерацию text_vec из текущего Qwen/text pipeline.")
query_vec_raw = st.text_area("Teacher text_vec JSON", height=160, placeholder="[0.01, 0.02, ...]")

if st.button("Найти"):
    try:
        text_vec = torch.tensor(parse_vector(query_vec_raw), dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            query_short = model.encode_text(text_vec).squeeze(0).cpu().numpy()
        results = search_numpy(query_short, matrix, metadata, top_k)
        st.subheader("Результаты")
        for i, item in enumerate(results, start=1):
            with st.container(border=True):
                st.markdown(f"### {i}. {item.get('title', item.get('id', 'item'))}")
                st.write(f"score: `{item['score']:.4f}`")
                if item.get("text"):
                    st.write(item["text"])
                if item.get("meta"):
                    st.json(item["meta"])
    except Exception as exc:
        st.error(str(exc))
