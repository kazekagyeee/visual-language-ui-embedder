import argparse
import json
import random

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/short_siamese.yaml")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ShortSiameseEncoder.load(cfg["paths"]["checkpoint"], map_location=device).to(device)
    model.eval()

    with open(cfg["paths"]["index"], "r", encoding="utf-8") as f:
        index = json.load(f)

    query_vec = fake_embedding("TEXT:" + args.query, cfg["model"]["input_dim"])

    with torch.no_grad():
        query_tensor = torch.tensor([query_vec], dtype=torch.float32).to(device)
        query_short = model.encode_text(query_tensor)[0].cpu().tolist()

    scored = []
    for item in index:
        score = cosine(query_short, item["short_vec"])
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)

    for score, item in scored[: args.top_k]:
        print("=" * 80)
        print(f"score={score:.4f} page={item['page']} id={item['id']}")
        print(item["text"][:1000].replace("\n", " "))


if __name__ == "__main__":
    main()
