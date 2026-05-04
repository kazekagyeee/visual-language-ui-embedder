import argparse
import json
from pathlib import Path

import torch
import yaml

from models.short_siamese_encoder import ShortSiameseEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/short_siamese.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ShortSiameseEncoder.load(cfg["paths"]["checkpoint"], map_location=device).to(device)
    model.eval()

    items = []
    with open(cfg["data"]["index_items"], "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    output = []

    with torch.no_grad():
        for item in items:
            image_vec = torch.tensor([item["image_vec"]], dtype=torch.float32).to(device)
            short_vec = model.encode_image(image_vec)[0].cpu().tolist()

            output.append(
                {
                    "id": item["id"],
                    "page": item["page"],
                    "text": item["text"],
                    "image_vec": item["image_vec"],
                    "short_vec": short_vec,
                }
            )

    index_path = Path(cfg["paths"]["index"])
    index_path.parent.mkdir(parents=True, exist_ok=True)

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved index: {index_path}")
    print(f"Items: {len(output)}")


if __name__ == "__main__":
    main()
