#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
python retrieval/build_short_index.py \
  --config configs/short_siamese.yaml \
  --checkpoint checkpoints/short_siamese/best.pt
