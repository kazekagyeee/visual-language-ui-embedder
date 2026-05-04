#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
python training/train_short_siamese.py --config configs/short_siamese.yaml
