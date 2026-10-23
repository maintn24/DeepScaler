#!/usr/bin/env bash
# Chạy DeepScaler thay cho HPA, với model đã train (giải nén zip từ Colab/Kaggle vào model_states/).
set -euo pipefail
NS=${NS:-boutique}; MODEL_DIR=${MODEL_DIR:-model_states/boutique_cluster}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q pyyaml requests pandas numpy
kubectl -n "$NS" delete hpa --all --ignore-not-found
python -m pipeline.autoscaler --model_dir "$MODEL_DIR" --namespace "$NS" \
    --log "${LOG:-Logs/autoscaler.jsonl}" ${DRY_RUN:+--dry_run}
