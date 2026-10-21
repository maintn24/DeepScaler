#!/usr/bin/env bash
# Bơm tải trace FIFA (Locust) trong HOURS giờ rồi xuất CSV metric + call graph từ Prometheus.
# 1 vòng trace = 903 x 5 s ≈ 75 phút. Nên thu >= 6 giờ (≈ 720 mẫu ở step 30 s).
set -euo pipefail
HOURS=${HOURS:-6}; STEP=${STEP:-30}; NS=${NS:-boutique}
HOST=${HOST:-http://localhost:31000}; PROM=${PROM:-http://localhost:30090}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
RUN=${RUN:-run_$(date +%Y%m%d_%H%M)}
mkdir -p data/raw Logs
python -c "import locust" 2>/dev/null || pip install -q locust requests pandas numpy
START=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$START] bắt đầu bơm tải $HOURS giờ -> $HOST"
locust -f pipeline/locustfile.py --headless --host "$HOST" --run-time "${HOURS}h" \
       --csv "Logs/${RUN}_locust" > "Logs/${RUN}_locust.log" 2>&1 || true
END=$(date '+%Y-%m-%d %H:%M:%S')
python -m pipeline.collect --prom "$PROM" --namespace "$NS" --start "$START" --end "$END" \
       --step "$STEP" --out "data/raw/${RUN}.csv" --adj_out "data/raw/${RUN}_od.npy"
python -m pipeline.build_dataset --raw "data/raw/${RUN}.csv" --od_adj "data/raw/${RUN}_od.npy" \
       --out_dir data/boutique_cluster
echo "Xong. Upload data/raw/${RUN}.csv (+ _od.npy) lên Colab/Kaggle, chọn DATA_SOURCE='cluster'."
