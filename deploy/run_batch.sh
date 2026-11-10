#!/usr/bin/env bash
# Chạy nối tiếp nhiều lần online (deploy/05_online_run.sh), tiếp tục nếu một lần lỗi, ghi tóm tắt.
#   deploy/run_batch.sh hpa ds aci hpa ds aci          # thứ tự xen kẽ, mỗi lần ~1 h 40 (FRESH=1)
#   ACI_MODEL_DIR=model_states/ds_demand deploy/run_batch.sh aci aci
# Biến môi trường được truyền nguyên cho 05_online_run.sh (FRESH, SLO, EPS, PACER, MU, CAMPAIGN, ...).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p Logs
SUM=Logs/batch_$(date +%Y%m%d_%H%M).txt
for AS in "$@"; do
  echo "$(date '+%F %T') start $AS" | tee -a "$SUM"
  if deploy/05_online_run.sh "$AS"; then st=ok; else st=FAILED; fi
  echo "$(date '+%F %T') $st $AS -> $(cat Logs/CURRENT_RUN_DIR 2>/dev/null)" | tee -a "$SUM"
  python -m pipeline.index_runs >/dev/null 2>&1 || true
done
echo "summary: $SUM"
