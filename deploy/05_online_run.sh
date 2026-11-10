#!/usr/bin/env bash
# One online run for Table II / Fig. 10 of the DeepScaler paper (Sec. V-B): 75 min FIFA trace,
# every service starts at 1 pod, then metrics are collected from Prometheus.
#
#   deploy/05_online_run.sh <hpa|ds|slo|mmn|aci>
#
#   hpa  AWS [47] stand-in: Kubernetes HPA, CPU target 50 % (deploy/02_boutique.sh)
#   ds   DeepScaler (pipeline/autoscaler.py, MODEL_DIR, trust threshold Sec. IV-D)
#   slo  SLO [48]: SLO decomposition, rule-based on response time (pipeline/rule_autoscalers.py)
#   mmn  MMN [33]: M/M/n/PS queue per service, non-holistic (pipeline/rule_autoscalers.py)
#   aci  thesis: same STGNN forecast + application-level ACI + SLA budget pacer
#        (pipeline/aci_autoscaler.py). `ds` stays the ORIGINAL DeepScaler for the comparison.
#        ACI_MODEL_DIR (model trained on 'demand', build_dataset --features demand,...; a 'pod' model
#        also runs), MU (default: mu stored in the model), CALIB_CSV (warm start + active set),
#        EPS=0.05 (SLA budget), PACER=on|off (off = fixed alpha = EPS, ACI-only ablation)
#
# Environment (defaults):
#   FRESH=1        delete and rebuild the kind cluster first (~15-20 min), so every run starts from
#                  the same clean state (removes the order / cluster-age effect measured on 04-08/11)
#   RUN_TIME=4500  Locust seconds (one FIFA loop = 75 min)
#   SLO=0.2        end-to-end SLA on frontend p95 (s), for slo / mmn / ds trust threshold
#                  Latency is logged twice: Istio sidecar (Prometheus, needs a bucket boundary at SLO,
#                  deploy/02_boutique.sh) and per request by Locust (runs/<CAMPAIGN>/RUN/RUN_latency.csv, exact).
#   MODEL_DIR=model_states/ds_burst_tuned
#   REF_CSV=<latest runs/*/online_hpa_*/online_hpa_*.csv>  reference run for the SLO decomposition (slo, mmn);
#                  v2_exact sorts after v1, so run hpa first in a new campaign
#   CAMPAIGN=v2_exact   output folder runs/<CAMPAIGN>/<RUN>/ (csv, od/npas, locust, latency, autoscaler, console, time)
#   MU=reports/mu.json                            profiled pod capacity (mmn)
set -euo pipefail
AS=${1:?usage: deploy/05_online_run.sh <hpa|ds|slo|mmn>}
case "$AS" in hpa|ds|slo|mmn|aci) ;; *) echo "unknown autoscaler $AS" >&2; exit 1;; esac
FRESH=${FRESH:-1}; RUN_TIME=${RUN_TIME:-4500}; SLO=${SLO:-0.2}; NS=boutique; CAMPAIGN=${CAMPAIGN:-v2_exact}
MODEL_DIR=${MODEL_DIR:-model_states/ds_burst_tuned}; MU=${MU:-reports/mu.json}
ACI_MODEL_DIR=${ACI_MODEL_DIR:-$MODEL_DIR}; EPS=${EPS:-0.05}; PACER=${PACER:-on}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
CALIB_CSV=${CALIB_CSV:-$(ls data/raw/train_burst_*.csv 2>/dev/null | grep -v -e _od -e _npas | tail -1)}
[ -f ~/dsenv/bin/activate ] && source ~/dsenv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p Logs runs

if [ "$AS" = slo ] || [ "$AS" = mmn ]; then
  REF_CSV=${REF_CSV:-$(ls runs/*/online_hpa_*/online_hpa_*.csv 2>/dev/null | grep -v -e _od -e _npas | tail -1)}
  [ -n "$REF_CSV" ] && [ -f "$REF_CSV" ] || { echo "REF_CSV not found (need one online_hpa_*.csv)" >&2; exit 1; }
  if [ "$AS" = mmn ] && [ ! -f "$MU" ]; then echo "$MU not found" >&2; exit 1; fi
fi
if [ "$AS" = aci ]; then
  [ -f "$ACI_MODEL_DIR/AdapGLA/AdapGLA.pkl" ] || { echo "model not found in $ACI_MODEL_DIR" >&2; exit 1; }
  python -c "import torch" 2>/dev/null || pip install -q torch --index-url https://download.pytorch.org/whl/cpu
fi
if [ "$AS" = ds ]; then
  [ -f "$MODEL_DIR/AdapGLA/AdapGLA.pkl" ] || { echo "model not found in $MODEL_DIR" >&2; exit 1; }
  python -c "import torch" 2>/dev/null || pip install -q torch --index-url https://download.pytorch.org/whl/cpu
fi

if [ "$FRESH" = 1 ]; then
  echo ">> rebuilding the cluster"
  kind delete cluster --name deepscaler 2>/dev/null || true
  deploy/00_kind_cluster.sh && deploy/01_istio_prometheus.sh && deploy/02_boutique.sh
elif [ "$AS" = hpa ]; then
  deploy/02_boutique.sh                          # (re)creates the 10 HPAs
fi
[ "$AS" = hpa ] || kubectl -n "$NS" delete hpa --all --ignore-not-found
kubectl -n "$NS" scale deploy --all --replicas=1
kubectl -n "$NS" rollout status deploy --timeout=600s
curl -sf 'http://localhost:30090/api/v1/query?query=up' >/dev/null || { echo "Prometheus not reachable" >&2; exit 1; }
sleep 120
# mới (10/11/2026): kiểm tra biên bucket tại SLO; cách cũ không có 2 dòng dưới
for i in $(seq 1 20); do curl -s -o /dev/null http://localhost:31000/ || true; done; sleep 30   # create the series
deploy/check_latency_buckets.sh "$SLO" || echo "WARNING: continuing; Istio p95 at the SLO will be approximate (client log is still exact)"

RUN=online_${AS}_$(date +%Y%m%d_%H%M)
RD="runs/${CAMPAIGN}/${RUN}"; mkdir -p "$RD"          # mọi file của lần chạy nằm ở đây
echo "$RUN" > Logs/CURRENT_RUN; echo "$RD" > Logs/CURRENT_RUN_DIR
echo ">> run $RUN"
AS_PID=""
case "$AS" in
  ds)  python -m pipeline.autoscaler --model_dir "$MODEL_DIR" --namespace "$NS" --slo "$SLO" \
           --log "$RD/${RUN}_autoscaler.jsonl" > "$RD/${RUN}_console.log" 2>&1 & AS_PID=$! ;;
  slo) python -m pipeline.rule_autoscalers --policy slo --slo "$SLO" --ref_csv "$REF_CSV" \
           --log "$RD/${RUN}_autoscaler.jsonl" > "$RD/${RUN}_console.log" 2>&1 & AS_PID=$! ;;
  aci) python -m pipeline.aci_autoscaler --model_dir "$ACI_MODEL_DIR" --namespace "$NS" --slo "$SLO" \
           --eps "$EPS" --pacer "$PACER" --period_min "$(python3 -c "print($RUN_TIME/60)")" \
           $( [ -n "$CALIB_CSV" ] && echo --calib_csv "$CALIB_CSV" ) \
           $( python3 -c "import json,sys; m=json.load(open('$ACI_MODEL_DIR/metrics.json'))['meta']; sys.exit(0 if 'mu' in m else 1)" 2>/dev/null || echo --mu "$MU" ) \
           --log "$RD/${RUN}_autoscaler.jsonl" > "$RD/${RUN}_console.log" 2>&1 & AS_PID=$! ;;
  mmn) python -m pipeline.rule_autoscalers --policy mmn --slo "$SLO" --ref_csv "$REF_CSV" --mu "$MU" \
           --log "$RD/${RUN}_autoscaler.jsonl" > "$RD/${RUN}_console.log" 2>&1 & AS_PID=$! ;;
esac
cleanup() { [ -n "$AS_PID" ] && kill "$AS_PID" 2>/dev/null || true; }
trap cleanup EXIT
if [ -n "$AS_PID" ]; then
  sleep 20
  kill -0 "$AS_PID" 2>/dev/null || { echo "autoscaler died:"; tail -20 "$RD/${RUN}_console.log"; exit 1; }
  echo ">> autoscaler running (pid $AS_PID), log: $RD/${RUN}_console.log"
  if [ "$AS" = slo ] || [ "$AS" = mmn ]; then
    echo "   reference run for the SLO decomposition: $REF_CSV" | tee -a "$RD/${RUN}_console.log"
  fi
fi

echo "START $(date '+%Y-%m-%d %H:%M:%S')" | tee "$RD/${RUN}_time.txt"
# CÁCH CŨ (trước 10/11/2026), không ghi độ trễ từng request:
#   RUN_TIME=$RUN_TIME locust -f pipeline/locustfile.py --headless --host http://localhost:31000 \
#       --csv "$RD/${RUN}_locust" > "$RD/${RUN}_locust.log" 2>&1 || true
echo "KEEPALIVE=${KEEPALIVE:-1}" | tee -a "$RD/${RUN}_time.txt"     # 0 = new TCP connection per request (11/11)
KEEPALIVE=${KEEPALIVE:-1} LAT_LOG="$RD/${RUN}_latency.csv" RUN_TIME=$RUN_TIME locust -f pipeline/locustfile.py --headless \
    --host http://localhost:31000 --csv "$RD/${RUN}_locust" > "$RD/${RUN}_locust.log" 2>&1 || true
echo "END   $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$RD/${RUN}_time.txt"
cleanup

START=$(sed -n 's/^START *//p' "$RD/${RUN}_time.txt"); END=$(sed -n 's/^END *//p' "$RD/${RUN}_time.txt")
python -m pipeline.collect --start "$START" --end "$END" --step 30 \
    --out "$RD/${RUN}.csv" --adj_out "$RD/${RUN}_od.npy"
python -m pipeline.client_latency --csv "$RD/${RUN}.csv" --lat_log "$RD/${RUN}_latency.csv"   # mới; cách cũ: không có bước này
echo ">> done: $RD/${RUN}.csv  ($(($(wc -l < "$RD/${RUN}.csv") - 1)) rows, expect ~151)"
