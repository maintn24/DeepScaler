#!/usr/bin/env bash
# Đo năng lực pod μ theo SLA, mỗi lần một service (thay μ theo CPU 50 % của bước 3b, guide/RUN_ONLINE.md).
#
# Mỗi service đích: đích 1 pod, các service có tải khác dư pod, service gần như không tải 1 pod; tăng tải
# theo bậc; thu Prometheus + log độ trễ từng request của Locust. Khi chỉ đích bị giới hạn, vi phạm SLA
# end-to-end là do đích -> μ_i = tải/pod lớn nhất mà p95 <= SLO (pipeline/profile_sla.py, theo cách định
# cỡ replica của BACC v2 Sec. III-C3a, bản thảo chưa công bố).
#
#   deploy/07_profile_sla.sh            # ~15 phút mỗi service, 6 service ≈ 1,5-2 giờ
#
# Biến môi trường (mặc định):
#   TARGETS="frontend recommendationservice checkoutservice cartservice currencyservice productcatalogservice"
#   SLO=0.2  LO=10 HI=150 STEP_USERS=10 HOLD_MIN=1   FRONT=6 OTHER=3 (số pod khi KHÔNG phải đích)
#   IDLE="adservice emailservice paymentservice shippingservice"   (luôn 1 pod)
# Trước đó: cluster dựng bằng deploy/02_boutique.sh mới (có LAT_BUCKETS).
# Kết quả: runs/<CAMPAIGN=v2_exact>/profile_sla_<svc>_<stamp>/ và reports/mu_sla.json
set -euo pipefail
NS=boutique; SLO=${SLO:-0.2}
TARGETS=${TARGETS:-"frontend recommendationservice checkoutservice cartservice currencyservice productcatalogservice"}
IDLE=${IDLE:-"adservice emailservice paymentservice shippingservice"}
LO=${LO:-10}; HI=${HI:-150}; STEP_USERS=${STEP_USERS:-10}; HOLD_MIN=${HOLD_MIN:-1}
FRONT=${FRONT:-6}; OTHER=${OTHER:-3}; TICK=5
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
[ -f ~/dsenv/bin/activate ] && source ~/dsenv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p runs sendFlow; CAMPAIGN=${CAMPAIGN:-v2_exact}
SLO_MS=$(python3 -c "print(int(round(float('$SLO')*1000)))")
STAMP=$(date +%Y%m%d_%H%M)
TRACE=sendFlow/ramp-sla-${LO}-${HI}.req
python -m pipeline.make_traces ramp --lo "$LO" --hi "$HI" --step "$STEP_USERS" --hold_min "$HOLD_MIN" \
    --tick "$TICK" --out "$TRACE"
RUN_TIME=$(( $(grep -c . "$TRACE") * TICK ))
kubectl -n "$NS" delete hpa --all --ignore-not-found

replicas_for() {   # $1 = deployment, $2 = target
  if [ "$1" = "$2" ]; then echo 1
  elif [[ " $IDLE " == *" $1 "* ]]; then echo 1
  elif [ "$1" = frontend ]; then echo "$FRONT"
  else echo "$OTHER"; fi
}

RUNS=()
for T in $TARGETS; do
  echo ">> target $T"
  for d in $(kubectl -n "$NS" get deploy -o jsonpath='{.items[*].metadata.name}'); do
    kubectl -n "$NS" scale deploy "$d" --replicas="$(replicas_for "$d" "$T")" >/dev/null
  done
  kubectl -n "$NS" rollout status deploy --timeout=600s >/dev/null
  for i in $(seq 1 20); do curl -s -o /dev/null http://localhost:31000/ || true; done
  sleep 60
  deploy/check_latency_buckets.sh "$SLO" || { echo "fix the histogram buckets first" >&2; exit 1; }
  RUN=profile_sla_${T}_${STAMP}
  RD="runs/${CAMPAIGN}/${RUN}"; mkdir -p "$RD"
  START=$(date '+%Y-%m-%d %H:%M:%S')
  LAT_LOG="$RD/${RUN}_latency.csv" TRACE="$TRACE" TICK=$TICK RUN_TIME=$RUN_TIME \
    locust -f pipeline/locustfile.py --headless --host http://localhost:31000 \
    --csv "$RD/${RUN}_locust" > "$RD/${RUN}_locust.log" 2>&1 || true
  END=$(date '+%Y-%m-%d %H:%M:%S')
  echo "START $START" > "$RD/${RUN}_time.txt"; echo "END   $END" >> "$RD/${RUN}_time.txt"
  python -m pipeline.collect --start "$START" --end "$END" --step 30 --slow_ms "$SLO_MS" --out "$RD/${RUN}.csv"
  python -m pipeline.client_latency --csv "$RD/${RUN}.csv" --lat_log "$RD/${RUN}_latency.csv" --slow_ms "$SLO_MS"
  RUNS+=("$RD/${RUN}.csv")
done
python -m pipeline.profile_sla --runs "${RUNS[@]}" --slo_ms "$SLO_MS" --base reports/mu.json --out reports/mu_sla.json
echo ">> reports/mu_sla.json  (dùng: --mu reports/mu_sla.json / MU=reports/mu_sla.json)"
