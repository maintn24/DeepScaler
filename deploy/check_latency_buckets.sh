#!/usr/bin/env bash
# Kiểm tra histogram độ trễ của Istio có biên bucket đúng tại SLO (deploy/02_boutique.sh, LAT_BUCKETS).
#   deploy/check_latency_buckets.sh [SLO_seconds=0.2] [PROM=http://localhost:30090]
# Cần đã có traffic (series chỉ xuất hiện sau request đầu tiên). Exit 1 nếu thiếu biên.
set -euo pipefail
SLO=${1:-0.2}; PROM=${2:-${PROM:-http://localhost:30090}}
MS=$(python3 -c "print(int(round(float('$SLO')*1000)))")
Q="count(istio_request_duration_milliseconds_bucket{reporter=\"destination\",le=~\"${MS}|${MS}.0\"})"
N=$(curl -sG "$PROM/api/v1/query" --data-urlencode "query=$Q" | python3 -c \
  'import json,sys; r=json.load(sys.stdin)["data"]["result"]; print(int(float(r[0]["value"][1])) if r else 0)')
if [ "$N" -gt 0 ]; then
  echo "OK: $N histogram series have a bucket boundary at ${MS} ms"
else
  echo "MISSING: no bucket boundary at ${MS} ms (no traffic yet, or pods created before LAT_BUCKETS:" \
       "rerun deploy/02_boutique.sh or 'kubectl -n boutique rollout restart deploy')" >&2
  exit 1
fi
