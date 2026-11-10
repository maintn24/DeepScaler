#!/usr/bin/env bash
# Online Boutique trong namespace "boutique" có Istio sidecar, + HPA cho cả 10 service
# (HPA = bộ scale "giáo viên" khi thu dữ liệu train, giống quy trình của DeepScaler).
set -euo pipefail
NS=${NS:-boutique}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl -n "$NS" apply -f -
kubectl label namespace "$NS" istio-injection=enabled --overwrite
kubectl -n "$NS" apply -f "$ROOT/benchmarks/boutique/boutique.yaml"
kubectl -n "$NS" apply -f "$ROOT/benchmarks/boutique/boutique-gateway.yaml"
# Biên bucket histogram độ trễ của Istio. Mặc định Istio nhảy 100 -> 250 ms, nên p95 quanh SLO 200 ms
# bị nội suy trong bucket (100, 250] và "p95 > SLO" bị sai. Đặt biên đúng tại mọi ngưỡng SLO đang dùng
# (100, 150, 200, 250, 300 ms) thì "p95 > SLO" <=> "tỷ lệ request > SLO vượt 5 %" là chính xác.
# Annotation: sidecar.istio.io/statsHistogramBuckets (istio.io/docs/reference/config/annotations).
LAT_BUCKETS=${LAT_BUCKETS:-[1,5,10,25,50,75,100,125,150,175,200,225,250,300,400,500,750,1000,2500,5000,10000]}
[ "$LAT_BUCKETS" = default ] || PATCH=$(python3 -c 'import json,sys; print(json.dumps({"spec":{"template":{"metadata":{"annotations":
  {"sidecar.istio.io/statsHistogramBuckets": json.dumps({"istiocustom": json.loads(sys.argv[1])})}}}}}))' "$LAT_BUCKETS")
# CÁCH CŨ (trước 10/11/2026): không patch, dùng bucket mặc định của Istio
#   (0.5, 1, 5, 10, 25, 50, 100, 250, 500, 1000, ...). Quay lại: LAT_BUCKETS=default deploy/02_boutique.sh
if [ "$LAT_BUCKETS" != default ]; then
  for d in $(kubectl -n "$NS" get deploy -o name); do
    kubectl -n "$NS" patch "$d" --type merge -p "$PATCH" >/dev/null
  done
fi
kubectl -n "$NS" wait --for=condition=available deploy --all --timeout=600s
for d in adservice cartservice checkoutservice currencyservice emailservice frontend \
         paymentservice productcatalogservice recommendationservice shippingservice; do
  kubectl -n "$NS" autoscale deployment "$d" --cpu-percent=50 --min=1 --max=10 \
    --dry-run=client -o yaml | kubectl -n "$NS" apply -f -
done
# frontend được đặt sẵn replicas: 4 trong boutique.yaml -> đưa về 1 để HPA tự quyết
kubectl -n "$NS" scale deploy frontend --replicas=1
kubectl -n "$NS" get pods,hpa
echo "Frontend: http://localhost:31000"
