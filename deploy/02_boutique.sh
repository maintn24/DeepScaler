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
