#!/usr/bin/env bash
# BookInfo as deployed by the DeepScaler authors (paper Sec. V-A, Fig. 7a: 4 microservices):
# benchmarks/bookinfo/bookinfo.yaml, unchanged. It hard-codes namespace "bookinfossj", deployments
# productpage / reviews / ratings / details (+ mongodb, not autoscaled), images tongjingwan/*,
# productpage NodePort 30012. We only add the Istio sidecar (namespace injection; the authors'
# manifest has the istio-proxy container commented out, i.e. it relied on injection) and HPA
# (CPU 50 %, 1..MAXR pods) as the teacher autoscaler while collecting training data, as for Boutique.
# Needs the kind cluster from deploy/00_kind_cluster.sh (port 30012 mapped) and 01_istio_prometheus.sh.
set -euo pipefail
NS=bookinfossj; MAXR=${MAXR:-10}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl apply -f -
kubectl label namespace "$NS" istio-injection=enabled --overwrite
kubectl apply -f "$ROOT/benchmarks/bookinfo/bookinfo.yaml"
if ! kubectl -n "$NS" wait --for=condition=available deploy --all --timeout=900s; then
  kubectl -n "$NS" get pods
  echo "!! Some pods are not ready. ImagePullBackOff = the authors' tongjingwan/* image is no longer" \
       "on Docker Hub: report it (kubectl -n $NS describe pod <pod>) before going on." >&2
  exit 1
fi
for d in productpage reviews ratings details; do
  kubectl -n "$NS" autoscale deployment "$d" --cpu-percent=50 --min=1 --max="$MAXR" \
    --dry-run=client -o yaml | kubectl -n "$NS" apply -f -
done
kubectl -n "$NS" get pods,hpa                      # app pods should be 2/2 (app + istio-proxy)
curl -s -o /dev/null -w "productpage HTTP %{http_code}\n" http://localhost:30012/productpage
echo "Next: export DS_APP=bookinfo (guide/EXPERIMENTS.md, scenario A)"
