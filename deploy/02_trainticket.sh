#!/usr/bin/env bash
# Train-Ticket as deployed by the DeepScaler authors (paper Sec. V-A, Fig. 7c: 41 microservices):
# benchmarks/train-ticket/ts-deployment-part1.yml (22 MongoDB/MySQL), part2.yml (41 ts-* services,
# codewisdom/*:0.0.4, requests 50m/160Mi, limits 200m/500Mi) and part3.yml (nginx front end
# ts-ui-dashboard, NodePort 32677), unchanged, in namespace "trainticket" with Istio injection.
# trainticket-gateway.yaml (Istio ingress) is not needed: Locust talks to the front end NodePort.
# HPA (CPU 50 %, 1..MAXR) on the 41 services is the teacher autoscaler while collecting data.
# MAXR defaults to 3 (memory: 41 JVMs x 500Mi limit); state it in the thesis.
# Machine: >= 16 vCPU / 64 GB RAM (e.g. m6i.4xlarge). Cluster from 00_kind_cluster.sh (port 32677).
set -euo pipefail
NS=${NS:-trainticket}; MAXR=${MAXR:-3}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; B="$ROOT/benchmarks/train-ticket"
kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl apply -f -
kubectl label namespace "$NS" istio-injection=enabled --overwrite
kubectl -n "$NS" apply -f "$B/ts-deployment-part1.yml"
echo "waiting for the databases ..."
kubectl -n "$NS" wait --for=condition=available deploy --all --timeout=1200s
kubectl -n "$NS" apply -f "$B/ts-deployment-part2.yml" -f "$B/ts-deployment-part3.yml"
# Java services must not start before their sidecar is up (otherwise DB connections fail at boot)
for d in $(kubectl -n "$NS" get deploy -o name | grep '/ts-'); do
  kubectl -n "$NS" patch "$d" --type=merge -p \
    '{"spec":{"template":{"metadata":{"annotations":{"proxy.istio.io/config":"{\"holdApplicationUntilProxyStarts\": true}"}}}}}' >/dev/null
done
echo "waiting for the 41 services (Java start-up takes 5-15 min) ..."
if ! kubectl -n "$NS" wait --for=condition=available deploy --all --timeout=2400s; then
  kubectl -n "$NS" get pods | awk 'NR==1 || $2 !~ /^2\/2$/ || $3 != "Running"'
  echo "!! Not all pods are ready (ImagePullBackOff = codewisdom/*:0.0.4 image missing)." >&2
  exit 1
fi
for d in $(kubectl -n "$NS" get deploy -o name | grep '/ts-' | grep -v ts-ui-dashboard | cut -d/ -f2); do
  kubectl -n "$NS" autoscale deployment "$d" --cpu-percent=50 --min=1 --max="$MAXR" \
    --dry-run=client -o yaml | kubectl -n "$NS" apply -f -
done
kubectl -n "$NS" get hpa | wc -l                   # 41 HPAs + header
curl -s -o /dev/null -w "login HTTP %{http_code}\n" -X POST -H 'Content-Type: application/json' \
  -d '{"username":"fdse_microservice","password":"111111"}' http://localhost:32677/api/v1/users/login
echo "Next: export DS_APP=trainticket (guide/EXPERIMENTS.md, scenario A)"
