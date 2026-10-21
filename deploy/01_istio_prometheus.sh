#!/usr/bin/env bash
# Istio (thu metric request/latency qua sidecar) + Prometheus (NodePort 30090) + Grafana (30300)
# + metrics-server (cho HPA lúc thu dữ liệu).
set -euo pipefail
ISTIO_VERSION=${ISTIO_VERSION:-1.22.3}
cd "$(dirname "$0")"
if ! command -v istioctl >/dev/null; then
  curl -sL https://istio.io/downloadIstio | ISTIO_VERSION=$ISTIO_VERSION sh -
  export PATH="$PWD/istio-$ISTIO_VERSION/bin:$PATH"
fi
istioctl install --set profile=default -y
ADDONS="https://raw.githubusercontent.com/istio/istio/release-${ISTIO_VERSION%.*}/samples/addons"
kubectl apply -f "$ADDONS/prometheus.yaml"
kubectl apply -f "$ADDONS/grafana.yaml"
kubectl -n istio-system rollout status deploy/prometheus --timeout=300s
# Prometheus của Istio đã scrape cAdvisor (container_cpu_*) qua kubernetes-nodes-cadvisor job.
kubectl -n istio-system patch svc prometheus -p '{"spec":{"type":"NodePort","ports":[{"name":"http","port":9090,"targetPort":9090,"nodePort":30090}]}}'
kubectl -n istio-system patch svc grafana -p '{"spec":{"type":"NodePort","ports":[{"name":"service","port":3000,"targetPort":3000,"nodePort":30300}]}}'
# metrics-server (kind cần --kubelet-insecure-tls)
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.7.2/components.yaml
kubectl -n kube-system patch deploy metrics-server --type=json \
  -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"},
        {"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-preferred-address-types=InternalIP"}]'
kubectl -n kube-system rollout status deploy/metrics-server --timeout=180s
for i in $(seq 1 12); do kubectl top nodes >/dev/null 2>&1 && break; sleep 10; done
kubectl top nodes || { echo "metrics-server chưa chạy -> HPA sẽ không scale"; exit 1; }
echo "Prometheus: http://localhost:30090  (thử: curl 'http://localhost:30090/api/v1/query?query=up')"
