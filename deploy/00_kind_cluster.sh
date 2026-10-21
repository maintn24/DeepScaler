#!/usr/bin/env bash
# Option A (1 máy, để thử): cụm kind 1 control-plane + 2 worker.
# Cần: Docker, ~8 CPU / 16 GB RAM trống. Cụm thật nhiều máy: làm theo setUp-k8s.md (kubeadm) rồi sang 01_*.
set -euo pipefail
KIND_VERSION=${KIND_VERSION:-v0.24.0}
if ! command -v kind >/dev/null; then
  curl -Lo /tmp/kind "https://kind.sigs.k8s.io/dl/${KIND_VERSION}/kind-linux-amd64"
  chmod +x /tmp/kind && sudo mv /tmp/kind /usr/local/bin/kind
fi
if ! command -v kubectl >/dev/null; then
  curl -Lo /tmp/kubectl "https://dl.k8s.io/release/v1.30.4/bin/linux/amd64/kubectl"
  chmod +x /tmp/kubectl && sudo mv /tmp/kubectl /usr/local/bin/kubectl
fi
cat > /tmp/kind-deepscaler.yaml <<'YAML'
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  extraPortMappings:            # NodePort -> localhost
  - {containerPort: 30090, hostPort: 30090}   # Prometheus
  - {containerPort: 31000, hostPort: 31000}   # Boutique frontend
  - {containerPort: 30300, hostPort: 30300}   # Grafana
- role: worker
- role: worker
YAML
kind create cluster --name deepscaler --config /tmp/kind-deepscaler.yaml
kubectl get nodes -o wide
