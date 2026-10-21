"""Shared constants for the Online-Boutique version of DeepScaler.

Node order and feature order follow the original repo (predict_scale.py):
  nodes    = 10 Online-Boutique services (alphabetical, as in metrics_fetch.py)
  features = [pod, cpu, res, req]  -> feature 0 (pod) is the prediction target
"""
import numpy as np

SERVICES = [
    "adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice",
    "frontend", "paymentservice", "productcatalogservice", "recommendationservice",
    "shippingservice",
]
FEATURES = ["pod", "cpu", "res", "req"]
TARGET_FEATURE = 0  # "pod": number of replicas needed at the next step

# Caller -> callee edges of Online Boutique (from the gRPC clients in its source code).
CALL_GRAPH = {
    "frontend": ["adservice", "cartservice", "checkoutservice", "currencyservice",
                 "productcatalogservice", "recommendationservice", "shippingservice"],
    "checkoutservice": ["cartservice", "currencyservice", "emailservice", "paymentservice",
                        "productcatalogservice", "shippingservice"],
    "recommendationservice": ["productcatalogservice"],
}


def call_graph_adj(services=SERVICES, graph=CALL_GRAPH):
    """Directed 0/1 adjacency A[i, j] = 1 if service i calls service j."""
    idx = {s: i for i, s in enumerate(services)}
    a = np.zeros((len(services), len(services)), dtype=np.float32)
    for src, dsts in graph.items():
        for d in dsts:
            a[idx[src], idx[d]] = 1.0
    return a
