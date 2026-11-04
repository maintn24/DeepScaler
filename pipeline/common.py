"""Shared constants of the DeepScaler pipeline, for the three benchmark applications of the paper
(Sec. V-A, Fig. 7): Online-Boutique (default), BookInfo and Train-Ticket.

Choose the application with the environment variable DS_APP = boutique | bookinfo | trainticket
(every script: collect.py, build_dataset.py, autoscaler.py, uncertainty_report.py, ...).

Per service we keep the Kubernetes Deployment name (= Istio destination_workload, = node name in
the CSV columns) and the name of its app container (cAdvisor `container` label).
Feature order follows the original repo (predict_scale.py / data_process.py):
  features = [pod, cpu, res, req, mem] -> feature 0 (pod) is the prediction target.
"""
import os

import numpy as np

FEATURES = ["pod", "cpu", "res", "req", "mem"]
TARGET_FEATURE = 0  # "pod": number of replicas needed at the next step

_BOUTIQUE = [
    "adservice", "cartservice", "checkoutservice", "currencyservice", "emailservice",
    "frontend", "paymentservice", "productcatalogservice", "recommendationservice",
    "shippingservice",
]
# Train-Ticket as deployed by the DeepScaler authors (benchmarks/train-ticket/ts-deployment-part2.yml,
# images codewisdom/*:0.0.4): the 41 business microservices of the paper (Sec. V-A, Fig. 7c).
# Databases (part1) and the nginx front end ts-ui-dashboard (part3) are not autoscaled / not nodes.
_TRAINTICKET = [
    "ts-admin-basic-info-service", "ts-admin-order-service", "ts-admin-route-service",
    "ts-admin-travel-service", "ts-admin-user-service", "ts-assurance-service", "ts-basic-service",
    "ts-cancel-service", "ts-config-service", "ts-consign-price-service", "ts-consign-service",
    "ts-contacts-service", "ts-execute-service", "ts-food-map-service", "ts-food-service",
    "ts-inside-payment-service", "ts-auth-service", "ts-news-service", "ts-notification-service",
    "ts-order-other-service", "ts-order-service", "ts-payment-service", "ts-preserve-other-service",
    "ts-preserve-service", "ts-price-service", "ts-rebook-service", "ts-route-plan-service",
    "ts-route-service", "ts-seat-service", "ts-security-service", "ts-user-service",
    "ts-station-service", "ts-ticket-office-service", "ts-ticketinfo-service", "ts-train-service",
    "ts-travel2-service", "ts-travel-plan-service", "ts-travel-service",
    "ts-verification-code-service", "ts-voucher-service", "ts-avatar-service",
]
assert len(_TRAINTICKET) == 41

APPS = {
    "boutique": {
        "namespace": "boutique",
        "services": _BOUTIQUE,
        "containers": {s: s for s in _BOUTIQUE},
        "entry": "frontend",                       # service whose latency = end-to-end latency
        "manifest": "benchmarks/boutique/boutique.yaml",
        # caller -> callee edges (gRPC clients in the Online-Boutique source code)
        "call_graph": {
            "frontend": ["adservice", "cartservice", "checkoutservice", "currencyservice",
                         "productcatalogservice", "recommendationservice", "shippingservice"],
            "checkoutservice": ["cartservice", "currencyservice", "emailservice", "paymentservice",
                                "productcatalogservice", "shippingservice"],
            "recommendationservice": ["productcatalogservice"],
        },
    },
    "bookinfo": {                                  # authors' benchmarks/bookinfo/bookinfo.yaml (Fig. 7a)
        "namespace": "bookinfossj",                # hard-coded in the authors' manifest
        "services": ["details", "productpage", "ratings", "reviews"],
        "containers": {s: s for s in ["details", "productpage", "ratings", "reviews"]},
        "entry": "productpage",                    # NodePort 30012
        "manifest": "benchmarks/bookinfo/bookinfo.yaml",
        "call_graph": {"productpage": ["details", "reviews"], "reviews": ["ratings"]},
    },
    "trainticket": {                               # authors' benchmarks/train-ticket/ (Fig. 7c)
        "namespace": "trainticket",
        "services": _TRAINTICKET,
        "containers": {s: s for s in _TRAINTICKET},
        # end-to-end latency is measured at the nginx front end (NodePort 32677), which is not one
        # of the 41 model nodes; collect.py adds its p95/res columns separately
        "entry": "ts-ui-dashboard",
        "manifest": "benchmarks/train-ticket/ts-deployment-part2.yml",
        "call_graph": None,                        # too large to hard-code: use the Istio graph (--adj_out)
    },
}

APP = os.environ.get("DS_APP", "boutique")
if APP not in APPS:
    raise ValueError(f"DS_APP={APP!r}: choose one of {list(APPS)}")
CFG = APPS[APP]
SERVICES = CFG["services"]
CONTAINERS = CFG["containers"]
ENTRY = CFG["entry"]
NAMESPACE = CFG["namespace"]
CALL_GRAPH = CFG["call_graph"]


def call_graph_adj(services=SERVICES, graph=CALL_GRAPH):
    """Directed 0/1 adjacency A[i, j] = 1 if service i calls service j."""
    if graph is None:
        raise ValueError(f"no static call graph for {APP}: pass the Istio graph (*_od.npy from collect.py)")
    idx = {s: i for i, s in enumerate(services)}
    a = np.zeros((len(services), len(services)), dtype=np.float32)
    for src, dsts in graph.items():
        for d in dsts:
            a[idx[src], idx[d]] = 1.0
    return a
