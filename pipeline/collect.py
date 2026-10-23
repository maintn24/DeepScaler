"""Collect DeepScaler training metrics from Prometheus (Kubernetes + Istio).

Replaces metrics_fetch.py for data collection: one `query_range` call per
(service, metric) instead of one HTTP call per time point, and the result is a
single CSV in the format consumed by pipeline/build_dataset.py.

Metrics (same PromQL as the original repo's metrics_fetch.py):
  pod : number of running containers of the service
  cpu : CPU usage / CPU limit (summed over pods)
  res : mean response time (s) seen by the Istio sidecar (reporter=destination)
  req : request rate (req/s) received by the service
  mem : memory usage / memory limit (summed over pods)
Extra columns for evaluation (not model inputs; build_dataset only reads --features):
  p95    : 95th-percentile response time (s) at the service's sidecar (reporter=destination);
           frontend_p95 = end-to-end latency seen by the load generator (it calls frontend)
  err    : rate of 5xx responses (req/s) returned by the service
  cpupod : CPU cores used by all containers of the service's pods, incl. the istio-proxy
           sidecar (what HPA compares against the pod CPU request)
It also writes the call graph observed by Istio (source -> destination
workload) as an adjacency matrix (`--adj_out`), used as the "od" graph.

Example (Prometheus exposed on NodePort 30090):
  python -m pipeline.collect --prom http://localhost:30090 --namespace boutique \
      --start "2026-10-21 14:00:00" --end "2026-10-21 20:00:00" --step 30 \
      --out data/raw/boutique_run1.csv --adj_out data/raw/boutique_run1_od.npy
  # or the last N minutes:  --last 360
"""
import argparse
import datetime as dt
import os
import time

import numpy as np
import pandas as pd
import requests

from pipeline.common import SERVICES

TEMPLATES = {
    "pod": "count(container_spec_cpu_period{{namespace='{ns}',container='{svc}'}})",
    "cpu": ("sum(irate(container_cpu_usage_seconds_total{{namespace='{ns}',container='{svc}'}}[1m]))"
            "/sum(container_spec_cpu_quota{{namespace='{ns}',container='{svc}'}}"
            "/container_spec_cpu_period{{namespace='{ns}',container='{svc}'}})"),
    "res": ("sum(rate(istio_request_duration_milliseconds_sum{{reporter='destination',"
            "destination_workload_namespace='{ns}',destination_workload='{svc}'}}[{win}]))"
            "/sum(rate(istio_request_duration_milliseconds_count{{reporter='destination',"
            "destination_workload_namespace='{ns}',destination_workload='{svc}'}}[{win}]))/1000"),
    "req": ("sum(rate(istio_requests_total{{reporter='destination',"
            "destination_workload_namespace='{ns}',destination_workload='{svc}'}}[{win}]))"),
    "mem": ("sum(container_memory_usage_bytes{{namespace='{ns}',container='{svc}'}})"
            "/sum(container_spec_memory_limit_bytes{{namespace='{ns}',container='{svc}'}})"),
}
# evaluation-only metrics (uncertainty / SLA analysis, pipeline/uncertainty_report.py)
EXTRA_TEMPLATES = {
    "p95": ("histogram_quantile(0.95, sum by (le) (rate(istio_request_duration_milliseconds_bucket{{"
            "reporter='destination',destination_workload_namespace='{ns}',destination_workload='{svc}'}}"
            "[{win}])))/1000"),
    "err": ("sum(rate(istio_requests_total{{reporter='destination',destination_workload_namespace='{ns}',"
            "destination_workload='{svc}',response_code=~'5..'}}[{win}]))"),
    "cpupod": ("sum(irate(container_cpu_usage_seconds_total{{namespace='{ns}',"
               "pod=~'{svc}-[a-z0-9]+-[a-z0-9]+',container!='',container!='POD'}}[1m]))"),
}
EDGE_QUERY = ("sum by (source_workload, destination_workload) (increase(istio_requests_total{{"
              "reporter='source',source_workload_namespace='{ns}',destination_workload_namespace='{ns}'}}[{dur}s]))")


def query_range(prom, q, start, end, step):
    """Prometheus caps a range query at 11k points -> chunk it."""
    out = {}
    chunk = step * 10000
    t = start
    while t < end:
        t2 = min(end, t + chunk)
        r = requests.get(f"{prom}/api/v1/query_range",
                         params={"query": q, "start": t, "end": t2, "step": step}, timeout=60)
        r.raise_for_status()
        res = r.json()["data"]["result"]
        if res:
            for ts, v in res[0]["values"]:
                out[float(ts)] = float(v) if v not in ("NaN", "+Inf", "-Inf") else np.nan
        t = t2 + step
    return out


def collect(prom, ns, start, end, step, services=SERVICES, metrics=None, verbose=True):
    """metrics: list of metric names to query (default: model inputs + evaluation extras)."""
    templates = {**TEMPLATES, **EXTRA_TEMPLATES}
    names = list(templates) if metrics is None else list(metrics)
    # integer count: with unix times (~1.7e9 s) `end + 1e-9 == end`, so np.arange(start, end + 1e-9)
    # silently dropped the last step
    grid = start + step * np.arange(int(round((end - start) / step)) + 1)
    cols = {"time": grid - grid[0], "unix_time": grid}
    win = f"{max(2 * step, 30)}s"
    for svc in services:
        for m in names:
            vals = query_range(prom, templates[m].format(ns=ns, svc=svc, win=win), start, end, step)
            cols[f"{svc}_{m}"] = [vals.get(float(t), np.nan) for t in grid]
        if verbose:
            print("collected", svc)
    df = pd.DataFrame(cols)
    # no traffic -> NaN latency/rate/errors; treat as 0, pods/cpu are forward-filled
    zero_fill = [m for m in ("res", "req", "p95", "err") if m in names]
    for svc in services:
        df[[f"{svc}_{m}" for m in zero_fill]] = df[[f"{svc}_{m}" for m in zero_fill]].fillna(0.0)
    return df.ffill().fillna(0.0)


def call_graph(prom, ns, end, duration, services=SERVICES):
    r = requests.get(f"{prom}/api/v1/query",
                     params={"query": EDGE_QUERY.format(ns=ns, dur=int(duration)), "time": end}, timeout=60)
    r.raise_for_status()
    idx = {s: i for i, s in enumerate(services)}
    a = np.zeros((len(services), len(services)), dtype=np.float32)
    for item in r.json()["data"]["result"]:
        s, d = item["metric"].get("source_workload"), item["metric"].get("destination_workload")
        if s in idx and d in idx and s != d and float(item["value"][1]) > 0:
            a[idx[s], idx[d]] = 1.0
    return a


def parse_time(s):
    return time.mktime(dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prom", default="http://localhost:30090")
    p.add_argument("--namespace", default="boutique")
    p.add_argument("--start", help="'YYYY-mm-dd HH:MM:SS' (local time of this machine)")
    p.add_argument("--end")
    p.add_argument("--last", type=float, help="collect the last N minutes instead of --start/--end")
    p.add_argument("--step", type=int, default=30, help="sampling interval in seconds")
    p.add_argument("--out", default="data/raw/boutique.csv")
    p.add_argument("--adj_out", default=None)
    a = p.parse_args()
    if a.last:
        end = time.time()
        start = end - a.last * 60
    else:
        start, end = parse_time(a.start), parse_time(a.end)
    start, end = start - start % a.step, end - end % a.step
    df = collect(a.prom, a.namespace, start, end, a.step)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(df)} rows")
    if a.adj_out:
        adj = call_graph(a.prom, a.namespace, end, end - start)
        np.save(a.adj_out, adj)
        print(f"wrote {a.adj_out}: {int(adj.sum())} call edges")


if __name__ == "__main__":
    main()
