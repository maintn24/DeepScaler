"""Add client-side (load generator) end-to-end latency columns to a run CSV from pipeline.collect.

DeepScaler defines SLA violations on the end-to-end response time (paper Sec. III-A, eq. (1)). The
load generator is the only place that sees the full end-to-end time of each request, exactly, so
this is the reference measurement; the Istio sidecar view (<entry>_p95, <entry>_slow<ms>) is kept
for online control, where the autoscaler reads Prometheus.

For every row of the run CSV (step t covers (unix_time - step, unix_time]):
  client_n          requests that finished in the step
  client_p95        exact 95th percentile of their latency (s)
  client_slow<ms>   share of them slower than <ms> (failed requests count as slow)
  client_cpu        Locust's own CPU % (mean over the step), if LAT_LOG.cpu.csv exists
Source 1 (exact): the per-request log of pipeline/latency_log.py (--lat_log).
Source 2 (fallback for runs made before that log existed): Locust's *_stats_history.csv
  (--history). Its percentiles are over Locust's short rolling window and rounded, so client_p95 is
  the median of the values reported inside the step: an approximation, marked client_src=history.

  python -m pipeline.client_latency --csv data/raw/RUN.csv --lat_log Logs/RUN_latency.csv
  python -m pipeline.client_latency --csv data/raw/RUN.csv --history Logs/RUN_locust_stats_history.csv
"""
import argparse
import os

import numpy as np
import pandas as pd


def from_log(df, lat_log, step, slow_ms):
    r = pd.read_csv(lat_log)
    t, rt = r["t_end"].to_numpy(), r["rt_ms"].to_numpy(dtype=float)
    ok = r["ok"].to_numpy().astype(bool)
    edges = df["unix_time"].to_numpy(dtype=float)
    idx = np.searchsorted(edges, t, side="left")           # request ending in (edges[i]-step, edges[i]] -> i
    valid = (idx < len(edges)) & (t > edges[np.minimum(idx, len(edges) - 1)] - step)
    out = {"client_n": np.zeros(len(df)), "client_p95": np.full(len(df), np.nan)}
    for ms in slow_ms:
        out[f"client_slow{ms}"] = np.full(len(df), np.nan)
    order = np.argsort(idx[valid], kind="stable")
    iv, rv, okv = idx[valid][order], rt[valid][order], ok[valid][order]
    bounds = np.searchsorted(iv, np.arange(len(df) + 1))
    for i in range(len(df)):
        a, b = bounds[i], bounds[i + 1]
        if b <= a:
            continue
        x, k = rv[a:b], okv[a:b]
        out["client_n"][i] = b - a
        out["client_p95"][i] = np.quantile(x, 0.95) / 1000
        for ms in slow_ms:
            out[f"client_slow{ms}"][i] = np.mean((x > ms) | ~k)
    for k, v in out.items():
        df[k] = v
    cpu_path = lat_log + ".cpu.csv"
    if os.path.exists(cpu_path):
        c = pd.read_csv(cpu_path)
        ci = np.searchsorted(edges, c["t"].to_numpy(), side="left")
        m = ci < len(edges)
        df["client_cpu"] = pd.Series(c["cpu_percent"].to_numpy()[m]).groupby(ci[m]).mean().reindex(range(len(df))).to_numpy()
    df["client_src"] = "log"
    return df


def from_history(df, hist, step, slow_ms):
    h = pd.read_csv(hist)
    h = h[h["Name"] == "Aggregated"].copy()
    h["p95"] = pd.to_numeric(h["95%"], errors="coerce")
    h = h.dropna(subset=["p95"])
    edges = df["unix_time"].to_numpy(dtype=float)
    idx = np.searchsorted(edges, h["Timestamp"].to_numpy(dtype=float), side="left")
    m = idx < len(edges)
    p95 = pd.Series(h["p95"].to_numpy()[m]).groupby(idx[m]).median().reindex(range(len(df)))
    df["client_p95"] = p95.to_numpy() / 1000
    df["client_src"] = "history"
    print("NOTE: client_p95 from Locust stats_history is approximate (rolling window, rounded); "
          "client_slow<ms> needs the per-request log (LAT_LOG).")
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="run CSV from pipeline.collect (updated in place unless --out)")
    p.add_argument("--lat_log", help="per-request log written by pipeline/latency_log.py")
    p.add_argument("--history", help="Locust *_stats_history.csv (fallback, approximate)")
    p.add_argument("--slow_ms", default="100,150,200,250,300")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    df = pd.read_csv(a.csv)
    step = float(np.median(np.diff(df["unix_time"])))
    slow_ms = [int(x) for x in a.slow_ms.split(",") if x.strip()]
    if a.lat_log and os.path.exists(a.lat_log):
        df = from_log(df, a.lat_log, step, slow_ms)
    elif a.history and os.path.exists(a.history):
        df = from_history(df, a.history, step, slow_ms)
    else:
        raise SystemExit("give --lat_log (exact) or --history (approximate); file not found")
    out = a.out or a.csv
    df.to_csv(out, index=False)
    cov = df["client_p95"].notna().mean()
    print(f"wrote {out}: client latency on {100 * cov:.0f}% of steps ({df['client_src'].iloc[0]})")
    if "client_cpu" in df and (df["client_cpu"] > 90).any():
        print(f"WARNING: Locust CPU > 90% on {(df['client_cpu'] > 90).sum()} steps: client latencies there "
              "include load-generator queueing (run Locust with more workers).")


if __name__ == "__main__":
    main()
