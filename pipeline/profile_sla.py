"""SLA-based pod capacity mu_i from one-service-at-a-time profiling runs (deploy/07_profile_sla.sh).

mu in reports/mu.json (pipeline/uncertainty_report.estimate_mu) is the req/s per pod at which the
pod's CPU reaches 50 % of its request, i.e. HPA's operating point, not what one pod can serve within
the SLA. On Online Boutique pods kept p95 < 200 ms at 2-3x that load, so the demand label
max(1, ceil(req/mu)) overstated the replicas needed for the SLA.

Here mu_i is the highest load per replica of service i at which the end-to-end p95 stays within the
SLO, following BACC's replica sizing: measure the violation probability as a function of load per
replica, then take the highest safe load (Liu et al., BACC v2, unpublished manuscript 2026, Sec. III-C3a, eq. (8)); its latency
profile uses the share of bad requests at each pooled load point. In each profiling run only the
target service is constrained (1 replica, others over-provisioned), so end-to-end violations are
attributable to it.

Per run: x = req_i / pods_i (req/s per replica); bad = share of requests slower than the SLO
(client_slow<ms>, exact; else <entry>_slow<ms>; else 1[p95 > SLO], approximate). Load is binned;
per bin K bad of U requests are pooled and the one-sided 95 % Wilson upper bound of K/U is compared
with q = 0.05 (p95 <= SLO <=> share of requests slower than the SLO <= 5 %). mu_i = upper edge of the
last bin of the longest safe prefix. If no bin is unsafe the run never reached the limit: mu_i is a
lower bound (censored=true). Services not profiled keep their CPU-based mu (method="cpu").

OLD METHOD (kept, unchanged): CPU-based mu = (0.5 x (app CPU request + 0.1 sidecar) - a) / b from the
linear fit cpu_pod = a + b x req_per_pod (pipeline/uncertainty_report.estimate_mu), file reports/mu.json:
  python -m pipeline.uncertainty_report --csv data/raw/profile.csv --mu_basis pod --mu_only --out_dir reports
Every script still reads reports/mu.json by default; use reports/mu_sla.json only where you pass it.

  python -m pipeline.profile_sla --runs data/raw/profile_sla_*_20261111_0900.csv --slo_ms 200 \
      --base reports/mu.json --out reports/mu_sla.json
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd

from pipeline.common import ENTRY, SERVICES

Z95 = 1.6448536269514722  # one-sided 95 %


def wilson_upper(k, n, z=Z95):
    if n <= 0:
        return 1.0
    p = k / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float(min(1.0, (c + r) / (1 + z * z / n)))


def bad_share(df, slo_ms, step):
    """Share of slow requests per step and number of requests per step, with the source used."""
    if f"client_slow{slo_ms}" in df and df[f"client_slow{slo_ms}"].notna().mean() > 0.9:
        return df[f"client_slow{slo_ms}"].to_numpy(), df["client_n"].to_numpy(), "client"
    n = (df[f"{ENTRY}_req"] * step).to_numpy()
    if f"{ENTRY}_slow{slo_ms}" in df and df[f"{ENTRY}_slow{slo_ms}"].notna().mean() > 0.9:
        return df[f"{ENTRY}_slow{slo_ms}"].to_numpy(), n, "istio"
    # last resort: a whole step counts as bad when its interpolated p95 exceeds the SLO
    return (df[f"{ENTRY}_p95"] > slo_ms / 1000).astype(float).to_numpy() * 1.0, n, "istio_p95~"


def profile_one(df, svc, slo_ms, q, bins, min_req, warmup_steps):
    df = df.iloc[warmup_steps:].reset_index(drop=True)
    step = float(np.median(np.diff(df["unix_time"])))
    pods = df[f"{svc}_pod"].clip(lower=1).to_numpy()
    x = df[f"{svc}_req"].to_numpy() / pods
    bad, n, src = bad_share(df, slo_ms, step)
    m = np.isfinite(x) & np.isfinite(bad) & (n > 0) & (x > 0)
    x, bad, n = x[m], np.clip(bad[m], 0, 1), n[m]
    edges = np.linspace(0, x.max(), bins + 1)
    rows, mu, censored = [], np.nan, True
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x > lo) & (x <= hi)
        U, K = float(n[sel].sum()), float((bad[sel] * n[sel]).sum())
        if U < min_req:
            rows.append({"lo": lo, "hi": hi, "requests": U, "bad_share": None, "ucb": None})
            continue
        ucb = wilson_upper(K, U)
        rows.append({"lo": float(lo), "hi": float(hi), "requests": U, "bad_share": K / U, "ucb": ucb})
        if ucb <= q and censored:
            mu = float(hi)
        elif ucb > q:
            censored = False
            break
    return {"mu": mu, "method": "sla", "slo_ms": slo_ms, "q": q, "censored": bool(censored),
            "x_max": float(x.max()), "latency_source": src, "bins": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True, help="data/raw/profile_sla_<service>_<stamp>.csv")
    p.add_argument("--slo_ms", type=int, default=200)
    p.add_argument("--q", type=float, default=0.05, help="max share of requests slower than the SLO (p95 -> 0.05)")
    p.add_argument("--bins", type=int, default=12)
    p.add_argument("--min_req", type=float, default=200, help="min requests in a load bin to judge it")
    p.add_argument("--warmup_steps", type=int, default=2)
    p.add_argument("--base", default="reports/mu.json", help="CPU-based mu kept for services not profiled")
    p.add_argument("--out", default="reports/mu_sla.json")
    a = p.parse_args()
    base = json.load(open(a.base)) if os.path.exists(a.base) else {}
    out = {s: {**(base[s] if isinstance(base.get(s), dict) else {"mu": base.get(s, np.nan)}), "method": "cpu"}
           for s in SERVICES}
    for path in a.runs:
        mt = re.search(r"profile_sla_(.+?)_\d{8}_\d{4}", os.path.basename(path))
        if not mt or mt.group(1) not in SERVICES:
            raise SystemExit(f"cannot read the target service from {path} (expect profile_sla_<svc>_<stamp>.csv)")
        svc = mt.group(1)
        r = profile_one(pd.read_csv(path), svc, a.slo_ms, a.q, a.bins, a.min_req, a.warmup_steps)
        r["run"] = os.path.basename(path)
        r["mu_cpu"] = out[svc].get("mu")
        out[svc] = r
        flag = "  (lower bound: never violated)" if r["censored"] else ""
        if np.isnan(r["mu"]):
            flag = "  (violates the SLO already at the lowest load: p0 problem, check the measurement)"
        print(f"{svc:24s} mu_sla={r['mu']:.1f} req/s/pod  mu_cpu={r['mu_cpu']:.1f}  [{r['latency_source']}]{flag}")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
