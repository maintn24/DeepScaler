"""Table II / Fig. 10 of the DeepScaler paper from several online runs (pipeline/collect.py CSVs).

  python -m pipeline.compare_runs --slo 0.2 --warmup_min 10 --out reports/table2 \
      DeepScaler=runs/v2_exact/online_ds_<stamp>/online_ds_<stamp>.csv \
      HPA=runs/v2_exact/online_hpa_<stamp>/online_hpa_<stamp>.csv DeepScaler=<another ds run>.csv ...
  (old runs: runs/v1_istio-p95/<RUN>/<RUN>.csv; their latency is interpolated -> lat_source istio~)
(repeat a label for repeated runs -> mean +- std, as in Table II)

Metrics (paper Sec. III / V-A):
  violation rate R = T_violation / T (eq. 1): share of time the end-to-end latency (p95 at the entry
                     service, pipeline/common.py ENTRY) exceeds the SLO.
  cost (core-hours) (eq. 2): sum over time and services of replicas x CPU cores configured per
                     instance (CPU limit of the app container in the manifest).
  CAE (seconds)    : cumulative absolute error of the end-to-end latency w.r.t. the SLO,
                     sum_t |RT_t - SLO|. The paper reports CAE in seconds without a formula; this is
                     our reading, stated in the thesis. Caveat: it also grows when latency is far
                     BELOW the SLO, so a faster system can get a larger CAE.
  CAE_over (s)     : one-sided version, sum_t max(0, RT_t - SLO): only the excess over the SLO
                     (how badly the SLA is violated), reported next to CAE.
Also: violation rate for a range of SLOs (the paper gives no SLO value) and Fig. 10-style
cumulative curves. The first --warmup_min minutes of every run are skipped (model window filling).
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from pipeline.common import CFG, ENTRY, SERVICES
from pipeline.uncertainty_report import cpu_resources


def latency(df, source, slo):
    """End-to-end latency series used for eq. (1) of the paper (SLA on end-to-end response time).
    client: exact p95 seen by the load generator (pipeline/client_latency.py);
    istio : p95 at the entry service's sidecar. It is exact for `p95 > SLO` only if the Istio histogram
            has a bucket boundary at the SLO (column <entry>_slow<ms> present, deploy/02_boutique.sh);
            otherwise it is interpolated inside the bucket around the SLO (default buckets 100, 250 ms).
    auto  : client if available, else istio.

    OLD METHOD (before 10/11/2026), kept for comparison / rollback. Same numbers as `--lat_source istio`
    on the old CSVs. To restore the old code exactly, use instead of latency()+run_metrics(..., lat):
        def run_metrics(df, cores, slo, step):
            lat = df[f"{ENTRY}_p95"] if f"{ENTRY}_p95" in df else df[f"{ENTRY}_res"]
            ...                                   # rest of run_metrics unchanged
    and in main():
        m, v, c, e = run_metrics(df, cores, a.slo, step)
        rows.append({"label": label, "run": os.path.basename(path), **m})
        ...
        lat = df[f"{ENTRY}_p95"] if f"{ENTRY}_p95" in df else df[f"{ENTRY}_res"]   # for the SLO sweep
        print(f"Table II style (SLO {a.slo}s on {ENTRY} p95, warm-up {a.warmup_min} min skipped)\n", table.to_string())
    """
    if source in ("client", "auto") and "client_p95" in df and df["client_p95"].notna().mean() > 0.9:
        return df["client_p95"].ffill(), "client" + ("~" if df.get("client_src", pd.Series(["log"])).iloc[0] == "history" else "")
    if source == "client":
        raise SystemExit("client latency missing: run pipeline.client_latency on this CSV first")
    lat = df[f"{ENTRY}_p95"] if f"{ENTRY}_p95" in df else df[f"{ENTRY}_res"]
    exact = f"{ENTRY}_slow{int(round(1000 * slo))}" in df and df[f"{ENTRY}_slow{int(round(1000 * slo))}"].notna().any()
    return lat, "istio" + ("" if exact else "~")


def run_metrics(df, cores, slo, step, lat):
    pods = df[[f"{s}_pod" for s in SERVICES]].to_numpy()
    cost_t = (pods * cores).sum(1) * step / 3600                       # core-hours per step
    viol_t = (lat > slo).to_numpy()
    cae_t = (lat - slo).abs().to_numpy()
    over_t = (lat - slo).clip(lower=0).to_numpy()
    return {"violation_rate_%": 100 * viol_t.mean(), "cost_core_h": cost_t.sum(), "CAE_s": cae_t.sum(),
            "CAE_over_s": over_t.sum(),
            "pod_hours": pods.sum() * step / 3600, "p95_median_ms": 1000 * lat.median(),
            "minutes": len(df) * step / 60}, viol_t, cost_t, cae_t


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="LABEL=path.csv (repeat a label for repeated runs)")
    p.add_argument("--slo", type=float, default=0.2)
    p.add_argument("--slo_sweep", default="0.1,0.15,0.2,0.25,0.3")
    p.add_argument("--warmup_min", type=float, default=10)
    p.add_argument("--manifest", default=CFG["manifest"])
    p.add_argument("--out", default="reports/table2")
    p.add_argument("--lat_source", default="auto", choices=["auto", "client", "istio"],
                   help="end-to-end latency for eq. (1): client (Locust, exact), istio (sidecar p95), auto")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = cpu_resources(a.manifest)
    cores = np.array([res[s][1] for s in SERVICES])
    rows, curves, sweep = [], defaultdict(list), []
    for item in a.runs:
        label, path = item.split("=", 1)
        df = pd.read_csv(path)
        step = float(np.median(np.diff(df["unix_time"])))
        df = df.iloc[int(a.warmup_min * 60 / step):].reset_index(drop=True)
        lat, src = latency(df, a.lat_source, a.slo)
        m, v, c, e = run_metrics(df, cores, a.slo, step, lat)
        rows.append({"label": label, "run": os.path.basename(path), "lat_source": src, **m})
        curves[label].append((np.arange(len(v)) * step / 60, np.cumsum(v) * step / 60, np.cumsum(c), np.cumsum(e)))
        sweep.append({"label": label, **{f"SLO {int(1000 * float(t))} ms": 100 * (lat > float(t)).mean()
                                         for t in a.slo_sweep.split(",")}})
    per_run = pd.DataFrame(rows)
    per_run.to_csv(f"{a.out}/per_run.csv", index=False)
    g = per_run.groupby("label")
    keys = ["violation_rate_%", "cost_core_h", "CAE_s", "CAE_over_s"]
    table = pd.DataFrame({k: g[k].mean().map("{:.3f}".format) + " ± " + g[k].std(ddof=0).fillna(0).map("{:.3f}".format)
                          for k in keys})
    table["runs"] = g.size()
    table.to_csv(f"{a.out}/table2.csv")
    sw = pd.DataFrame(sweep).groupby("label").mean().round(1)
    sw.to_csv(f"{a.out}/slo_sweep.csv")
    pd.set_option("display.width", 200)
    srcs = sorted(set(per_run["lat_source"]))
    print(f"Table II style (SLO {a.slo}s, end-to-end p95 from {srcs}, warm-up {a.warmup_min} min skipped)\n",
          table.to_string())
    if any(x.endswith("~") for x in srcs):
        print("WARNING: '~' = approximate latency (istio~: p95 interpolated inside a histogram bucket that "
              "contains the SLO; client~: Locust stats_history). Do not mix sources inside one table.")
    print("\nViolation rate (%) vs SLO\n", sw.to_string())
    print("\nPer run\n", per_run.round(3).to_string(index=False))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        colors = dict(zip(curves, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
        for label, cs in curves.items():
            for k, (t, v, c, e) in enumerate(cs):
                kw = {"color": colors[label], "alpha": 0.8, "label": label if k == 0 else None}
                ax[0].plot(t, v, **kw); ax[1].plot(t, c, **kw); ax[2].plot(t, e, **kw)
        for x, tt in zip(ax, ["cumulative SLA violation (min)", "cumulative cost (core-hours)", "cumulative absolute error (s)"]):
            x.set_title(tt, fontsize=10); x.set_xlabel("minutes"); x.grid(alpha=.3)
        ax[0].legend()
        fig.suptitle(f"Fig. 10 style, SLO {int(a.slo * 1000)} ms")
        fig.tight_layout(); fig.savefig(f"{a.out}/fig10_cumulative.png", dpi=120)
        print("figure:", f"{a.out}/fig10_cumulative.png")
    except ImportError:
        print("matplotlib missing -> no figure")


if __name__ == "__main__":
    main()
