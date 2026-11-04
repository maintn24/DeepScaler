"""Quality check of a training-data CSV (pipeline/collect.py) before training: does it contain
SLA violations, bursts and enough variation of the label?

  python -m pipeline.check_data data/raw/<run>.csv --slo 0.2 \
      --bursts sendFlow/train-burst-6h.req.bursts.csv --start "<START of the run>"

Reports
  * SLA: share of time the entry p95 > SLO (eq. 1 of the paper), overall / inside bursts (+2 min) /
    outside bursts, and the longest violation.
  * Label variation (paper Sec. III: label = pods at t+1, here chosen by HPA): per service min/max
    pods and how often the count changes; MAE of the persistence forecast "pods(t+1) = pods(t)".
    This is the floor any model must beat: if it is ~0 the label is almost constant and the data
    cannot show the differences of Fig. 12 (the old run2_hpa had 0.040 pod).
  * The same for the demand label max(1, ceil(req / mu)) when reports/mu.json exists.
  * Total CPU used (cores) vs the machine, to tell bursts from a saturated node.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from pipeline.common import ENTRY, SERVICES


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("--slo", type=float, default=0.2)
    p.add_argument("--bursts", help="<trace>.bursts.csv from make_traces train")
    p.add_argument("--start", help="START time of the Locust run ('YYYY-mm-dd HH:MM:SS'), to place bursts")
    p.add_argument("--mu", default="reports/mu.json")
    p.add_argument("--cores", type=float, default=8)
    a = p.parse_args()
    d = pd.read_csv(a.csv)
    step = float(np.median(np.diff(d["unix_time"])))
    lat = d[f"{ENTRY}_p95"] if f"{ENTRY}_p95" in d else d[f"{ENTRY}_res"]
    v = (lat > a.slo).to_numpy()
    print(f"{a.csv}: {len(d)} steps x {step:.0f}s = {len(d) * step / 3600:.2f} h")
    print(f"\n[SLA {int(a.slo * 1000)} ms on {ENTRY} p95]  violation {100 * v.mean():.1f}%  "
          f"p95 median/p99/max = {1000 * lat.median():.0f}/{1000 * lat.quantile(.99):.0f}/{1000 * lat.max():.0f} ms")
    runs = np.diff(np.flatnonzero(np.diff(np.r_[0, v.astype(int), 0])))[::2] if v.any() else [0]
    print(f"  violation episodes: {int(np.sum(np.diff(np.r_[0, v.astype(int)]) == 1))}, longest {max(runs) * step / 60:.1f} min")
    if a.bursts and a.start:
        b = pd.read_csv(a.bursts)
        t0 = time.mktime(time.strptime(a.start, "%Y-%m-%d %H:%M:%S"))
        m = (d["unix_time"] - t0).to_numpy() / 60
        inb = np.zeros(len(d), bool)
        for s, ln in zip(b.start_min, b.length_min):
            inb |= (m >= s) & (m < s + ln + 2)
        print(f"  inside bursts (+2 min): {100 * v[inb].mean():.1f}% of {inb.sum()} steps | "
              f"outside: {100 * v[~inb].mean():.1f}%")
        hit = [bool(v[(m >= s) & (m < s + ln + 2)].any()) for s, ln in zip(b.start_min, b.length_min)]
        print(f"  bursts that caused >= 1 violation: {sum(hit)}/{len(hit)}")

    pods = d[[f"{s}_pod" for s in SERVICES]].round().to_numpy()
    print("\n[HPA label = pods(t+1)]")
    print("  " + "  ".join(f"{s.replace('service', '')}:{int(pods[:, i].min())}-{int(pods[:, i].max())}"
                           for i, s in enumerate(SERVICES)))
    ch = (np.abs(np.diff(pods, axis=0)) > 0)
    print(f"  steps with any change {100 * ch.any(1).mean():.1f}%, per service "
          + " ".join(f"{100 * ch[:, i].mean():.0f}%" for i in range(len(SERVICES))))
    print(f"  persistence MAE {np.abs(np.diff(pods, axis=0)).mean():.3f} pod (old run2_hpa: 0.040)")
    if os.path.exists(a.mu):
        raw = json.load(open(a.mu))
        mu = np.array([raw[s]["mu"] if isinstance(raw[s], dict) else float(raw[s]) for s in SERVICES])
        dem = np.maximum(1, np.ceil(d[[f"{s}_req" for s in SERVICES]].to_numpy() / mu))
        print(f"[demand label max(1, ceil(req/mu))] persistence MAE {np.abs(np.diff(dem, axis=0)).mean():.3f}, "
              f"max per service " + " ".join(str(int(x)) for x in dem.max(0)))
    cpu = d[[f"{s}_cpu" for s in SERVICES]].sum(axis=1) if f"{SERVICES[0]}_cpu" in d else None
    if cpu is not None:
        print(f"\n[CPU of the app containers] max {cpu.max():.2f} cores, p95 {cpu.quantile(.95):.2f} "
              f"(machine {a.cores:g}; sidecars + Locust + Prometheus come on top)")


if __name__ == "__main__":
    main()
