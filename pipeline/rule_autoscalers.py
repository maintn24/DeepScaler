"""Baseline autoscalers of Table II that are not in the authors' repository (paper Sec. V-B).

The paper only names them; these are our re-implementations, written from the cited papers and
documented here so the thesis can state exactly what was compared:

  slo  [48] Gergin et al., "A decentralized autonomic architecture for performance control in the
       cloud" (rule-based on end-to-end response time, SLO decomposition). The end-to-end SLO is
       split into per-service local SLOs in proportion to each service's share of the latency
       measured in a reference run (--ref_csv). Every period: if a service's p95 > its local SLO
       -> +1 pod; if p95 < `down` x local SLO for `down_steps` consecutive periods -> -1 pod.
  mmn  [33] Jiang, Pierre, Chi, "Autonomous resource provisioning for multi-service web
       applications" (WWW'10): each service is an M/M/n/PS queue modelled independently
       (non-holistic). Service rate of one pod = mu_i / target_util (mu_i from profiling,
       reports/mu.json, is the req/s one pod serves at the CPU target), mean response time
       R_i(n) = (1/s_i) / (1 - lambda_i / (n s_i)); the smallest n with R_i(n) <= local SLO_i and
       lambda_i / (n s_i) <= rho_max is applied every period.
  (AWS [47], CPU target tracking, = Kubernetes HPA with CPU 50 %: deploy/02_*.sh.)

Both log one JSON line per decision (same format as pipeline/autoscaler.py, without "pred").

  python -m pipeline.rule_autoscalers --policy slo --slo 0.2 --ref_csv data/raw/run2_hpa_5f.csv
  python -m pipeline.rule_autoscalers --policy mmn --slo 0.2 --mu reports/mu.json --ref_csv data/raw/run2_hpa_5f.csv
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd

from pipeline.collect import collect
from pipeline.common import ENTRY, NAMESPACE, SERVICES


def local_slos(ref_csv, slo):
    """SLO decomposition: share of each service in the end-to-end latency of a reference run."""
    d = pd.read_csv(ref_csv)
    med = np.array([d.loc[d[f"{s}_res"] > 0, f"{s}_res"].median() for s in SERVICES])
    med = np.nan_to_num(med, nan=0.0)
    entry = d.loc[d[f"{ENTRY}_res"] > 0, f"{ENTRY}_res"].median()  # entry latency includes downstream
    share = med / max(float(np.nan_to_num(entry)), 1e-9)
    return {s: float(min(max(share[i], 0.05), 1.0) * slo) for i, s in enumerate(SERVICES)}


def mmn_pods(lam, s_rate, target, rho_max, lo, hi):
    for n in range(lo, hi + 1):
        rho = lam / (n * s_rate)
        if rho < 1 and rho <= rho_max and (1.0 / s_rate) / (1 - rho) <= target:
            return n
    return hi


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", choices=["slo", "mmn"], required=True)
    p.add_argument("--slo", type=float, default=0.2, help="end-to-end SLO on p95 latency (s)")
    p.add_argument("--ref_csv", required=True, help="reference run for the SLO decomposition")
    p.add_argument("--mu", default="reports/mu.json", help="mmn: profiled req/s per pod at the CPU target")
    p.add_argument("--target_util", type=float, default=0.5, help="mmn: CPU target used when mu was profiled")
    p.add_argument("--rho_max", type=float, default=0.8, help="mmn: max utilisation of the queue")
    p.add_argument("--down", type=float, default=0.5, help="slo: scale in below down x local SLO ...")
    p.add_argument("--down_steps", type=int, default=4, help="slo: ... for this many periods")
    p.add_argument("--prom", default="http://localhost:30090")
    p.add_argument("--namespace", default=NAMESPACE)
    p.add_argument("--period", type=int, default=30)
    p.add_argument("--min_replicas", type=int, default=1)
    p.add_argument("--max_replicas", type=int, default=10)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--log", default="Logs/rule_autoscaler.jsonl")
    a = p.parse_args()

    loc = local_slos(a.ref_csv, a.slo)
    if a.policy == "mmn":
        mu = json.load(open(a.mu))
        s_rate = {s: (mu[s]["mu"] if isinstance(mu[s], dict) else float(mu[s])) / a.target_util for s in SERVICES}
    calm = {s: 0 for s in SERVICES}
    os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)
    print(f"{a.policy} autoscaler: SLO {a.slo}s, local SLOs", {k: round(v, 3) for k, v in loc.items()}, flush=True)
    while True:
        t0 = time.time()
        end = t0 - t0 % a.period
        df = collect(a.prom, a.namespace, end - a.period, end, a.period, metrics=["pod", "req", "p95"], verbose=False)
        last = df.iloc[-1]
        cur = np.array([max(1, int(round(last[f"{s}_pod"]))) for s in SERVICES])
        new = cur.copy()
        for i, s in enumerate(SERVICES):
            if a.policy == "slo":
                if last[f"{s}_p95"] > loc[s]:
                    new[i], calm[s] = cur[i] + 1, 0
                elif last[f"{s}_p95"] < a.down * loc[s]:
                    calm[s] += 1
                    if calm[s] >= a.down_steps:
                        new[i], calm[s] = cur[i] - 1, 0
                else:
                    calm[s] = 0
            else:
                new[i] = mmn_pods(last[f"{s}_req"], s_rate[s], loc[s], a.rho_max, a.min_replicas, a.max_replicas)
        new = np.clip(new, a.min_replicas, a.max_replicas).astype(int)
        for s, n_new, n_old in zip(SERVICES, new, cur):
            if n_new != n_old and not a.dry_run:
                subprocess.run(["kubectl", "scale", "deployment", s, "-n", a.namespace, f"--replicas={n_new}"],
                               check=False, capture_output=True)
        with open(a.log, "a") as f:
            f.write(json.dumps({"time": end, "pods": new.tolist(), "current": cur.tolist(), "policy": a.policy}) + "\n")
        print(time.strftime("%H:%M:%S"), dict(zip([s.replace("service", "") for s in SERVICES], new.tolist())), flush=True)
        time.sleep(max(0.0, a.period - (time.time() - t0)))


if __name__ == "__main__":
    main()
