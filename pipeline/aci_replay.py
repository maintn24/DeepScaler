"""Offline replay: original DeepScaler rounding vs the thesis' application-level ACI on the same forecast.

Replays a run CSV (pipeline.collect) step by step. At step t the forecast yhat_t for the demand
n_t = max(1, ceil(req_t / mu)) uses data up to t-1:
  --forecaster persistence : yhat_t = n_{t-1}                              (thesis H4, model-agnostic)
  --forecaster model       : trained AdapGLA (--model_dir, target 'demand' or 'pod'), needs torch
Policies on the same yhat (replicas assumed Ready for step t, no SLA feedback: the recorded latency
belongs to whichever controller ran during the run, so the pacer is off and alpha is fixed):
  deepscaler   original rule of pipeline/autoscaler.py / predict_scale.py: floor if frac < 0.35 else ceil
  aci          application-level ACI (pipeline/aci.py), immediate scale-in
  aci+sched    the same + scheduler (scale in after --down_steps lower proposals, -1 per step)
Metrics after the calibration prefix: miss_any = share of steps where some active service has fewer
replicas than its demand (the event ACI controls), per-service miss, mean replicas, replica changes.

  python -m pipeline.aci_replay --csv data/raw/train_burst_20261104_0751.csv --mu reports/mu.json --alpha 0.05
"""
import argparse
import json

import numpy as np
import pandas as pd

from pipeline.aci import AppACI, Scheduler, active_set, decide, demand, ratio_score
from pipeline.common import SERVICES


def deepscaler_rule(pred, restriction=0.35):
    """Copy of pipeline/autoscaler.scale_to_pods (kept there unchanged; copied to avoid importing torch)."""
    frac = pred - np.floor(pred)
    return np.where(frac < restriction, np.floor(pred), np.ceil(pred))


def load_mu(path, services):
    raw = json.load(open(path))
    return np.array([float(raw[s]["mu"] if isinstance(raw[s], dict) else raw[s]) for s in services])


def model_forecast(df, model_dir, services, mu):
    """yhat[t] = model forecast of step t (feature 0, horizon 0) from the step_in steps before t."""
    from pipeline.aci_autoscaler import load_model, predict_window   # torch only here
    model, adj, meta, mean, std = load_model(model_dir)
    feats, step_in = meta["features"], meta["step_in"]
    if "demand" in feats:
        mu_m = np.array([meta["mu"][s] for s in services])
        for i, s in enumerate(services):
            df[f"{s}_demand"] = demand(df[f"{s}_req"].fillna(0), mu_m[i])
    X = np.stack([np.stack([df[f"{s}_{f}"].ffill().fillna(0).to_numpy() for f in feats], -1) for s in services], 1)
    yhat = np.full((len(df), len(services)), np.nan)
    for t in range(step_in, len(df)):
        yhat[t] = predict_window(model, adj, X[t - step_in:t], mean, std)
    return yhat


def run(n, yhat, start, calib, alpha, gamma, m, c, rho, down_steps, r_min, r_max, policy):
    T, N = n.shape
    act = active_set(n[start:calib], rho)
    warm = [ratio_score(n[t, act], yhat[t, act], c).max() for t in range(start, calib)]
    aci = AppACI(act, m=m, gamma=gamma, c=c, beta0=alpha, warm_scores=warm)
    sched = Scheduler(N, r_min, r_max, down_steps=down_steps if policy == "aci+sched" else 0, max_down=r_max)
    y_prev = np.clip(np.ceil(n[calib - 1]), r_min, r_max)
    Y = []
    for t in range(calib, T):
        if policy == "deepscaler":
            y = np.clip(deepscaler_rule(yhat[t]), r_min, r_max)
        else:
            y = decide(yhat[t], aci, sched, y_prev, r_min, r_max)
            aci.observe(yhat[t], n[t], y, alpha)
        Y.append(y)
        y_prev = y
    Y = np.array(Y, float)
    nn = n[calib:]
    short = Y < nn
    return {"miss_any": short[:, act].any(1).mean(), "miss_per_service": {str(s): float(v) for s, v in zip(np.array(SERVICES)[act], short[:, act].mean(0).round(3))},
            "replicas": Y.sum(1).mean(), "demand": nn.sum(1).mean(),
            "changes": int((np.diff(Y, axis=0) != 0).sum()), "active": int(act.sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True)
    p.add_argument("--mu", default="reports/mu.json")
    p.add_argument("--forecaster", default="persistence", choices=["persistence", "model"])
    p.add_argument("--model_dir", default=None)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--gamma", type=float, default=0.01)
    p.add_argument("--m", type=int, default=120)
    p.add_argument("--c", type=float, default=1.0)
    p.add_argument("--rho", type=float, default=0.05, help="active-set threshold")
    p.add_argument("--calib", type=int, default=120, help="calibration steps (warm start, active set)")
    p.add_argument("--down_steps", type=int, default=3)
    p.add_argument("--min_replicas", type=int, default=1)
    p.add_argument("--max_replicas", type=int, default=50, help="large by default: offline, no cluster cap")
    a = p.parse_args()
    df = pd.read_csv(a.csv)
    mu = load_mu(a.mu, SERVICES)
    n = np.stack([demand(df[f"{s}_req"].fillna(0), mu[i]) for i, s in enumerate(SERVICES)], 1)
    if a.forecaster == "persistence":
        yhat = np.vstack([n[:1], n[:-1]]).astype(float)
        start = 1
    else:
        yhat = model_forecast(df, a.model_dir, SERVICES, mu)
        start = int(np.argmax(~np.isnan(yhat).any(1)))
    calib = start + a.calib
    rows = {}
    for pol in ["deepscaler", "aci", "aci+sched"]:
        rows[pol] = run(n, yhat, start, calib, a.alpha, a.gamma, a.m, a.c, a.rho, a.down_steps,
                        a.min_replicas, a.max_replicas, pol)
    out = pd.DataFrame({k: {kk: v for kk, v in r.items() if kk != "miss_per_service"} for k, r in rows.items()}).T
    print(f"{a.csv}  forecaster={a.forecaster}  alpha={a.alpha}  steps evaluated={len(n) - calib}")
    print(out.to_string())
    for k, r in rows.items():
        print(f"  {k:11s} per-service miss: {r['miss_per_service']}")


if __name__ == "__main__":
    main()
