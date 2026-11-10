"""Thesis autoscaler: DeepScaler's STGNN forecast + application-level ACI + SLA budget pacer.

The ORIGINAL DeepScaler autoscaler is unchanged in pipeline/autoscaler.py (rounding rule of the
authors' predict_scale.py + trust threshold of paper Sec. IV-D); run both for the comparison
(deploy/05_online_run.sh ds | aci). Only the decision step differs; Prometheus queries, model, graph
and kubectl calls are the same.

Every period (= data interval, h = 1 step):
  1. feedback for the previous decision: realised demand n = max(1, ceil(req/mu)), replicas actually
     running (Prometheus pod count), SLA violation of the last step (entry p95 > SLO; exact when the
     Istio histogram has a bucket at the SLO, deploy/02_boutique.sh) -> ACI update, pacer record;
  2. outer loop: alpha from the SLA budget (SLAPacer; --pacer off -> fixed --alpha, ACI-only ablation);
  3. forecast yhat (feature 0 of the model, horizon 0): 'demand' model (build_dataset --features
     demand,...) recommended; a 'pod' model also works (ACI calibrates any forecaster against demand);
  4. replicas: active services floor((yhat + c) Q - c), others ceil(yhat); scheduler; kubectl scale;
  5. any failure (Prometheus, model) -> keep the current replicas (logged as "hold").

  python -m pipeline.aci_autoscaler --model_dir model_states/ds_demand --mu reports/mu_sla.json \
      --calib_csv data/raw/train_burst_20261104_0751.csv --eps 0.05 --slo 0.2 --period_min 75
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
import yaml

from pipeline.aci import AppACI, Scheduler, SLAPacer, active_set, decide, demand, ratio_score
from pipeline.collect import SLOW_TEMPLATE, collect, query_range
from pipeline.common import ENTRY, NAMESPACE


def load_model(model_dir, model_name="AdapGLA"):
    """Same loading as pipeline/autoscaler.py (torch imported here so that pipeline.aci_replay
    --forecaster persistence runs without torch)."""
    import torch
    from models import AdapGL
    with open(os.path.join(model_dir, "config_used.yaml")) as f:
        cfg = yaml.safe_load(f)["model"]
    with open(os.path.join(model_dir, "metrics.json")) as f:
        meta = json.load(f)["meta"]
    with open(os.path.join(model_dir, "scaler.json")) as f:
        sc = json.load(f)
    cfg["device"] = "cpu"
    model = getattr(AdapGL, model_name)(**cfg)
    best = os.path.join(model_dir, model_name)
    model.load_state_dict(torch.load(os.path.join(best, f"{model_name}.pkl"), map_location="cpu"))
    model.eval()
    adj = torch.as_tensor(np.load(os.path.join(best, "best_adj_mx.npy")), dtype=torch.float32)
    return model, adj, meta, np.array(sc["mean"]), np.array(sc["std"])


def predict_window(model, adj, x, mean, std):
    """x: (step_in, N, F) raw features -> forecast of feature 0 for the next step, (N,)."""
    import torch
    xs = torch.as_tensor((x[None] - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        out = model(xs, adj).numpy()[0, 0] * std[0] + mean[0]
    return np.nan_to_num(out, nan=1.0)


def features_from(df, services, feats, mu):
    if "demand" in feats:
        for i, s in enumerate(services):
            df[f"{s}_demand"] = demand(df[f"{s}_req"].fillna(0), mu[i])
    return np.stack([np.stack([df[f"{s}_{f}"].ffill().fillna(0).to_numpy() for f in feats], -1)
                     for s in services], 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--model_name", default="AdapGLA")
    p.add_argument("--mu", default=None, help="mu json (default: the one stored in the model meta)")
    p.add_argument("--calib_csv", default=None, help="run CSV for warm-start scores and the active set")
    p.add_argument("--calib_steps", type=int, default=240)
    p.add_argument("--active", default=None, help="comma list of active services (overrides calib)")
    p.add_argument("--prom", default="http://localhost:30090")
    p.add_argument("--namespace", default=NAMESPACE)
    p.add_argument("--min_replicas", type=int, default=1)
    p.add_argument("--max_replicas", type=int, default=10)
    p.add_argument("--slo", type=float, default=0.2, help="SLO on end-to-end p95 (s)")
    p.add_argument("--eps", type=float, default=0.05, help="SLA error budget")
    p.add_argument("--period_min", type=float, default=75, help="accounting period T (minutes)")
    p.add_argument("--pacer", default="on", choices=["on", "off"])
    p.add_argument("--alpha", type=float, default=None, help="fixed alpha when --pacer off (default eps)")
    p.add_argument("--gamma", type=float, default=0.01)
    p.add_argument("--m", type=int, default=120)
    p.add_argument("--c", type=float, default=1.0)
    p.add_argument("--rho", type=float, default=0.05)
    # outer-loop step sizes. Design (tex Sec. 2.4, "On dinh giua hai vong"): the outer loop must be slower
    # than the inner one, eta*|e| << gamma. |e| <= 1 - eps, so eta_tight = gamma/2 keeps eta*|e| <= gamma/2.
    # ORIGINAL (before 10/11/2026 review): SLAPacer defaults eta_tight=0.02, eta_relax=0.002 (> gamma=0.01).
    p.add_argument("--eta_tight", type=float, default=0.005)
    p.add_argument("--eta_relax", type=float, default=0.0005)
    p.add_argument("--a_min", type=float, default=0.01)
    p.add_argument("--a_max", type=float, default=0.5)
    p.add_argument("--pacer_window", type=int, default=120, help="steps of the rolling violation rate")
    p.add_argument("--viol_win", type=int, default=None,
                   help="seconds of the Istio window for the per-step violation v_t (default 2 x step, "
                        "as before; = step matches the design's 'p95 within step t' but needs >= 2 scrapes)")
    p.add_argument("--down_steps", type=int, default=3)
    p.add_argument("--max_down", type=int, default=1)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--log", default="Logs/aci_autoscaler.jsonl")
    a = p.parse_args()

    model, adj, meta, mean, std = load_model(a.model_dir, a.model_name)
    services, feats, step_in = meta["services"], meta["features"], meta["step_in"]
    step = int(meta["interval_s"] or 30)
    if a.mu:
        raw = json.load(open(a.mu))
        mu = np.array([float(raw[s]["mu"] if isinstance(raw[s], dict) else raw[s]) for s in services])
    elif "mu" in meta:
        mu = np.array([meta["mu"][s] for s in services])
    else:
        raise SystemExit("--mu needed (the model meta has no mu: it was trained on 'pod')")
    if feats[0] != "demand":
        print("NOTE: model target is", feats[0], "- ACI calibrates it against demand anyway")
    N = len(services)

    # ---- warm start and active set
    warm, act = [], np.ones(N, bool)
    if a.calib_csv:
        cdf = pd.read_csv(a.calib_csv).tail(a.calib_steps + step_in).reset_index(drop=True)
        X = features_from(cdf.copy(), services, feats, mu)
        n_c = np.stack([demand(cdf[f"{s}_req"].fillna(0), mu[i]) for i, s in enumerate(services)], 1)
        act = active_set(n_c, a.rho)
        for t in range(step_in, len(cdf)):
            yh = predict_window(model, adj, X[t - step_in:t], mean, std)
            warm.append(float(ratio_score(n_c[t, act], yh[act], a.c).max()))
    if a.active:
        act = np.array([s in a.active.split(",") for s in services])
    alpha0 = a.alpha if a.alpha is not None else a.eps
    aci = AppACI(act, m=a.m, gamma=a.gamma, c=a.c, beta0=alpha0, warm_scores=warm[-a.m:])
    pacer = SLAPacer(a.eps, int(a.period_min * 60 / step), window=a.pacer_window, eta_tight=a.eta_tight,
                     eta_relax=a.eta_relax, a_min=a.a_min, a_max=a.a_max,
                     enabled=(a.pacer == "on"), alpha_fixed=alpha0)
    # ORIGINAL: pacer = SLAPacer(a.eps, int(a.period_min * 60 / step), enabled=(a.pacer == "on"), alpha_fixed=alpha0)
    viol_win = a.viol_win or 2 * step
    sched = Scheduler(N, a.min_replicas, a.max_replicas, a.down_steps, a.max_down)
    slo_ms = int(round(a.slo * 1000))
    print(f"ACI autoscaler: step={step}s active={[s for s, f in zip(services, act) if f]} "
          f"warm={len(warm)} pacer={a.pacer} eps={a.eps} dry_run={a.dry_run}", flush=True)
    os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)

    pending = None                           # (target step end, yhat, deployed placeholder)
    while True:
        t0 = time.time()
        end = t0 - t0 % step
        rec = {"time": end}
        try:
            df = collect(a.prom, a.namespace, end - (step_in + 2) * step, end, step, services=services,
                         metrics=sorted(set(feats) - {"demand"} | {"pod", "req"}), verbose=False)
            if len(df) < step_in:
                raise RuntimeError(f"only {len(df)} steps from Prometheus")
            X = features_from(df.copy(), services, feats, mu)
            n_now = np.stack([demand(df[f"{s}_req"].fillna(0).to_numpy()[-1:], mu[i]) for i, s in enumerate(services)], 1)[0]
            running = df[[f"{s}_pod" for s in services]].to_numpy()[-1]
            lat = collect(a.prom, a.namespace, end - step, end, step, services=[ENTRY], metrics=["p95"],
                          verbose=False)[f"{ENTRY}_p95"].to_numpy()[-1]
            slow = query_range(a.prom, SLOW_TEMPLATE.format(ns=a.namespace, svc=ENTRY, ms=slo_ms, win=f"{viol_win}s"),
                               end - step, end, step).get(float(end))
            violated = bool(slow > 0.05) if slow is not None and np.isfinite(slow) else bool(lat > a.slo)
            # 1. feedback for the previous decision (its target step has just ended)
            alpha_prev = pacer.alpha_fixed if pending is None else pending["alpha"]
            missed = False
            if pending is not None:
                missed, score = aci.observe(pending["yhat"], n_now, running, alpha_prev)
                rec.update(score=score)
            pacer.record(violated, missed)
            # 2. outer loop
            alpha, pinfo = pacer.alpha()
            # 3. forecast + 4. decision
            yhat = predict_window(model, adj, X[-step_in:], mean, std)
            pods = decide(yhat, aci, sched, running, a.min_replicas, a.max_replicas)
            for s, n_new, n_old in zip(services, pods, running.astype(int)):
                if n_new != n_old and not a.dry_run:
                    subprocess.run(["kubectl", "scale", "deployment", s, "-n", a.namespace,
                                    f"--replicas={int(n_new)}"], check=False, capture_output=True)
            pending = {"yhat": yhat, "alpha": alpha}
            rec.update(yhat=yhat.round(3).tolist(), demand=n_now.tolist(), running=running.astype(int).tolist(),
                       pods=pods.astype(int).tolist(), violated=violated, slow=slow, p95=float(lat),
                       missed=missed, alpha=alpha, beta=aci.beta, Q=aci.quantile(), **pinfo)
            tag = ""
        except Exception as ex:                               # fail-safe: keep current replicas
            rec.update(hold=True, error=repr(ex)[:200])
            # the pending forecast was for the step that just ended; next period would compare it with the
            # demand one step later. Drop it (no ACI update for this decision). ORIGINAL: pending kept.
            pending = None
            tag = "[hold] " + repr(ex)[:80]
        with open(a.log, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        if "pods" in rec:
            print(time.strftime("%H:%M:%S"), f"a={rec['alpha']:.3f} b={rec['beta']:.3f} Q={rec['Q']:.2f}",
                  dict(zip([s.replace("service", "") for s in services], rec["pods"])), flush=True)
        else:
            print(time.strftime("%H:%M:%S"), tag, flush=True)
        time.sleep(max(0.0, step - (time.time() - t0)))


if __name__ == "__main__":
    main()
