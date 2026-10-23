"""DeepScaler online autoscaler (cleaned-up predict_scale.py).

Every `--period` seconds:
  1. read the last `step_in` samples of [pod, cpu, res, req] for every service
     from Prometheus (same PromQL as training data, same sampling step);
  2. predict next-step pods with the trained AdapGLA + learned graph;
  3. round with the original rule (ceil unless fraction < 0.35), clip to
     [min, max] replicas and apply with `kubectl scale`.
Remove the HPAs first (`kubectl delete hpa --all -n boutique`), otherwise the
HPA and DeepScaler fight over the replica count.

  python -m pipeline.autoscaler --model_dir model_states/boutique_cluster \
      --prom http://localhost:30090 --namespace boutique --period 30 [--dry_run]
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch
import yaml

from models import AdapGL
from pipeline.collect import collect
from pipeline.common import SERVICES


def scale_to_pods(pred, restriction=0.35):
    frac = pred - np.floor(pred)
    return np.where(frac < restriction, np.floor(pred), np.ceil(pred))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_dir", required=True, help="out_dir of train.py")
    p.add_argument("--model_name", default="AdapGLA")
    p.add_argument("--prom", default="http://localhost:30090")
    p.add_argument("--namespace", default="boutique")
    p.add_argument("--period", type=int, default=None, help="seconds between decisions (default = data interval)")
    p.add_argument("--min_replicas", type=int, default=1)
    p.add_argument("--max_replicas", type=int, default=10)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--log", default="Logs/autoscaler.jsonl")
    a = p.parse_args()

    with open(os.path.join(a.model_dir, "config_used.yaml")) as f:
        cfg = yaml.safe_load(f)["model"]
    with open(os.path.join(a.model_dir, "metrics.json")) as f:
        meta = json.load(f)["meta"]
    with open(os.path.join(a.model_dir, "scaler.json")) as f:
        sc = json.load(f)
    mean, std = np.array(sc["mean"]), np.array(sc["std"])
    step = int(meta["interval_s"] or 30)
    step_in = meta["step_in"]
    features = meta["features"]                  # same feature order as training
    period = a.period or step

    device = torch.device("cpu")
    cfg["device"] = "cpu"
    model = getattr(AdapGL, a.model_name)(**cfg)
    best_dir = os.path.join(a.model_dir, a.model_name)
    model.load_state_dict(torch.load(os.path.join(best_dir, f"{a.model_name}.pkl"), map_location=device))
    model.eval()
    adj = torch.as_tensor(np.load(os.path.join(best_dir, "best_adj_mx.npy")), dtype=torch.float32)

    os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True)
    print(f"DeepScaler autoscaler: step={step}s window={step_in} period={period}s dry_run={a.dry_run}")
    while True:
        t0 = time.time()
        end = t0 - t0 % step
        df = collect(a.prom, a.namespace, end - (step_in + 2) * step, end, step,
                     metrics=features, verbose=False)
        if len(df) < step_in:
            print(f"only {len(df)} steps from Prometheus (need {step_in}), skipping this period")
            time.sleep(max(0.0, period - (time.time() - t0)))
            continue
        x = np.stack([np.stack([df[f"{s}_{f}"].to_numpy()[-step_in:] for f in features], -1)
                      for s in SERVICES], axis=1)[None]          # (1, step_in, N, F)
        xs = torch.as_tensor((x - mean) / std, dtype=torch.float32)
        with torch.no_grad():
            pred = model(xs, adj).numpy()[0, 0] * std[0] + mean[0]  # (N,)
        pred = np.nan_to_num(pred, nan=1.0)
        pods = np.clip(scale_to_pods(pred), a.min_replicas, a.max_replicas).astype(int)
        current = x[0, -1, :, 0].astype(int)
        for s, n_new, n_old in zip(SERVICES, pods, current):
            if n_new != n_old and not a.dry_run:
                subprocess.run(["kubectl", "scale", "deployment", s, "-n", a.namespace,
                                f"--replicas={n_new}"], check=False, capture_output=True)
        rec = {"time": end, "pred": pred.round(3).tolist(), "pods": pods.tolist(), "current": current.tolist()}
        with open(a.log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(time.strftime("%H:%M:%S"), dict(zip([s.replace("service", "") for s in SERVICES], pods.tolist())))
        time.sleep(max(0.0, period - (time.time() - t0)))


if __name__ == "__main__":
    main()
