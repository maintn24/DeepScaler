"""Free hyper-parameter search for DeepScaler (AdapGLA) on our cluster data: values written in the
authors' code may change here, because our deployment (1 kind machine, c6i.2xlarge, 10 services,
30 s steps, 6 h of data) differs from the paper's (8 VMs, 144 cores, Sec. V-A).

Selection is on the VALIDATION MAE only; the test split is reported for the final configs, never
used to choose. Two stages:
  1. one-factor-at-a-time from BASE (seed 2021): each factor value is tried alone.
  2. BEST = BASE + every factor value that beat BASE on validation; BASE, BEST and the authors'
     code defaults (5/5) are then run with 3 seeds (mean +- std).

  python experiments/tune_free.py --raw /content/x.csv --od /content/x_od.npy --npas /content/x_npas.npy --out /content/free
"""
import argparse
import copy
import json
import os
import subprocess
import sys
import time

import numpy as np
import yaml

p = argparse.ArgumentParser()
p.add_argument("--raw", required=True)
p.add_argument("--od", required=True)
p.add_argument("--npas", default=None)
p.add_argument("--out", default="free")
p.add_argument("--split", default="0.7,0.1,0.2")
p.add_argument("--seeds", default="2021,2022,2023")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
ENV = {**os.environ, "PYTHONWARNINGS": "ignore"}
MODEL0 = yaml.safe_load(open("config/train_dataset_speed.yaml"))
TRAIN0 = yaml.safe_load(open("config/train_config.yaml"))

BASE = dict(epoch=30, iter=6, step_in=80, lr=1e-3, filters=128, blocks=2, kernel=5, batch=64,
            adj="od", epsilon=None, delta=None)
FACTORS = {
    "epoch": [15, 50],
    "step_in": [20, 40],
    "lr": [3e-4, 3e-3],
    "filters": [64],
    "blocks": [1, 3],
    "kernel": [3],
    "batch": [32],
    "adj": ["od,cc"],
    "epsilon": [0.01, 0.1],          # code: 0.5/N = 0.05 for N = 10 (eq. 6)
    "delta": [0.2],                  # code: 0.05 (eq. 7)
}
CODE = dict(BASE, epoch=5, iter=5)   # authors' predict_scale.py budget


def run(cmd, log):
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=ENV)
    open(log, "w").write(r.stdout + r.stderr)
    print(f"{'OK ' if r.returncode == 0 else 'ERR'} {time.time() - t0:5.0f}s {os.path.basename(log)}", flush=True)
    return r.returncode == 0


def dataset(step_in):
    d = f"{a.out}/data_in{step_in}"
    if not os.path.exists(f"{d}/meta.json"):
        cmd = [sys.executable, "-m", "pipeline.build_dataset", "--raw", a.raw, "--od_adj", a.od, "--out_dir", d,
               "--step_in", step_in, "--split", a.split, "--split_by", "target"]
        if a.npas:
            cmd += ["--npas", a.npas]
        run(cmd, f"{d}.log")
    return d


def train(cfg, seed, name):
    out = f"{a.out}/{name}_s{seed}"
    if not os.path.exists(f"{out}/metrics.json"):
        m = copy.deepcopy(MODEL0)
        m["AdapGLA"].update(num_cheb_filter=cfg["filters"], num_time_filter=cfg["filters"],
                            num_block=cfg["blocks"], kernel_size=cfg["kernel"])
        m["dataset"]["batch_size"] = cfg["batch"]
        t = copy.deepcopy(TRAIN0)
        t.update(seed=seed, learning_rate=cfg["lr"])
        t["lr_scheduler"]["T_max"] = cfg["epoch"]
        mp, tp = f"{out}_model.yaml", f"{out}_train.yaml"
        yaml.safe_dump(m, open(mp, "w")); yaml.safe_dump(t, open(tp, "w"))
        cmd = [sys.executable, "train.py", "--data_dir", dataset(cfg["step_in"]), "--out_dir", out,
               "--model_config_path", mp, "--train_config_path", tp, "--num_epoch", cfg["epoch"],
               "--num_iter", cfg["iter"], "--adj", cfg["adj"]]
        for k in ("epsilon", "delta"):
            if cfg[k] is not None:
                cmd += [f"--{k}", cfg[k]]
        if not run(cmd, f"{out}.log"):
            return None
    mt = json.load(open(f"{out}/metrics.json"))
    r = mt["results"]
    g = mt.get("graph", {})
    return {"name": name, "seed": seed, **{k: v for k, v in cfg.items()},
            "valid_mae": r["valid"]["DeepScaler(AdapGLA)"]["mae"], "test_mae": r["test"]["DeepScaler(AdapGLA)"]["mae"],
            "test_persist": r["test"]["Persistence(last pods)"]["mae"], "edges": g.get("learned_edges(off-diag)"),
            "sim_od": g.get("fig11_sim_od"), "train_s": mt.get("train_time_s")}


rows = []
save = lambda: json.dump(rows, open(f"{a.out}/results.json", "w"), indent=1, default=str)
base = train(BASE, 2021, "base"); rows.append(base); save()
best = dict(BASE)
for f, vals in FACTORS.items():
    cand = [(base["valid_mae"], BASE[f])]
    for v in vals:
        r = train(dict(BASE, **{f: v}), 2021, f"{f}={v}")
        if r:
            rows.append(r); save()
            cand.append((r["valid_mae"], v))
    best[f] = min(cand, key=lambda c: c[0])[1]          # lowest validation MAE (ties -> BASE)
print("BEST", best, flush=True)
json.dump(best, open(f"{a.out}/best.json", "w"), indent=1, default=str)
for name, cfg in (("final_code", CODE), ("final_base", BASE), ("final_best", best)):
    for s in [int(x) for x in a.seeds.split(",")]:
        r = train(cfg, s, name)
        if r:
            rows.append(r); save()
for name in ("final_code", "final_base", "final_best"):
    v = [x for x in rows if x["name"] == name]
    print(name, "valid %.3f" % np.mean([x["valid_mae"] for x in v]),
          "test %.3f +- %.3f" % (np.mean([x["test_mae"] for x in v]), np.std([x["test_mae"] for x in v])), flush=True)
open(f"{a.out}/DONE", "w").write("done")
