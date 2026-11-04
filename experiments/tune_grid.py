"""Tuning grid for DeepScaler on Colab: only knobs NOT fixed by the authors' code are varied.

Fixed (authors' code / paper, never changed here):
  model  config/train_dataset_speed.yaml AdapGLA: num_block 2, kernel 5, cheb/time filters 128, K 3,
         init_feature_num 64, batch 64
  train  config/train_config.yaml: Adam lr 1e-3, wd 1e-4, CosineAnnealing(T_max 15, eta_min 1e-5),
         StandardScaler; trainer: EarlyStop(5), +5 epochs at mid iteration, epsilon = 0.5/N (eq. 6),
         delta = 0.05 (eq. 7), max_graph_num 3 (main.py / predict_scale.py)
  data   features pod,cpu,res,req,mem (data_process.py, Fig. 4), step_in 80 (points_per_hour=80),
         horizon 1 (num_for_predict=1), initial graph = invocation graph only (Sec. IV-B, V-C)
Varied (two values written in the code/paper, or not specified anywhere):
  interval     30 s (paper Sec. V-B) | 60 s (data_process.py interval=60, paper Sec. VI)
  epoch/iter   1/1 (main.py defaults) | 5/5 (predict_scale.py defaults)
  split        0.6/0.2/0.2 | 0.7/0.1/0.2 (no split in the authors' code)
  seeds        2021 (train_config.yaml), 2022, 2023 -> mean +- std, never picked
Baselines (Fig. 12) DCRNN / T-GCN on every data config, same seeds, max 300 epochs, patience 20.

  python tune_grid.py --raw /content/run2_hpa_5f.csv --od /content/run2_hpa_od.npy --out /content/grid
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time

import yaml

p = argparse.ArgumentParser()
p.add_argument("--raw", required=True)
p.add_argument("--od", required=True)
p.add_argument("--npas", default=None)
p.add_argument("--out", default="grid")
p.add_argument("--seeds", default="2021,2022,2023")
p.add_argument("--no_baselines", action="store_true")
a = p.parse_args()
seeds = [int(s) for s in a.seeds.split(",")]
os.makedirs(a.out, exist_ok=True)
ENV = {**os.environ, "PYTHONWARNINGS": "ignore"}
base_cfg = yaml.safe_load(open("config/train_config.yaml"))


def run(cmd, log):
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=ENV)
    open(log, "w").write(r.stdout + r.stderr)
    print(f"{'OK ' if r.returncode == 0 else 'ERR'} {time.time() - t0:5.0f}s {os.path.basename(log)}", flush=True)
    return r.returncode == 0


rows = []
for stride, split in itertools.product([1, 2], ["0.6,0.2,0.2", "0.7,0.1,0.2"]):
    tag = f"i{30 * stride}_s{split.split(',')[0][2:]}"
    data_dir = f"{a.out}/data_{tag}"
    cmd = [sys.executable, "-m", "pipeline.build_dataset", "--raw", a.raw, "--od_adj", a.od, "--out_dir", data_dir,
           "--step_in", 80, "--split", split, "--split_by", "target", "--stride", stride]
    if a.npas:
        cmd += ["--npas", a.npas]
    if not run(cmd, f"{a.out}/build_{tag}.log"):
        continue
    for (ep, it), seed in itertools.product([(1, 1), (5, 5)], seeds):
        cfg = f"/tmp/tc_{seed}.yaml"
        yaml.safe_dump({**base_cfg, "seed": seed}, open(cfg, "w"))
        out = f"{a.out}/ds_{tag}_e{ep}i{it}_s{seed}"
        if run([sys.executable, "train.py", "--data_dir", data_dir, "--out_dir", out, "--train_config_path", cfg,
                "--num_epoch", ep, "--num_iter", it, "--adj", "od"], f"{out}.log"):
            m = json.load(open(f"{out}/metrics.json"))
            rows.append({"model": "DeepScaler", "data": tag, "epoch_iter": f"{ep}/{it}", "seed": seed,
                         **{k: m["results"]["test"]["DeepScaler(AdapGLA)"][k] for k in ("mae", "rmse", "mape", "pods_exact_match")},
                         **{k: m["graph"].get(k) for k in ("learned_edges(off-diag)", "fig11_sim_od", "fig11_sim_cc",
                                                           "fig11_identity_baseline_od", "diag_mean")},
                         "persist_mae": m["results"]["test"]["Persistence(last pods)"]["mae"]})
    if a.no_baselines:
        continue
    for model, seed in itertools.product(["DCRNN", "TGCN"], seeds):
        out = f"{a.out}/bl_{tag}_{model}_s{seed}"
        if run([sys.executable, "train_baselines.py", "--data_dir", data_dir, "--model", model, "--seed", seed,
                "--out_dir", out, "--max_epoch", 300, "--patience", 20], f"{out}.log"):
            m = json.load(open(f"{out}/metrics.json"))
            rows.append({"model": model, "data": tag, "epoch_iter": "-", "seed": seed,
                         **{k: m["results"]["test"][model][k] for k in ("mae", "rmse", "mape", "pods_exact_match")},
                         "persist_mae": m["results"]["test"]["Persistence(last pods)"]["mae"]})
    json.dump(rows, open(f"{a.out}/results.json", "w"), indent=1)

json.dump(rows, open(f"{a.out}/results.json", "w"), indent=1)
open(f"{a.out}/DONE", "w").write("done")
print("DONE", len(rows), "runs")
