"""Turn a raw metrics CSV into DeepScaler training tensors.

Input : CSV with columns  time, <svc>_<feature>  (from pipeline/collect.py or
        pipeline/simulate.py). Missing/NaN values are forward-filled, then 0.
Output (in --out_dir):
  train.npz / valid.npz / test.npz   x: (S, step_in, N, F)   y: (S, horizon, N)
  adj_od.npy   call-graph adjacency (Istio topology, or the static Boutique graph)
  adj_cc.npy   |Pearson correlation| of request rates on the training split
  meta.json    services, features, interval, split sizes

Same windowing as the original prepareData.py: x = the `step_in` most recent
steps of all features, y = pod count (feature 0) of the next `horizon` steps.
Splits are chronological (no shuffling) so test data is strictly "future".
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from pipeline.common import FEATURES, SERVICES, TARGET_FEATURE, call_graph_adj


def load_raw(csv_path, services=SERVICES, features=FEATURES):
    df = pd.read_csv(csv_path)
    missing = [f"{s}_{f}" for s in services for f in features if f"{s}_{f}" not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing[:5]}...")
    df = df.ffill().fillna(0.0)
    data = np.stack([np.stack([df[f"{s}_{f}"].to_numpy(dtype=np.float64) for f in features], axis=-1)
                     for s in services], axis=1)                       # (T, N, F)
    interval = float(np.median(np.diff(df["time"]))) if "time" in df and len(df) > 1 else None
    return data, interval


def make_windows(data, step_in, horizon):
    T = data.shape[0]
    xs, ys = [], []
    for start in range(0, T - step_in - horizon + 1):
        xs.append(data[start:start + step_in])
        ys.append(data[start + step_in:start + step_in + horizon, :, TARGET_FEATURE])
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def corr_adj(data_train, features=FEATURES, threshold=0.3):
    req = data_train[:, :, features.index("req")]
    std = req.std(axis=0)
    c = np.corrcoef(req.T) if np.all(std > 0) else np.nan_to_num(np.corrcoef(req.T))
    a = np.abs(np.nan_to_num(c))
    a[a < threshold] = 0.0
    np.fill_diagonal(a, 0.0)
    return a.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", required=True, help="raw metrics CSV")
    p.add_argument("--out_dir", default="data/boutique")
    p.add_argument("--step_in", type=int, default=20, help="history length (original repo: 80)")
    p.add_argument("--horizon", type=int, default=1, help="steps to predict (original repo: 1)")
    p.add_argument("--split", default="0.7,0.1,0.2")
    p.add_argument("--features", default=",".join(FEATURES),
                   help="input features, first must be 'pod' (target). Paper: pod,cpu,res,req,mem; "
                        "original predict_scale.py: pod,cpu,res,req")
    p.add_argument("--od_adj", default=None,
                   help=".npy call graph from collect.py; default = static Online-Boutique graph")
    a = p.parse_args()

    features = [f.strip() for f in a.features.split(",")]
    assert features[0] == "pod", "feature 0 must be 'pod' (prediction target)"
    data, interval = load_raw(a.raw, features=features)
    T = data.shape[0]
    fr = [float(v) for v in a.split.split(",")]
    n_tr, n_va = int(T * fr[0]), int(T * fr[1])
    parts = {"train": data[:n_tr], "valid": data[n_tr:n_tr + n_va], "test": data[n_tr + n_va:]}

    os.makedirs(a.out_dir, exist_ok=True)
    sizes = {}
    for name, seg in parts.items():
        if seg.shape[0] < a.step_in + a.horizon + 1:
            raise ValueError(f"{name} split has only {seg.shape[0]} steps; collect more data "
                             f"or reduce --step_in")
        x, y = make_windows(seg, a.step_in, a.horizon)
        np.savez_compressed(os.path.join(a.out_dir, f"{name}.npz"), x=x, y=y)
        sizes[name] = int(x.shape[0])
        print(f"{name:5s}: x {x.shape}  y {y.shape}")

    od = np.load(a.od_adj) if a.od_adj else call_graph_adj()
    np.save(os.path.join(a.out_dir, "adj_od.npy"), od.astype(np.float32))
    np.save(os.path.join(a.out_dir, "adj_cc.npy"), corr_adj(parts["train"], features))
    meta = {"services": SERVICES, "features": features, "target": features[TARGET_FEATURE],
            "interval_s": interval, "step_in": a.step_in, "horizon": a.horizon,
            "raw": os.path.abspath(a.raw), "samples": sizes}
    with open(os.path.join(a.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("saved to", a.out_dir)


if __name__ == "__main__":
    main()
