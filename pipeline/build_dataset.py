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

Thesis option (objective O1, 10/11/2026): feature 0 may be "demand" instead of "pod":
  demand_i(t) = max(1, ceil(req_i(t) / mu_i)), mu_i from --mu (reports/mu_sla.json or reports/mu.json).
The model then forecasts the replicas the load needs, not the replicas the autoscaler that ran during
data collection chose (DeepScaler's label y_t^i = x_{t+1}^{c,i}, paper Sec. III-A). The mu values are
saved in meta.json so that the online autoscaler computes the same channel.
  python -m pipeline.build_dataset --raw data/raw/X.csv --features demand,cpu,res,req,mem \
      --mu reports/mu_sla.json --out_dir data/boutique_demand
ORIGINAL behaviour (DeepScaler, kept): --features pod,cpu,res,req,mem (default), no --mu.
Splits are chronological (no shuffling) so test data is strictly "future".
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from pipeline.common import FEATURES, SERVICES, TARGET_FEATURE, call_graph_adj


def load_mu(path):
    raw = json.load(open(path))
    return {s: float(raw[s]["mu"] if isinstance(raw[s], dict) else raw[s]) for s in raw}


def add_demand(df, mu, services=SERVICES):
    """demand = max(1, ceil(req / mu)) per service (thesis O1); returns a copy with <svc>_demand columns."""
    df = df.copy()
    for s in services:
        df[f"{s}_demand"] = np.maximum(1.0, np.ceil(df[f"{s}_req"].fillna(0.0).to_numpy() / mu[s]))
    return df


def load_raw(csv_path, services=SERVICES, features=FEATURES, mu=None):
    df = pd.read_csv(csv_path)
    if "demand" in features:                    # thesis option; original DeepScaler never takes this branch
        if mu is None:
            raise ValueError("feature 'demand' needs --mu")
        df = add_demand(df, mu, services)
    missing = [f"{s}_{f}" for s in services for f in features if f"{s}_{f}" not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing[:5]}...")
    df = df.ffill().fillna(0.0)
    data = np.stack([np.stack([df[f"{s}_{f}"].to_numpy(dtype=np.float64) for f in features], axis=-1)
                     for s in services], axis=1)                       # (T, N, F)
    interval = float(np.median(np.diff(df["time"]))) if "time" in df and len(df) > 1 else None
    return data, interval


def cc_eq16(data_train, features=FEATURES, feature="req"):
    """CC relationship, eq. (16) of the DeepScaler paper: |correlation| of the RANKS of the history of
    node i and j (Spearman), kept only if above the threshold T_Ci of node i. The paper does not give
    T_Ci; we use the same rule as eq. (15): T_Ci = max_j |rho_ij| / 2 (fixed a priori, not tuned)."""
    s = data_train[:, :, features.index(feature)]
    r = np.apply_along_axis(lambda c: pd.Series(c).rank().to_numpy(), 0, s)
    rho = np.abs(np.nan_to_num(np.corrcoef(r.T)))
    np.fill_diagonal(rho, 0.0)
    thr = rho.max(axis=1, keepdims=True) / 2
    return np.where(rho > thr, rho, 0.0).astype(np.float32)


def od_eq15(npas):
    """OD relationship, eq. (15): A_ij = N_pas(i,j) (mean requests i->j) if N_pas(i,j) > max_j A_ij / 2."""
    npas = np.asarray(npas, dtype=np.float64)
    thr = npas.max(axis=1, keepdims=True) / 2
    return np.where((npas > thr) & (npas > 0), npas, 0.0).astype(np.float32)


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
    p.add_argument("--mu", default=None, help="mu json for the 'demand' feature (thesis O1)")
    p.add_argument("--features", default=",".join(FEATURES),
                   help="input features, first must be 'pod' (target). Paper: pod,cpu,res,req,mem; "
                        "original predict_scale.py: pod,cpu,res,req")
    p.add_argument("--od_adj", default=None,
                   help=".npy call graph from collect.py; default = static Online-Boutique graph")
    p.add_argument("--npas", default=None,
                   help="*_npas.npy from collect.py (mean req/s per call edge) -> adj_od15.npy (eq. 15)")
    p.add_argument("--stride", type=int, default=1,
                   help="keep every k-th sample: 30 s data with --stride 2 = 60 s interval of data_process.py")
    p.add_argument("--split_by", default="segment", choices=["segment", "target"],
                   help="segment: windows lie fully inside each split (old behaviour); target: a window "
                        "belongs to the split of the step it predicts, its history may come from the "
                        "previous split (no label leakage; needed for long windows such as step_in=80)")
    a = p.parse_args()

    features = [f.strip() for f in a.features.split(",")]
    # ORIGINAL: assert features[0] == "pod", "feature 0 must be 'pod' (prediction target)"
    #           data, interval = load_raw(a.raw, features=features)
    assert features[0] in ("pod", "demand"), "feature 0 is the prediction target: 'pod' (original) or 'demand'"
    mu = load_mu(a.mu) if a.mu else None
    data, interval = load_raw(a.raw, features=features, mu=mu)
    if a.stride > 1:
        data = data[::a.stride]
        interval = interval * a.stride if interval else None
    T = data.shape[0]
    fr = [float(v) for v in a.split.split(",")]
    n_tr, n_va = int(T * fr[0]), int(T * fr[1])
    bounds = {"train": (0, n_tr), "valid": (n_tr, n_tr + n_va), "test": (n_tr + n_va, T)}
    parts = {k: data[b0:b1] for k, (b0, b1) in bounds.items()}

    os.makedirs(a.out_dir, exist_ok=True)
    sizes = {}
    if a.split_by == "target":
        X, Y = make_windows(data, a.step_in, a.horizon)          # window k predicts step k + step_in
        tgt = np.arange(len(X)) + a.step_in
    for name, seg in parts.items():
        if a.split_by == "target":
            b0, b1 = bounds[name]
            m = (tgt >= b0) & (tgt + a.horizon - 1 < b1)
            x, y = X[m], Y[m]
            if len(x) == 0:
                raise ValueError(f"{name} split has no window; collect more data or reduce --step_in")
        else:
            if seg.shape[0] < a.step_in + a.horizon + 1:
                raise ValueError(f"{name} split has only {seg.shape[0]} steps; collect more data, "
                                 f"reduce --step_in or use --split_by target")
            x, y = make_windows(seg, a.step_in, a.horizon)
        np.savez_compressed(os.path.join(a.out_dir, f"{name}.npz"), x=x, y=y)
        sizes[name] = int(x.shape[0])
        print(f"{name:5s}: x {x.shape}  y {y.shape}")

    od = np.load(a.od_adj) if a.od_adj else call_graph_adj()
    np.save(os.path.join(a.out_dir, "adj_od.npy"), od.astype(np.float32))
    np.save(os.path.join(a.out_dir, "adj_cc.npy"), corr_adj(parts["train"], features))
    np.save(os.path.join(a.out_dir, "adj_cc16.npy"), cc_eq16(parts["train"], features))
    if a.npas:
        np.save(os.path.join(a.out_dir, "adj_od15.npy"), od_eq15(np.load(a.npas)))
    meta = {"services": SERVICES, "features": features, "target": features[TARGET_FEATURE],
            "interval_s": interval, "step_in": a.step_in, "horizon": a.horizon, "stride": a.stride,
            "split": a.split, "split_by": a.split_by,
            "raw": os.path.abspath(a.raw), "samples": sizes}
    if "demand" in features:                     # the online autoscaler must build the same channel
        meta["mu"] = mu
        meta["mu_file"] = os.path.abspath(a.mu)
    with open(os.path.join(a.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("saved to", a.out_dir)


if __name__ == "__main__":
    main()
