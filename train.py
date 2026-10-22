"""Train and test DeepScaler's predictor (AdapGLA + adaptive graph learning).

Fixed / reproducible replacement for main.py:
  * the StandardScaler is actually fitted on the training split (in main.py the
    fit was guarded by a file name that never matched, so training crashed);
  * dataset shape (nodes, features, window) is read from <data_dir>/meta.json;
  * uses the original configs (config/train_dataset_speed.yaml + config/train_config.yaml);
    only dataset-shape fields and graph paths are overridden from <data_dir>/meta.json;
  * falls back to CPU when the configured CUDA device is not available;
  * after training it evaluates the best checkpoint on the test split against a
    persistence baseline and saves everything needed later (predictions on the
    validation and test splits, e.g. for conformal calibration).

Example:
  python train.py --data_dir data/boutique_sim --out_dir model_states/boutique_sim \
                  --num_epoch 15 --num_iter 6
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

import trainer
from dataset import TPDataset
from models import AdapGL
from models.adj_mx import get_adj
from utils import scaler as scaler_mod
from utils.metrics import get_mae, get_mape, get_rmse


def paper_similarity(a1, a2):
    """Weighted Jaccard similarity exactly as in the authors' similarity.py (used for Fig. 11):
    graphs built with networkx.from_numpy_array (undirected, self-loops kept); for every node,
    sum min/max of the weights over edges present in BOTH graphs, then average over nodes."""
    import networkx as nx
    g1, g2 = nx.from_numpy_array(np.asarray(a1, dtype=float)), nx.from_numpy_array(np.asarray(a2, dtype=float))
    inter, union = {}, {}
    for n in g1.nodes():
        if n in g2:
            for m in g1.neighbors(n):
                if m in g2[n]:
                    w1, w2 = g1[n][m]["weight"], g2[n][m]["weight"]
                    inter[n] = inter.get(n, 0.0) + min(w1, w2)
                    union[n] = union.get(n, 0.0) + max(w1, w2)
    sims = [inter[n] / union[n] for n in inter if union[n] > 0]
    return float(sum(sims) / len(sims)) if sims else 0.0


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def scale_to_pods(pred, restriction=0.35):
    """Rounding rule of predict_scale.py: ceil unless the fraction < restriction."""
    frac = pred - np.floor(pred)
    pods = np.where(frac < restriction, np.floor(pred), np.ceil(pred))
    return np.maximum(pods, 1)


def scaling_report(pred, true):
    pods = scale_to_pods(pred)
    return {
        "mae": float(get_mae(pred, true)),
        "rmse": float(get_rmse(pred, true)),
        "mape": float(get_mape(pred, true)),
        "pods_exact_match": float((pods == true).mean()),
        "under_provision_rate": float((pods < true).mean()),
        "over_provision_rate": float((pods > true).mean()),
        "avg_pods_pred": float(pods.mean()),
        "avg_pods_true": float(true.mean()),
    }


@torch.no_grad()
def predict(model, adj, x_scaled, device, batch_size=256):
    model.eval()
    out = []
    for i in range(0, x_scaled.shape[0], batch_size):
        xb = torch.as_tensor(x_scaled[i:i + batch_size], dtype=torch.float32, device=device)
        out.append(model(xb, adj).cpu().numpy())
    return np.concatenate(out, axis=0)


def main(args):
    model_cfg = load_yaml(args.model_config_path)
    train_cfg = load_yaml(args.train_config_path)
    torch.manual_seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])

    data_cfg = model_cfg["dataset"]
    data_dir = args.data_dir or data_cfg["data_dir"]
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    data_cfg.update({
        "data_dir": data_dir,
        "num_nodes": len(meta["services"]),
        "input_size": len(meta["features"]),
        "step_num_in": meta["step_in"],
        "step_num_out": meta["horizon"],
    })
    dev = str(data_cfg.get("device", "auto"))
    if dev == "auto" or (dev.startswith("cuda") and not torch.cuda.is_available()):
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev)
    print(f"device={device}  data={data_dir}  meta={meta['samples']}")

    # ----------------------------- data -----------------------------
    Scaler = getattr(scaler_mod, train_cfg["scaler"])
    data_scaler = Scaler(axis=(0, 1, 2))           # one mean/std per feature
    raw = {}
    loaders = []
    for i, split in enumerate(("train", "valid", "test")):
        ds = TPDataset(os.path.join(data_dir, f"{split}.npz"))
        raw[split] = {k: v.copy() for k, v in ds.data.items()}
        if split == "train":
            data_scaler.fit(ds.data["x"])
            data_scaler.std = np.where(data_scaler.std == 0, 1.0, data_scaler.std)
        ds.fit(data_scaler)
        loaders.append(DataLoader(ds, batch_size=data_cfg["batch_size"],
                                  shuffle=(split == "train" and args.shuffle)))

    # ----------------------------- model ----------------------------
    net_name = args.model_name
    net_cfg = dict(model_cfg[net_name])
    net_cfg.update(data_cfg)
    adj_names = [s.strip() for s in args.adj.split(",")]
    net_cfg["adj_mx_path"] = ",".join(os.path.join(data_dir, f"adj_{a}.npy") for a in adj_names)

    Model = getattr(AdapGL, net_name)
    net_pred = Model(**net_cfg).to(device)
    net_graph = AdapGL.GraphLearn(net_cfg["num_nodes"], net_cfg["init_feature_num"]).to(device)
    if args.epsilon is not None:            # paper eq. (6): threshold epsilon in (0,1), value not given
        net_graph.epsilon = args.epsilon    # original code: 0.5 / num_nodes
    Optimizer = getattr(torch.optim, train_cfg["optimizer"])
    opt_pred = Optimizer(net_pred.parameters(), lr=train_cfg["learning_rate"],
                         weight_decay=train_cfg["weight_decay"])
    opt_graph = Optimizer(net_graph.parameters(), lr=train_cfg["learning_rate"])
    sc = dict(train_cfg.get("lr_scheduler") or {})
    sched_pred = None
    if sc:
        sched_pred = getattr(torch.optim.lr_scheduler, sc.pop("name"))(opt_pred, **sc)

    os.makedirs(args.out_dir, exist_ok=True)
    model_save_path = os.path.join(args.out_dir, f"{net_name}.pkl")
    t0 = time.time()
    net_trainer = trainer.AdapGLTrainer(
        net_cfg["adj_mx_path"], net_pred, net_graph, opt_pred, opt_graph, sched_pred, None,
        args.num_epoch, args.num_iter, args.max_graph_num, data_scaler, model_save_path)
    if args.epsilon is not None:            # same epsilon for the weighted union, eq. (9)
        net_trainer.epsilon = args.epsilon
    if args.delta is not None:              # paper eq. (7): delta in (0,1), original code 0.05
        net_trainer.model_graph_trainer.delta = args.delta
    net_trainer.train(loaders[0], loaders[1])
    net_trainer.test(loaders[2])                    # original per-horizon printout
    train_time = time.time() - t0

    # ------------------- evaluation of the best checkpoint -------------------
    best_dir = os.path.join(args.out_dir, net_name)
    net_pred.load_state_dict(torch.load(os.path.join(best_dir, f"{net_name}.pkl"), map_location=device))
    best_adj = np.load(os.path.join(best_dir, "best_adj_mx.npy"))
    adj_t = torch.as_tensor(best_adj, dtype=torch.float32, device=device)

    results, preds = {}, {}
    for split, loader in (("valid", loaders[1]), ("test", loaders[2])):
        p = predict(net_pred, adj_t, loader.dataset.data["x"], device)
        p = data_scaler.inverse_transform(p, axis=0)
        y = raw[split]["y"]
        persist = np.repeat(raw[split]["x"][:, -1:, :, 0], y.shape[1], axis=1)
        results[split] = {"DeepScaler(AdapGLA)": scaling_report(p, y),
                          "Persistence(last pods)": scaling_report(persist, y)}
        preds[split] = p
        preds[f"{split}_true"] = y
    np.savez_compressed(os.path.join(args.out_dir, "predictions.npz"), **preds)

    # graph diagnostics: compare learned graph with the call graph
    od = (np.load(os.path.join(data_dir, "adj_od.npy")) > 0)
    od_sym = od | od.T
    learned = best_adj.copy()
    np.fill_diagonal(learned, 0)
    learned_e = learned > 1e-6
    inter = (learned_e & od_sym).sum()
    union = (learned_e | od_sym).sum()
    graph_info = {"learned_edges(off-diag)": int(learned_e.sum()),
                  "callgraph_edges(sym)": int(od_sym.sum()),
                  "jaccard_vs_callgraph": float(inter / union) if union else 0.0,
                  "sparsity": float((best_adj == 0).mean()),
                  "epsilon": float(net_graph.epsilon), "delta": float(net_trainer.model_graph_trainer.delta)}
    try:                                    # Fig. 11 metric (authors' similarity.py)
        graph_info["paper_similarity_od"] = paper_similarity(best_adj, np.load(os.path.join(data_dir, "adj_od.npy")))
        graph_info["paper_similarity_cc"] = paper_similarity(best_adj, np.load(os.path.join(data_dir, "adj_cc.npy")))
    except ImportError:
        pass

    with open(os.path.join(args.out_dir, "scaler.json"), "w") as f:
        json.dump({"type": train_cfg["scaler"], "mean": np.asarray(data_scaler.mean).tolist(),
                   "std": np.asarray(data_scaler.std).tolist()}, f, indent=2)
    summary = {"model": net_name, "device": str(device), "train_time_s": round(train_time, 1),
               "args": vars(args), "meta": meta, "graph": graph_info, "results": results}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump({"model": net_cfg, "train": train_cfg}, f)

    print("\n================ TEST (best checkpoint) ================")
    for name, r in results["test"].items():
        print(f"{name:24s} MAE {r['mae']:.3f}  RMSE {r['rmse']:.3f}  MAPE {r['mape']*100:.1f}%  "
              f"exact {r['pods_exact_match']*100:.1f}%  under {r['under_provision_rate']*100:.1f}%  "
              f"over {r['over_provision_rate']*100:.1f}%")
    print("graph:", graph_info)
    print("artifacts in", args.out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=None, help="dataset dir from pipeline/build_dataset.py")
    p.add_argument("--model_config_path", default="config/train_dataset_speed.yaml")
    p.add_argument("--train_config_path", default="config/train_config.yaml")
    p.add_argument("--model_name", default="AdapGLA")
    p.add_argument("--num_epoch", type=int, default=15, help="max epochs per alternating round")
    p.add_argument("--num_iter", type=int, default=6, help="alternating (predictor, graph) rounds")
    p.add_argument("--max_graph_num", type=int, default=3, help="size of the adjacency set")
    p.add_argument("--adj", default="od,cc", help="initial graphs: od (call graph), cc (correlation)")
    p.add_argument("--out_dir", default="model_states/boutique_sim")
    p.add_argument("--no_shuffle", dest="shuffle", action="store_false")
    p.add_argument("--epsilon", type=float, default=None,
                   help="edge threshold of eq. (6); default = original code (0.5/num_nodes)")
    p.add_argument("--delta", type=float, default=None,
                   help="new-edge ratio of eq. (7); default = original code (0.05)")
    main(p.parse_args())
