"""Baselines of DeepScaler's Fig. 12 on the same dataset as train.py.

DeepScaler paper, Sec. V-C "Varying Different Applications": DeepScaler is compared with
two GNN baselines, (i) DCRNN [49] and (ii) T-GCN [50], which "only use predefined affinity
matrices (i.e., invocation relationships)", with MAE / RMSE / MAPE (Sec. V-A).

  * DCRNN : the authors' implementation models/DCRNN.py (bidirectional diffusion conv,
            K=3, 128 hidden, 1 layer, curriculum learning 2000 = AdapGLD block of
            config/train_dataset_speed.yaml), fixed call graph adj_od.npy.
  * T-GCN : not in the authors' repo -> standard T-GCN of Zhao et al. (GCN inside a GRU
            cell, A_hat = D^-1/2 (A+I) D^-1/2, hidden 64), fixed call graph adj_od.npy.
  * LSTM  : the authors' models/LSTM.py (no graph). NOT in Fig. 12, extra reference only.

Same data, split, StandardScaler, L1 loss, Adam (config/train_config.yaml) and report
(scaling_report of train.py) as train.py, so metrics.json is directly comparable.

  python train_baselines.py --data_dir data/c20 --model DCRNN --out_dir model_states/bl_DCRNN_s2021
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import TPDataset
from models.DCRNN import DCRNN
from models.LSTM import LSTM
from train import load_yaml, scaling_report
from utils import scaler as scaler_mod


class TGCNCell(torch.nn.Module):
    """T-GCN cell (Zhao et al., IEEE T-ITS 2019, eq. 3-6): GRU whose gates use a 1-layer GCN."""

    def __init__(self, adj, in_size, hidden):
        super().__init__()
        self.register_buffer("adj", adj)
        self.hidden = hidden
        self.gate = torch.nn.Linear(in_size + hidden, 2 * hidden)
        self.cand = torch.nn.Linear(in_size + hidden, hidden)

    def gc(self, x):                       # A_hat X  (B, N, F)
        return torch.matmul(self.adj, x)

    def forward(self, x, h):
        ru = torch.sigmoid(self.gate(self.gc(torch.cat([x, h], -1))))
        r, u = ru.chunk(2, -1)
        c = torch.tanh(self.cand(self.gc(torch.cat([x, r * h], -1))))
        return u * h + (1 - u) * c


class TGCN(torch.nn.Module):
    def __init__(self, adj, input_size, step_num_out, hidden=64):
        super().__init__()
        self.cell = TGCNCell(adj, input_size, hidden)
        self.out = torch.nn.Linear(hidden, step_num_out)

    def forward(self, x):                  # x (B, T, N, F) -> (B, step_out, N)
        h = x.new_zeros(x.size(0), x.size(2), self.cell.hidden)
        for t in range(x.size(1)):
            h = self.cell(x[:, t], h)
        return self.out(h).transpose(1, 2)


def build(name, meta, data_dir, device):
    n, f, out = len(meta["services"]), len(meta["features"]), meta["horizon"]
    od_path = os.path.join(data_dir, "adj_od.npy")
    if name == "DCRNN":
        # hyper-parameters of the AdapGLD (DCRNN-based) block in config/train_dataset_speed.yaml
        return DCRNN(input_size=f, rnn_hidden_size=128, rnn_layer_num=1, use_gc=True, K=3,
                     graph_num=2, step_num_out=out, batch_first=True, use_curriculum_learning=True,
                     cl_decay_steps=2000, activation="tanh", adj_path=od_path,
                     adj_type="random_walk", device=str(device)).to(device)
    if name == "TGCN":
        adj = np.load(od_path).astype(np.float64)
        adj = np.maximum(adj, adj.T)       # T-GCN uses an undirected graph
        adj = adj + np.eye(len(adj))      # A_hat = D~^-1/2 (A + I) D~^-1/2 (T-GCN paper, eq. 2)
        d = adj.sum(1) ** -0.5
        a = d[:, None] * adj * d[None, :]
        return TGCN(torch.as_tensor(a, dtype=torch.float32), f, out).to(device)
    if name == "LSTM":
        return LSTM(input_size=f, num_nodes=n, step_num_out=out, hidden_lstm=128).to(device)
    raise ValueError(name)


def forward(model, name, x, y=None, step=0):
    if name == "DCRNN":
        # labels for curriculum learning: (B, N) when step_num_out == 1 (see GeneralDCRNN.decoder)
        lab = None if y is None else (y[:, 0] if y.size(1) == 1 else y)
        return model(x, labels=lab, batches_seen=step)
    return model(x)


def main(a):
    tcfg = load_yaml(a.train_config_path)
    seed = a.seed if a.seed is not None else tcfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(a.data_dir, "meta.json")) as f:
        meta = json.load(f)

    sc = getattr(scaler_mod, tcfg["scaler"])(axis=(0, 1, 2))
    raw, loaders = {}, []
    for split in ("train", "valid", "test"):
        ds = TPDataset(os.path.join(a.data_dir, f"{split}.npz"))
        raw[split] = {k: v.copy() for k, v in ds.data.items()}
        if split == "train":
            sc.fit(ds.data["x"])
            sc.std = np.where(sc.std == 0, 1.0, sc.std)
        ds.fit(sc)
        loaders.append(DataLoader(ds, batch_size=a.batch_size, shuffle=(split == "train")))

    model = build(a.model, meta, a.data_dir, device)
    opt = getattr(torch.optim, tcfg["optimizer"])(model.parameters(), lr=tcfg["learning_rate"],
                                                  weight_decay=tcfg["weight_decay"])
    os.makedirs(a.out_dir, exist_ok=True)
    ckpt = os.path.join(a.out_dir, f"{a.model}.pkl")
    best, bad, step, t0 = float("inf"), 0, 0, time.time()
    for ep in range(1, a.max_epoch + 1):
        model.train()
        for x, y in loaders[0]:
            x, y = x.float().to(device), y.float().to(device)
            loss = F.l1_loss(forward(model, a.model, x, y, step), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
        model.eval()
        with torch.no_grad():
            vl = np.mean([F.l1_loss(forward(model, a.model, x.float().to(device)), y.float().to(device)).item()
                          for x, y in loaders[1]])
        if vl < best:
            best, bad = vl, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
        print(f"epoch {ep} valid_l1 {vl:.4f}")
        if bad >= a.patience:
            break
    train_time = time.time() - t0

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    results, preds = {}, {}
    for split, loader in (("valid", loaders[1]), ("test", loaders[2])):
        xs = loader.dataset.data["x"]
        with torch.no_grad():
            p = np.concatenate([forward(model, a.model, torch.as_tensor(xs[i:i + 256], dtype=torch.float32,
                                                                        device=device)).cpu().numpy()
                                for i in range(0, len(xs), 256)])
        p = sc.inverse_transform(p, axis=0)
        y = raw[split]["y"]
        persist = np.repeat(raw[split]["x"][:, -1:, :, 0], y.shape[1], axis=1)
        results[split] = {a.model: scaling_report(p, y), "Persistence(last pods)": scaling_report(persist, y)}
        preds[split], preds[f"{split}_true"] = p, y
    np.savez_compressed(os.path.join(a.out_dir, "predictions.npz"), **preds)
    with open(os.path.join(a.out_dir, "metrics.json"), "w") as f:
        json.dump({"model": a.model, "seed": seed, "device": str(device), "epochs_run": ep,
                   "train_time_s": round(train_time, 1), "args": vars(a), "meta": meta,
                   "results": results}, f, indent=2)
    r = results["test"][a.model]
    print(f"TEST {a.model}: MAE {r['mae']:.4f} RMSE {r['rmse']:.4f} MAPE {r['mape']*100:.2f}% "
          f"exact {r['pods_exact_match']*100:.1f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--model", choices=["DCRNN", "TGCN", "LSTM"], required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--train_config_path", default="config/train_config.yaml")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_epoch", type=int, default=90, help="= 15 epochs x 6 rounds of train.py")
    p.add_argument("--patience", type=int, default=10, help="early stop (EarlyStop(10) in e2e_trainer.py)")
    main(p.parse_args())
