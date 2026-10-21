"""Synthetic Online-Boutique metrics (SIMULATED, not measured).

Use this only when no Kubernetes cluster is available: it lets the whole
DeepScaler training pipeline run end to end (e.g. on Colab/Kaggle) with data in
exactly the same format that `pipeline/collect.py` produces from Prometheus.
Results trained on this data must be reported as "simulated" in the thesis.

Model of the system
-------------------
* Load: the FIFA-98-derived user trace shipped with the repo
  (sendFlow/random-100max.req, one value per 5 s, 25..100 users), replayed in
  loops with a random amplitude per loop, multiplicative noise and occasional
  bursts. Each Locust user runs one task, waits 1 s, repeats.
* Task mix and per-task downstream calls follow sendFlow/load_generator.py and
  the Online-Boutique call graph (pipeline/common.py).
* CPU demand  = sum(calls/s x CPU-seconds per call) + idle cost per pod.
* Replicas are decided by a Kubernetes-HPA-like controller (target 50 % of the
  CPU limit, 30 s readiness delay, 5 min scale-down stabilisation), i.e. the
  same kind of controller that is running while real training data is collected.
* Response time per service follows an M/M/1-style inflation 1/(1-rho);
  callers add the latency of their callees.

Output: CSV with columns  time, <svc>_<feature>  for feature in pod,cpu,res,req,
sampled every `--interval` seconds.
"""
import argparse
import os

import numpy as np
import pandas as pd

from pipeline.common import SERVICES, call_graph_adj

TICK = 5  # seconds per trace value (same as load_generator.py)

# Locust tasks: weight, frontend HTTP requests, downstream gRPC calls per task.
TASKS = {
    #             weight  fe   ad  cart chk  cur  email pay  pcat rec  ship
    "index":       (1,   1,   1,  1,   0,   10,  0,    0,   1,   0,   0),
    "setCurrency": (2,   1,   0,  0,   0,   0,   0,    0,   0,   0,   0),
    "browse":      (10,  1,   1,  1,   0,   2,   0,    0,   3,   1,   0),
    "viewCart":    (2,   1,   0,  1,   0,   3,   0,    0,   2,   1,   1),
    "addToCart":   (3,   2,   1,  2,   0,   2,   0,    0,   4,   1,   0),
    "checkout":    (1,   3,   1,  4,   1,   6,   1,    1,   6,   1,   2),
}
COL = {"frontend": 1, "adservice": 2, "cartservice": 3, "checkoutservice": 4,
       "currencyservice": 5, "emailservice": 6, "paymentservice": 7,
       "productcatalogservice": 8, "recommendationservice": 9, "shippingservice": 10}

# CPU-seconds per request, base latency (s), CPU limit per pod (cores; boutique.yaml).
CPU_COST = {"frontend": 0.0040, "adservice": 0.0030, "cartservice": 0.0020,
            "checkoutservice": 0.0030, "currencyservice": 0.0005, "emailservice": 0.0030,
            "paymentservice": 0.0015, "productcatalogservice": 0.0006,
            "recommendationservice": 0.0040, "shippingservice": 0.0008}
BASE_RT = {"frontend": 0.010, "adservice": 0.004, "cartservice": 0.003,
           "checkoutservice": 0.008, "currencyservice": 0.001, "emailservice": 0.003,
           "paymentservice": 0.002, "productcatalogservice": 0.001,
           "recommendationservice": 0.004, "shippingservice": 0.001}
CPU_LIMIT = 0.2
IDLE_CPU = 0.004
HPA_TARGET = 0.5
MAX_PODS = 10
READY_DELAY = 30
DOWN_WINDOW = 300


def per_service_calls_per_task(mix=None):
    """Expected requests per user-task for each service (dict svc -> float).

    `mix` multiplies the Locust task weights (user behaviour drift)."""
    w = np.array([t[0] for t in TASKS.values()], dtype=float)
    if mix is not None:
        w = w * mix
    w /= w.sum()
    out = {}
    for svc, c in COL.items():
        out[svc] = float(sum(wi * t[c] for wi, t in zip(w, TASKS.values())))
    return out


def build_user_series(trace, loops, rng):
    parts = []
    for _ in range(loops):
        amp = rng.uniform(0.7, 1.3)
        u = trace * amp * rng.lognormal(0, 0.05, size=trace.size)
        if rng.random() < 0.6:  # one burst per loop with prob. 0.6
            start = rng.integers(0, trace.size - 60)
            length = rng.integers(12, 60)          # 1..5 minutes
            u[start:start + length] *= rng.uniform(1.5, 2.5)
        parts.append(u)
    return np.clip(np.concatenate(parts), 1, None)


def simulate(trace, loops=16, interval=30, seed=2021):
    rng = np.random.default_rng(seed)
    users = build_user_series(trace, loops, rng)
    n_ticks = users.size
    # slow AR(1) drift of the task mix (e.g. more checkouts during peaks), so that
    # services are not all perfectly proportional to the user count
    log_mix = np.zeros(len(TASKS))
    adj = call_graph_adj()
    idx = {s: i for i, s in enumerate(SERVICES)}
    n = len(SERVICES)

    pods = np.ones(n)
    pending = []                      # (ready_time, svc_idx, delta)
    ready = np.ones(n)
    desired_hist = []                 # (time, desired vector) for scale-down window
    rt_front = 0.05
    rows = []
    acc = None
    for k in range(n_ticks):
        t = k * TICK
        # pods becoming ready
        for item in [p for p in pending if p[0] <= t]:
            ready[item[1]] += item[2]
            pending.remove(item)

        log_mix = 0.995 * log_mix + rng.normal(0, 0.04, size=log_mix.size)
        calls = per_service_calls_per_task(np.exp(log_mix))
        task_rate = users[k] / (1.0 + rt_front)            # tasks / s
        req = np.array([task_rate * calls[s] for s in SERVICES])
        req *= rng.lognormal(0, 0.06, size=n)
        demand = np.array([req[idx[s]] * CPU_COST[s] for s in SERVICES]) + IDLE_CPU * ready
        cap = ready * CPU_LIMIT
        rho = demand / cap
        usage = np.minimum(demand, cap)
        own_rt = np.array([BASE_RT[s] for s in SERVICES]) / (1 - np.minimum(rho, 0.95))
        own_rt *= np.where(rho > 1, rho ** 2, 1.0)        # CPU throttling penalty
        # end-to-end latency per service = own + latency of callees (graph is a DAG)
        rt = own_rt.copy()
        for s in ["recommendationservice", "checkoutservice", "frontend"]:
            i = idx[s]
            rt[i] = own_rt[i] + 0.5 * (adj[i] * rt).sum()
        rt *= rng.lognormal(0, 0.05, size=n)
        rt_front = rt[idx["frontend"]]

        # HPA every 15 s
        if t % 15 == 0:
            util = usage / (ready * CPU_LIMIT)
            desired = np.clip(np.ceil(ready * util / HPA_TARGET), 1, MAX_PODS)
            desired_hist.append((t, desired))
            desired_hist = [d for d in desired_hist if d[0] > t - DOWN_WINDOW]
            window_max = np.max([d[1] for d in desired_hist], axis=0)
            target = np.where(desired > pods, desired, np.minimum(pods, window_max))
            delta = target - pods
            for i in np.nonzero(delta)[0]:
                if delta[i] > 0:
                    pending.append((t + READY_DELAY, i, delta[i]))
                else:
                    ready[i] += delta[i]      # scale-down is immediate
            pods = target

        sample = np.stack([pods, usage / (pods * CPU_LIMIT), rt, req], axis=1)  # (N, F)
        acc = sample if acc is None else acc + sample
        if (k + 1) % (interval // TICK) == 0:
            avg = acc / (interval // TICK)
            avg[:, 0] = pods                  # pod count is a snapshot, not an average
            rows.append(avg)
            acc = None
    data = np.stack(rows)                      # (T, N, F)
    return data


def to_frame(data, interval):
    cols = {"time": np.arange(data.shape[0]) * interval}
    feats = ["pod", "cpu", "res", "req"]
    for j, s in enumerate(SERVICES):
        for f, name in enumerate(feats):
            cols[f"{s}_{name}"] = data[:, j, f]
    return pd.DataFrame(cols)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", default="sendFlow/random-100max.req")
    p.add_argument("--loops", type=int, default=16, help="trace replays (75 min each)")
    p.add_argument("--interval", type=int, default=30, help="sampling interval (s), multiple of 5")
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--out", default="data/raw/simulated.csv")
    a = p.parse_args()
    trace = np.loadtxt(a.trace)
    data = simulate(trace, a.loops, a.interval, a.seed)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df = to_frame(data, a.interval)
    df.to_csv(a.out, index=False)
    print(f"[SIMULATED] wrote {a.out}: {data.shape[0]} steps x {data.shape[1]} services x {data.shape[2]} features")
    print(df[[c for c in df.columns if c.endswith('_pod')]].describe().loc[['mean', 'max']].round(2).to_string())


if __name__ == "__main__":
    main()
