"""Evidence of the uncertainty problem in DeepScaler's scaling decisions.

Thesis problem 2 (claude/thesis-problem-objectives.md): DeepScaler turns a point forecast
into a replica count with a fixed rounding rule, so the risk of under-provisioning is
neither chosen nor controlled.

Why a separate ground truth is needed: DeepScaler's label is the observed replica count of
the next step (paper Sec. III, y_t^i = x_{t+1}^{c,i}). Online, that count is DeepScaler's own
decision, so "pods" cannot tell whether a decision was enough. We use the demand label of
objective O1 instead:  demand_i(t) = max(1, ceil(req_i(t) / mu_i)),  mu_i = req/s one pod of
service i can serve at the CPU target (estimated by --mu_csv or given by --mu).

Evidence computed (one row per service + overall):
  E0 feedback   : how often the decision equals the current replica count (persistence-like)
  E1 shortfall  : P(r_t < demand_{t+1}) per service, mean missing pods when short
  E2 residuals  : e = demand_{t+1} - y_hat_t; margin needed for 1-alpha coverage, overall and
                  in rolling windows (does the needed margin change over time?) vs the fixed
                  rounding rule (ceil unless fraction < 0.35, predict_scale.py)
  E3 shift      : --compare <other summary.json>: offline vs online residual quantiles/shortfall
  E4 SLA        : P(frontend latency > SLO at t+1 | some service short) vs | none short
  E5 bursts     : shortfall and |e| on steps where total load rises fast vs other steps

Online (EC2, after deploy/04_run_deepscaler.sh):
  python -m pipeline.uncertainty_report --csv data/raw/online_ds.csv \
      --decisions Logs/autoscaler.jsonl --mu reports/mu.json --label online --out_dir reports/online
Capacity profiling -> mu (ramp trace, fixed replicas, see deploy/RUN_ONLINE.md):
  python -m pipeline.uncertainty_report --csv data/raw/profile.csv --mu_basis pod --mu_only \
      --out_dir reports
HPA run (no forecast: decision = replicas that actually ran):
  python -m pipeline.uncertainty_report --csv data/raw/online_hpa.csv --mu reports/mu.json \
      --label hpa --out_dir reports/online_hpa
Offline (test split of a trained model):
  python -m pipeline.uncertainty_report --csv data/raw/run2_hpa_5f.csv \
      --pred model_states/ds_s2021/predictions.npz --meta data/c20/meta.json --split test \
      --mu_csv data/raw/run2_hpa_5f.csv --label offline --out_dir reports/offline
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from pipeline.common import SERVICES

RULE = 0.35                     # predict_scale.py: ceil(pred) unless frac(pred) < 0.35


def scale_to_pods(pred, lo=1, hi=10):
    frac = pred - np.floor(pred)
    return np.clip(np.where(frac < RULE, np.floor(pred), np.ceil(pred)), lo, hi)


# ----------------------------------------------------------------------------- capacity mu
def cpu_resources(yaml_path):
    """{svc: (cpu_request_cores, cpu_limit_cores)} of the app container, from boutique.yaml."""
    import yaml

    def cores(v):
        v = str(v)
        return float(v[:-1]) / 1000 if v.endswith("m") else float(v)

    out = {}
    for d in yaml.safe_load_all(open(yaml_path)):
        if d and d.get("kind") == "Deployment" and d["metadata"]["name"] in SERVICES:
            c = d["spec"]["template"]["spec"]["containers"][0]["resources"]
            out[d["metadata"]["name"]] = (cores(c["requests"]["cpu"]), cores(c["limits"]["cpu"]))
    return out


def estimate_mu(df, res, basis="app", target=0.5, sidecar_req=0.1):
    """Linear fit cpu_per_pod = a + b * (req/s per pod); mu = req/s per pod at target utilisation.

    basis="app": app container only (cpu column = usage/limit), target = target * app request.
    basis="pod": all containers incl. istio-proxy (cpupod column, as HPA sees it),
                 target = target * (app request + sidecar request).
    """
    rows = {}
    for s in SERVICES:
        req_c, lim_c = res[s]
        pods = df[f"{s}_pod"].clip(lower=1)
        x = df[f"{s}_req"] / pods
        if basis == "pod":
            if f"{s}_cpupod" not in df:
                raise ValueError("basis=pod needs the cpupod column (collect.py >= this version)")
            u, tgt = df[f"{s}_cpupod"] / pods, target * (req_c + sidecar_req)
        else:
            u, tgt = df[f"{s}_cpu"] * lim_c, target * req_c
        m = (df[f"{s}_req"] > 0) & np.isfinite(u)
        A = np.vstack([np.ones(m.sum()), x[m]]).T
        (a, b), *_ = np.linalg.lstsq(A, u[m], rcond=None)
        fit = A @ np.array([a, b])
        r2 = 1 - ((u[m] - fit) ** 2).sum() / max(((u[m] - u[m].mean()) ** 2).sum(), 1e-12)
        mu = (tgt - a) / b if b > 0 else float("nan")
        rows[s] = {"mu": float(mu), "a": float(a), "b": float(b), "r2": float(r2),
                   "x_max": float(x[m].max()), "extrapolated": bool(mu > x[m].max()),
                   "basis": basis, "target": target}
    return rows


# ----------------------------------------------------------------------------- alignment
def steps_online(df, jsonl, step):
    """Decision made at time t (autoscaler.jsonl 'time') applies until t+step -> demand at t+step."""
    recs = [json.loads(l) for l in open(jsonl) if l.strip()]
    idx = {int(round(t)): i for i, t in enumerate(df["unix_time"])}
    T, Y, R, C = [], [], [], []
    for r in recs:
        i = idx.get(int(round(r["time"] + step)))
        if i is not None:
            T.append(i); Y.append(r["pred"]); R.append(r["pods"]); C.append(r["current"])
    return np.array(T), np.array(Y, float), np.array(R, float), np.array(C, float)


def steps_offline(df, pred_npz, meta_json, split):
    meta = json.load(open(meta_json))
    d = np.load(pred_npz)
    s_in, h, n = meta["step_in"], meta["horizon"], meta["samples"]
    T = len(df)
    test_start = T - (n["test"] + s_in + h - 1)
    start = test_start if split == "test" else test_start - (n["valid"] + s_in + h - 1)
    y_hat = d[split][:, 0, :]
    rows = start + s_in + np.arange(len(y_hat))                      # index of the predicted step
    pods = df[[f"{s}_pod" for s in SERVICES]].to_numpy()
    assert np.allclose(pods[rows], d[f"{split}_true"][:, 0, :]), "split alignment mismatch"
    return rows, y_hat, scale_to_pods(y_hat), pods[rows - 1]


def steps_observed(df):
    """Any autoscaler (e.g. HPA): no forecast, the decision is the replica count that ran at t+1."""
    pods = df[[f"{s}_pod" for s in SERVICES]].to_numpy()
    rows = np.arange(1, len(df))
    return rows, pods[rows].astype(float), pods[rows].astype(float), pods[rows - 1]


# ----------------------------------------------------------------------------- evidence
def analyse(df, rows, y_hat, r, cur, mu, step, alphas, window_min, slo, burst_pct):
    req = df[[f"{s}_req" for s in SERVICES]].to_numpy()[rows]
    mu_v = np.array([mu[s] for s in SERVICES])
    demand = np.maximum(1, np.ceil(req / mu_v))
    e = demand - y_hat
    short = r < demand
    win = max(2, int(window_min * 60 / step))

    lat_col = "frontend_p95" if "frontend_p95" in df else "frontend_res"
    lat = df[lat_col].to_numpy()[rows]
    viol = lat > slo
    any_short = short.any(axis=1)
    tot = req.sum(axis=1)
    prev = np.r_[tot[:2], tot[:-2]]                                   # load 2 steps (1 min) earlier
    burst = (tot - prev) / np.maximum(prev, 1e-9) > burst_pct

    per = []
    for j, s in enumerate(SERVICES):
        ej = e[:, j]
        row = {"service": s, "mu": mu[s], "steps": int(len(ej)),
               "E0_same_as_current": float((r[:, j] == cur[:, j]).mean()),
               "E1_shortfall": float(short[:, j].mean()),
               "E1_missing_pods_when_short": float((demand[:, j] - r[:, j])[short[:, j]].mean())
               if short[:, j].any() else 0.0,
               "E1_overprov": float((r[:, j] > demand[:, j]).mean()),
               "demand_max": int(demand[:, j].max()), "demand_gt_10": float((demand[:, j] > 10).mean()),
               "E2_e_mean": float(ej.mean()), "E2_e_std": float(ej.std())}
        for a in alphas:
            q = float(np.quantile(ej, 1 - a))
            rq = [np.quantile(ej[k:k + win], 1 - a) for k in range(0, len(ej) - win + 1, max(1, win // 2))]
            row[f"E2_margin_needed_a{a}"] = q
            row[f"E2_margin_window_min_a{a}"] = float(np.min(rq)) if rq else q
            row[f"E2_margin_window_max_a{a}"] = float(np.max(rq)) if rq else q
        row["E5_shortfall_burst"] = float(short[burst, j].mean()) if burst.any() else float("nan")
        row["E5_shortfall_normal"] = float(short[~burst, j].mean()) if (~burst).any() else float("nan")
        row["E5_abs_e_burst"] = float(np.abs(ej[burst]).mean()) if burst.any() else float("nan")
        row["E5_abs_e_normal"] = float(np.abs(ej[~burst]).mean()) if (~burst).any() else float("nan")
        per.append(row)

    overall = {
        "steps": int(len(rows)), "step_s": step, "latency_column": lat_col, "slo_s": slo,
        "E0_same_as_current": float((r == cur).mean()),
        "E1_shortfall_all": float(short.mean()),
        "E1_shortfall_spread(max-min over services)": float(short.mean(0).max() - short.mean(0).min()),
        "E1_steps_with_any_short": float(any_short.mean()),
        "E4_P(viol | some short)": float(viol[any_short].mean()) if any_short.any() else float("nan"),
        "E4_P(viol | none short)": float(viol[~any_short].mean()) if (~any_short).any() else float("nan"),
        "E4_violation_rate": float(viol.mean()),
        "E5_burst_steps": int(burst.sum()), "E5_burst_threshold": burst_pct,
        "rule": f"ceil(y_hat) unless frac < {RULE}: r - y_hat in [-{RULE}, {1 - RULE})",
    }
    series = {"rows": rows, "demand": demand, "y_hat": y_hat, "r": r, "e": e, "short": short,
              "lat": lat, "viol": viol, "burst": burst, "tot": tot}
    return overall, pd.DataFrame(per), series


def plots(series, df, step, alphas, win_min, slo, out_dir, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_min = (df["unix_time"].to_numpy()[series["rows"]] - df["unix_time"].iloc[0]) / 60
    e, short = series["e"], series["short"]
    win = max(2, int(win_min * 60 / step))
    a = max(alphas)

    fig, axes = plt.subplots(5, 2, figsize=(13, 13), sharex=True)
    for j, (s, ax) in enumerate(zip(SERVICES, axes.ravel())):
        ax.plot(t_min, e[:, j], lw=0.8, color="#4C72B0", label="demand - y_hat")
        q = pd.Series(e[:, j]).rolling(win, min_periods=win // 2).quantile(1 - a)
        ax.plot(t_min, q, lw=1.5, color="#C44E52", label=f"rolling q{1 - a:.2f}")
        ax.axhline(1 - RULE, ls="--", color="gray", lw=1, label="max margin of rounding rule")
        ax.scatter(t_min[short[:, j]], e[short[:, j], j], s=8, color="#C44E52", zorder=3)
        ax.set_title(f"{s}  short {short[:, j].mean() * 100:.0f}%", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("minutes")
    fig.suptitle(f"[{label}] residual of the demand label; red dots = under-provisioned steps")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "E1_E2_residuals.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(t_min, series["tot"], color="#4C72B0")
    ax[0].scatter(t_min[series["burst"]], series["tot"][series["burst"]], s=10, color="#DD8452",
                  label="burst step")
    ax[0].set_ylabel("total req/s"); ax[0].legend(fontsize=8)
    ax[1].plot(t_min, short.sum(axis=1), drawstyle="steps-mid", color="#C44E52")
    ax[1].set_ylabel("# services short")
    ax[2].plot(t_min, series["lat"] * 1000, color="#55A868")
    ax[2].axhline(slo * 1000, ls="--", color="gray", label="SLO")
    ax[2].set_ylabel("frontend latency (ms)"); ax[2].set_xlabel("minutes"); ax[2].legend(fontsize=8)
    for x in ax:
        x.grid(alpha=0.3)
    fig.suptitle(f"[{label}] load, under-provisioning and latency")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "E4_E5_timeline.png"), dpi=120)
    plt.close(fig)


def compare(cur_dir, other_summary, out_dir):
    a = json.load(open(os.path.join(cur_dir, "summary.json")))
    b = json.load(open(other_summary))
    keys = [k for k in a["per_service"][0] if k.startswith(("E1_shortfall", "E2_margin_needed", "E2_e_"))]
    rows = []
    for ra, rb in zip(a["per_service"], b["per_service"]):
        row = {"service": ra["service"]}
        for k in keys:
            row[f"{a['label']}:{k}"], row[f"{b['label']}:{k}"] = ra[k], rb[k]
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, f"E3_compare_{b['label']}.csv"), index=False)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="metrics CSV from pipeline/collect.py")
    p.add_argument("--decisions", help="online: Logs/autoscaler.jsonl")
    p.add_argument("--pred", help="offline: predictions.npz from train.py")
    p.add_argument("--meta", help="offline: meta.json of the dataset used for training")
    p.add_argument("--split", default="test", choices=["valid", "test"])
    p.add_argument("--mu", help="json {service: req/s per pod} (e.g. from profiling)")
    p.add_argument("--mu_csv", help="estimate mu from this CSV (profiling or an earlier run)")
    p.add_argument("--mu_basis", default="app", choices=["app", "pod"])
    p.add_argument("--cpu_target", type=float, default=0.5, help="utilisation target (HPA: 50%%)")
    p.add_argument("--sidecar_req", type=float, default=0.1, help="istio-proxy CPU request (cores)")
    p.add_argument("--boutique_yaml", default="benchmarks/boutique/boutique.yaml")
    p.add_argument("--slo", type=float, default=0.2, help="latency SLO in seconds")
    p.add_argument("--alphas", default="0.05,0.1")
    p.add_argument("--window_min", type=float, default=20, help="rolling window for E2 (minutes)")
    p.add_argument("--burst_pct", type=float, default=0.3, help="E5: load rise over 1 min")
    p.add_argument("--compare", help="summary.json of another run (E3)")
    p.add_argument("--mu_only", action="store_true", help="only estimate mu (from --csv) and exit")
    p.add_argument("--label", default="run")
    p.add_argument("--out_dir", default="reports/run")
    a = p.parse_args()

    df = pd.read_csv(a.csv)
    step = float(np.median(np.diff(df["unix_time"])))
    alphas = [float(x) for x in a.alphas.split(",")]
    os.makedirs(a.out_dir, exist_ok=True)

    if a.mu:
        raw = json.load(open(a.mu))
        mu_info = {s: (raw[s] if isinstance(raw[s], dict) else {"mu": float(raw[s])}) for s in SERVICES}
    else:
        src = pd.read_csv(a.mu_csv or a.csv)
        mu_info = estimate_mu(src, cpu_resources(a.boutique_yaml), a.mu_basis, a.cpu_target, a.sidecar_req)
    json.dump(mu_info, open(os.path.join(a.out_dir, "mu.json"), "w"), indent=2)
    mu = {s: mu_info[s]["mu"] for s in SERVICES}
    if a.mu_only:
        print(pd.DataFrame(mu_info).T.round(4).to_string())
        print("written", os.path.join(a.out_dir, "mu.json"))
        return

    if a.decisions:
        rows, y_hat, r, cur = steps_online(df, a.decisions, step)
        mode = "online"
    elif not a.pred:
        rows, y_hat, r, cur = steps_observed(df)          # e = demand - replicas actually run
        mode = "observed"
    else:
        rows, y_hat, r, cur = steps_offline(df, a.pred, a.meta, a.split)
        mode = f"offline-{a.split}"
    if len(rows) == 0:
        raise SystemExit("no decision could be aligned with the CSV time grid")

    overall, per, series = analyse(df, rows, y_hat, r, cur, mu, step, alphas,
                                   a.window_min, a.slo, a.burst_pct)
    summary = {"label": a.label, "mode": mode, "csv": a.csv, "overall": overall,
               "per_service": per.to_dict(orient="records")}
    json.dump(summary, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=2)
    per.to_csv(os.path.join(a.out_dir, "per_service.csv"), index=False)
    plots(series, df, step, alphas, a.window_min, a.slo, a.out_dir, a.label)

    pd.set_option("display.width", 200)
    print(json.dumps(overall, indent=1))
    cols = ["service", "mu", "E0_same_as_current", "E1_shortfall", "E1_overprov",
            f"E2_margin_needed_a{max(alphas)}", f"E2_margin_window_min_a{max(alphas)}",
            f"E2_margin_window_max_a{max(alphas)}", "E5_shortfall_burst", "E5_shortfall_normal"]
    print(per[cols].round(3).to_string(index=False))
    if a.compare:
        print(compare(a.out_dir, a.compare, a.out_dir).round(3).to_string(index=False))
    print("written to", a.out_dir)


if __name__ == "__main__":
    main()
