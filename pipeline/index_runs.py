"""Catalogue of every run under runs/<campaign>/<RUN>/ -> runs/INDEX.csv and runs/INDEX.md.

  python -m pipeline.index_runs          # rerun after every new run (05_online_run.sh / 07_profile_sla.sh)

Per run: campaign, kind (online / profile_sla), policy, start/end, rows, which latency is available
(client = exact per-request log; istio = sidecar p95 with a bucket at 200 ms; istio~ = interpolated),
files present, and the hand-written status / used_in / note from runs/notes.csv.
Campaigns: v1_istio-p95 = before 10/11/2026 (latency interpolated, reference only);
           v2_exact = from 10/11/2026 (Istio bucket at the SLO + Locust per-request log).
Training / profiling inputs that do not depend on the SLA measurement stay in data/raw/ (listed too).
"""
import glob
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def one(rd):
    run, camp = os.path.basename(rd), os.path.basename(os.path.dirname(rd))
    files = os.listdir(rd)
    kind = "profile_sla" if run.startswith("profile_sla_") else "online"
    policy = run.split("_")[2] if kind == "profile_sla" else run.split("_")[1]
    row = {"campaign": camp, "run": run, "kind": kind, "policy": policy, "path": os.path.relpath(rd, ROOT)}
    t = os.path.join(rd, f"{run}_time.txt")
    if os.path.exists(t):
        for line in open(t):
            k, _, v = line.strip().partition(" ")
            row[k.lower()] = v.strip()
    csv = os.path.join(rd, f"{run}.csv")
    row["csv"] = os.path.exists(csv)
    lat = []
    if row["csv"]:
        cols = pd.read_csv(csv, nrows=1).columns
        row["rows"] = sum(1 for _ in open(csv)) - 1
        if "client_p95" in cols:
            src = pd.read_csv(csv, usecols=["client_src"]).iloc[0, 0] if "client_src" in cols else "log"
            lat.append("client" if src == "log" else "client~")
        lat.append("istio" if any(c.endswith("_slow200") for c in cols) else "istio~")
    row["latency"] = "+".join(lat)
    row["autoscaler_log"] = any(f.endswith("_autoscaler.jsonl") for f in files)
    row["latency_log"] = any(f.endswith("_latency.csv") for f in files)
    return row


def main():
    rows = [one(rd) for rd in sorted(glob.glob(os.path.join(ROOT, "runs", "*", "*"))) if os.path.isdir(rd)]
    df = pd.DataFrame(rows)
    notes_p = os.path.join(ROOT, "runs", "notes.csv")
    if os.path.exists(notes_p):
        df = df.merge(pd.read_csv(notes_p), on="run", how="left")
    df = df.fillna({"status": "new", "used_in": "", "note": ""})
    if "rows" in df:
        df["rows"] = df["rows"].astype("Int64").astype(str).replace("<NA>", "")
    cols = [c for c in ["campaign", "run", "kind", "policy", "start", "end", "rows", "latency", "status", "used_in",
                        "note", "csv", "latency_log", "autoscaler_log", "path"] if c in df]
    df = df[cols]
    df.to_csv(os.path.join(ROOT, "runs", "INDEX.csv"), index=False)
    raw = sorted(f for f in os.listdir(os.path.join(ROOT, "data", "raw")) if f.endswith(".csv"))
    with open(os.path.join(ROOT, "runs", "INDEX.md"), "w") as f:
        f.write("# Danh mục các lần chạy (tự sinh bởi `python -m pipeline.index_runs`, đừng sửa tay)\n\n")
        f.write("Sửa trạng thái/ghi chú trong `runs/notes.csv` rồi chạy lại lệnh trên.\n\n")
        f.write("`latency`: client = log từng request (chính xác); istio = p95 sidecar có biên bucket tại 200 ms; "
                "`~` = xấp xỉ (nội suy / Locust history). Campaign `v1_istio-p95` chỉ để tham khảo.\n\n")
        for camp, g in df.groupby("campaign", sort=True):
            f.write(f"## {camp}\n\n| run | policy | start | rows | latency | status | used_in | note |\n|---|---|---|---|---|---|---|---|\n")
            for _, r in g.iterrows():
                f.write(f"| {r['run']} | {r['policy']} | {r.get('start', '') if pd.notna(r.get('start', '')) else ''} | {r.get('rows', '')} | {r['latency']} | "
                        f"{r['status']} | {r['used_in']} | {r['note']} |\n")
            f.write("\n")
        f.write("## data/raw (dữ liệu train / profiling, không phụ thuộc cách đo SLA)\n\n")
        for r in raw:
            f.write(f"- `data/raw/{r}`\n")
    print(f"wrote runs/INDEX.csv and runs/INDEX.md: {len(df)} runs")


if __name__ == "__main__":
    main()
