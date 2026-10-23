"""Load traces (users per TICK, one int per line) for pipeline/locustfile.py (TRACE=...).

  ramp  : capacity profiling for mu (demand label, thesis objective O1). Users go up in
          equal steps and each level is held long enough for Prometheus rate windows
          (default 10 -> 150 users, step 10, 2 min/level at TICK=5 s => 30 min).
  burst : the FIFA trace used for training (sendFlow/random-100max.req) with sudden
          bursts: at each start index the user count is multiplied by --factor for
          --burst_min minutes, then drops back instantly. Same base load as training, so
          only the bursts differ (DeepScaler paper, Sec. V-B: bursts raise SLA violations
          for all prediction-based autoscalers, Fig. 10 min 30-50).

  python -m pipeline.make_traces ramp  --out sendFlow/ramp-10-150.req
  python -m pipeline.make_traces burst --out sendFlow/fifa-burst.req
  TRACE=sendFlow/fifa-burst.req RUN_TIME=4500 locust -f pipeline/locustfile.py --headless ...
"""
import argparse

import numpy as np


def ramp(lo, hi, step, hold_ticks):
    return [u for u in range(lo, hi + 1, step) for _ in range(hold_ticks)]


def burst(base, starts_min, burst_min, factor, tick_s, cap):
    users = np.array(base, dtype=float)
    n = int(burst_min * 60 / tick_s)
    for m in starts_min:
        i = int(m * 60 / tick_s)
        users[i:i + n] *= factor
    return [int(round(min(u, cap))) for u in users]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=["ramp", "burst"])
    p.add_argument("--out", required=True)
    p.add_argument("--tick", type=float, default=5, help="seconds per line (locustfile TICK)")
    p.add_argument("--lo", type=int, default=10)
    p.add_argument("--hi", type=int, default=150)
    p.add_argument("--step", type=int, default=10)
    p.add_argument("--hold_min", type=float, default=2)
    p.add_argument("--base", default="sendFlow/random-100max.req")
    p.add_argument("--starts_min", default="15,40,60", help="burst start times (minutes)")
    p.add_argument("--burst_min", type=float, default=5)
    p.add_argument("--factor", type=float, default=2.0)
    p.add_argument("--cap", type=int, default=200)
    a = p.parse_args()
    if a.kind == "ramp":
        users = ramp(a.lo, a.hi, a.step, int(a.hold_min * 60 / a.tick))
    else:
        base = [int(l) for l in open(a.base) if l.strip()]
        users = burst(base, [float(x) for x in a.starts_min.split(",")], a.burst_min, a.factor, a.tick, a.cap)
    with open(a.out, "w") as f:
        f.write("\n".join(str(u) for u in users) + "\n")
    print(f"{a.out}: {len(users)} ticks = {len(users) * a.tick / 60:.0f} min, users {min(users)}-{max(users)}")


if __name__ == "__main__":
    main()
