"""Load traces (users per TICK, one int per line) for pipeline/locustfile.py (TRACE=...).

  ramp  : capacity profiling for mu (demand label, thesis objective O1). Users go up in
          equal steps and each level is held long enough for Prometheus rate windows
          (default 10 -> 150 users, step 10, 2 min/level at TICK=5 s => 30 min).
  burst : the FIFA trace used for training (sendFlow/random-100max.req) with sudden
          bursts: at each start index the user count is multiplied by --factor for
          --burst_min minutes, then drops back instantly. Same base load as training, so
          only the bursts differ (DeepScaler paper, Sec. V-B: bursts raise SLA violations
          for all prediction-based autoscalers, Fig. 10 min 30-50).

  train : training-data trace (thesis): the FIFA trace looped for --hours, each loop multiplied by
          the next value of --scales (different load levels -> HPA uses many replica counts and
          also scales in), plus sudden bursts at random times (seeded): gaps ~ exponential with
          mean --burst_every_min, length uniform in --burst_len, factor uniform in --factor_rng,
          capped at --cap users. Bursts are faster than HPA reacts, so the data contains SLA
          violations (paper Sec. V-B / Fig. 10 discussion; Sec. VI: DeepScaler is "limited by
          the quality and quantity of available historical data"). Writes <out>.bursts.csv too.

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


def train(base, hours, scales, tick_s, every_min, burst_len, factor_rng, cap, warmup_min, seed):
    rng = np.random.default_rng(seed)
    n = int(hours * 3600 / tick_s)
    users, k = [], 0
    while len(users) < n:                                   # loop the trace, one scale per loop
        users += [u * scales[k % len(scales)] for u in base]
        k += 1
    users = np.array(users[:n], dtype=float)
    bursts, t = [], warmup_min + rng.exponential(every_min)
    while t < hours * 60 - burst_len[1]:
        i, ln, f = int(t * 60 / tick_s), rng.uniform(*burst_len), rng.uniform(*factor_rng)
        j = i + int(ln * 60 / tick_s)
        users[i:j] *= f
        bursts.append((round(t, 2), round(ln, 2), round(f, 2)))
        t += ln + rng.exponential(every_min)
    return [max(1, int(round(min(u, cap)))) for u in users], bursts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=["ramp", "burst", "train"])
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
    p.add_argument("--cap", type=int, default=200, help="max users (train: 220 recommended)")
    p.add_argument("--hours", type=float, default=6, help="train: length of the trace")
    p.add_argument("--scales", default="1.0,1.5,0.6,1.8,1.2", help="train: load multiplier per trace loop")
    p.add_argument("--burst_every_min", type=float, default=20, help="train: mean gap between bursts")
    p.add_argument("--burst_len", default="1,4", help="train: burst length range (minutes)")
    p.add_argument("--factor_rng", default="1.8,3.0", help="train: burst multiplier range")
    p.add_argument("--warmup_min", type=float, default=10, help="train: no burst before this")
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()
    if a.kind == "train":
        base = [int(l) for l in open(a.base) if l.strip()]
        users, bursts = train(base, a.hours, [float(x) for x in a.scales.split(",")], a.tick,
                              a.burst_every_min, [float(x) for x in a.burst_len.split(",")],
                              [float(x) for x in a.factor_rng.split(",")], a.cap, a.warmup_min, a.seed)
        with open(a.out + ".bursts.csv", "w") as f:
            f.write("start_min,length_min,factor\n" + "".join(f"{b[0]},{b[1]},{b[2]}\n" for b in bursts))
        print(f"{len(bursts)} bursts -> {a.out}.bursts.csv")
    elif a.kind == "ramp":
        users = ramp(a.lo, a.hi, a.step, int(a.hold_min * 60 / a.tick))
    else:
        base = [int(l) for l in open(a.base) if l.strip()]
        users = burst(base, [float(x) for x in a.starts_min.split(",")], a.burst_min, a.factor, a.tick, a.cap)
    with open(a.out, "w") as f:
        f.write("\n".join(str(u) for u in users) + "\n")
    print(f"{a.out}: {len(users)} ticks = {len(users) * a.tick / 60:.0f} min, users {min(users)}-{max(users)}")


if __name__ == "__main__":
    main()
