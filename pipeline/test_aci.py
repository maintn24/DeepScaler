"""Quick checks of the properties the system design (Claude outputs/problem_formulation_and_system_design.tex)
relies on, without a cluster:  python -m pipeline.test_aci

  1. simultaneous coverage: (forall i in V_act: n_i <= (yhat_i + c) Q - c)  <=>  S <= Q   (score not clipped)
  2. the same with negative / zero forecasts (forecast floored at 1 in AppACI)
  3. long-run one-sided bound of ACI (L = 0):  mean(err) <= alpha + (beta_1 + gamma) / (gamma K)
  4. pacer: alpha goes down while the SLA budget is overspent, up when it is not; eta|e| <= gamma/2
  5. scheduler: scale out at once, scale in only after down_steps lower proposals, -max_down per decision
  6. capacity correction (13/11): off = no-op; online drives p0 to its target when p0 depends on phi,
     sits at phi_min when it does not (infeasible budget); scale_for_capacity leaves V_act^c untouched
"""
import numpy as np

from pipeline.aci import AppACI, CapacityCorrector, Scheduler, SLAPacer, decide, ratio_score, scale_for_capacity
# ORIGINAL: from pipeline.aci import AppACI, Scheduler, SLAPacer, decide, ratio_score


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    return ok


def main():
    rng = np.random.default_rng(0)
    ok = True
    # 1 + 2
    bad1 = bad2 = 0
    for _ in range(5000):
        N = 6
        yhat = rng.uniform(-2, 8, N)
        n = rng.integers(1, 12, N).astype(float)
        aci = AppACI(np.ones(N, bool), c=1.0)
        S = ratio_score(n, AppACI._yh(yhat), 1.0).max()
        if S >= 20 or S <= 1 / 20:
            continue                                  # clipped scores are outside the equivalence
        Q = rng.uniform(0.3, 4)
        aci.beta, aci.scores = 0.5, [Q]               # quantile() -> Q
        covered = (n <= aci.upper(yhat) + 1e-9).all()
        bad1 += covered != (S <= Q + 1e-12)
        bad2 += (S <= Q) and not covered
    ok &= check("1 simultaneous coverage <=> S <= Q", bad1 == 0)
    ok &= check("2 holds with negative forecasts", bad2 == 0)
    # 3 long-run bound on a heavy-tailed, shifting demand, persistence forecast
    alpha, gamma, K = 0.05, 0.01, 4000
    lam = np.cumsum(rng.normal(0, 1, K)) + 40 + 25 * (rng.random(K) < 0.03)    # random walk + bursts
    n = np.maximum(1, np.ceil(np.abs(lam)[:, None] / np.array([5, 9, 13]))).astype(float)
    aci = AppACI(np.ones(3, bool), gamma=gamma, beta0=alpha)
    sched = Scheduler(3, 1, 10 ** 6, down_steps=0, max_down=10 ** 6)
    errs = []
    for t in range(1, K):
        y = decide(n[t - 1], aci, sched, n[t - 1], 1, 10 ** 6)
        err, _ = aci.observe(n[t - 1], n[t], y, alpha)
        errs.append(err)
    rate, bound = np.mean(errs), alpha + (alpha + gamma) / (gamma * (K - 1))
    ok &= check(f"3 ACI miss rate {rate:.4f} <= bound {bound:.4f} (alpha {alpha})", rate <= bound)
    # 4 pacer direction and time scale
    p = SLAPacer(0.05, 150, eta_tight=0.005, eta_relax=0.0005, warmup_steps=0)
    for _ in range(30):
        p.record(True, True)
    a_hi, _ = p.alpha()
    p2 = SLAPacer(0.05, 150, eta_tight=0.005, eta_relax=0.0005, warmup_steps=0)
    for _ in range(30):
        p2.record(False, False)
    a_lo, _ = p2.alpha()
    ok &= check(f"4 pacer: overspent -> alpha {a_hi:.4f} < 0.05 < {a_lo:.4f} <- under budget", a_hi < 0.05 < a_lo)
    ok &= check("4 time scales: eta_tight * (1 - eps) <= gamma / 2", 0.005 * 0.95 <= 0.01 / 2)
    # 5 scheduler
    s = Scheduler(1, 1, 10, down_steps=3, max_down=1)
    seq = [s.apply([t], [2])[0] for t in [5, 1, 1, 1, 1, 1, 1, 8]]
    ok &= check(f"5 scheduler {seq} == [5, 5, 5, 4, 4, 4, 3, 8]", seq == [5, 5, 5, 4, 4, 4, 3, 8])
    # 6 capacity correction
    off = CapacityCorrector("off")
    for _ in range(50):
        off.record(True, False)
    ok &= check("6 phi off stays 1 (original behaviour)", off.phi == 1.0)
    y = np.array([1.0, 2.0, 3.0]); act = np.array([False, True, True])
    ok &= check("6 scale_for_capacity: phi=1 identity, inactive untouched",
                np.allclose(scale_for_capacity(y, act, 1.0), y) and scale_for_capacity(y, act, 0.5)[0] == 1.0
                and np.allclose(scale_for_capacity(y, act, 0.5)[1:], [4.0, 6.0]))
    rng = np.random.default_rng(0)
    c = CapacityCorrector("online", target=0.025, phi_min=0.5)
    v_ok = []
    for t in range(20000):                       # p0 grows with the load per replica, i.e. with phi
        p0_true = 0.01 + 0.2 * (c.phi - 0.5) / 0.5
        v = rng.random() < p0_true
        c.record(v, False)
        if t > 5000:
            v_ok.append(v)
    ok &= check(f"6 online: long-run p0 {np.mean(v_ok):.4f} ~ target 0.025, phi {c.phi:.3f} > phi_min",
                abs(np.mean(v_ok) - 0.025) < 0.005 and c.phi > 0.5)
    c2 = CapacityCorrector("online", target=0.025, phi_min=0.5)
    for _ in range(2000):                        # interference: p0 = 0.25 whatever phi
        c2.record(rng.random() < 0.25, False)
    ok &= check(f"6 online: load-independent p0 -> phi at floor ({c2.phi}), flagged", c2.at_floor())
    c3 = CapacityCorrector("online", target=0.025)
    for _ in range(200):
        c3.record(True, True)                    # shortage steps are the calibrator's job
    ok &= check("6 online: steps with shortage do not move phi", c3.phi == 1.0)
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
