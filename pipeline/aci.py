"""Calibrated holistic decision layer for DeepScaler (thesis): application-level ACI + SLA budget pacer.

Replaces only the step "point forecast -> replicas" of DeepScaler. The original step is kept in
pipeline/autoscaler.py: round with floor/ceil at fraction 0.35 (authors' predict_scale.py) plus a
heuristic trust threshold (paper Sec. IV-D). Design and notation: Claude outputs/
problem_formulation_and_system_design.tex, Sec. 2.3-2.5.

  AppACI     inner loop. One nonconformity score per decision for the whole application,
             S_k = max_{i in V_act} (n_i + c) / (yhat_i + c), BACC's ratio score (Liu et al., BACC v2,
             unpublished manuscript 2026, eq. (4)); one quantile Q = Q_(1-beta) of the recent scores
             (split-conformal rank, Lei et al. 2018); every active service gets yhat_i' = (yhat_i + c) Q - c.
             (forall i in V_act: n_i <= yhat_i')  <=>  S_k <= Q, so coverage is simultaneous.
             beta follows ACI (Gibbs & Candes 2021, eq. (2)): beta += gamma (alpha_target - err),
             err = 1[some active service had fewer DEPLOYED replicas than its demand]. beta is capped
             at 1 from above only; with beta <= 0 -> Q = inf -> r_max (keeps the one-sided bound).
  SLAPacer   outer loop. alpha (target miss probability) from the SLA error budget eps
             (BACC v2 eq. (9) pacing + eq. (10)-(12) remaining-budget ceiling with p0/p1).
  Scheduler  scale out at once; scale in only after `down_steps` lower proposals, at most
             `max_down` replicas per decision; changes counted from the planned count.
"""
import math
from collections import deque

import numpy as np

SCORE_CLIP = (1 / 20, 20.0)


def ratio_score(n, yhat, c):
    return np.clip((np.asarray(n, float) + c) / (np.asarray(yhat, float) + c), *SCORE_CLIP)


def demand(req, mu_vec):
    """Replicas needed by the load (thesis O1): max(1, ceil(req / mu))."""
    return np.maximum(1.0, np.ceil(np.asarray(req, float) / mu_vec))


class AppACI:
    def __init__(self, active, m=120, gamma=0.01, c=1.0, beta0=0.05, warm_scores=()):
        self.active = np.asarray(active, bool)
        self.m, self.gamma, self.c, self.beta = m, gamma, c, float(beta0)
        self.scores = deque((float(s) for s in warm_scores), maxlen=m)

    def quantile(self):
        if self.beta <= 0:
            return math.inf
        if not self.scores:
            return 1.0                                   # BACC v2: q = 1 before any window completes
        s = sorted(self.scores)
        if self.beta >= 1:
            return s[0]
        j = min(len(s), math.ceil((len(s) + 1) * (1 - self.beta)))
        return s[j - 1]

    @staticmethod
    def _yh(yhat):
        """Forecast floored at 1 replica (demand >= 1 by definition). The design needs yhat + c > 0:
        a model can output negative values, and then (yhat + c) Q - c no longer bounds n when S <= Q.
        ORIGINAL (before 10/11/2026 review): no floor, np.asarray(yhat, float)."""
        return np.maximum(np.asarray(yhat, float), 1.0)

    def upper(self, yhat):
        q = self.quantile()
        if math.isinf(q):
            return np.full(len(yhat), np.inf)
        return (self._yh(yhat) + self.c) * q - self.c

    def observe(self, yhat, n, deployed, alpha_target):
        """Feedback for one past decision: forecast yhat, realised demand n, replicas actually deployed."""
        n, deployed = np.asarray(n, float), np.asarray(deployed, float)
        s = float(ratio_score(n[self.active], self._yh(yhat)[self.active], self.c).max())
        self.scores.append(s)
        err = bool((deployed[self.active] < n[self.active]).any())
        self.beta = min(1.0, self.beta + self.gamma * (alpha_target - float(err)))
        return err, s


class SLAPacer:
    """alpha_k from SLA feedback. enabled=False -> fixed alpha = alpha_fixed (ACI-only ablation)."""

    def __init__(self, eps, period_steps, window=120, eta_tight=0.02, eta_relax=0.002,
                 a_min=0.01, a_max=0.5, a_cal_min=0.005, startup_steps=1, enabled=True, ceiling=True,
                 alpha_fixed=None, warmup_steps=10, min_count=30):
        # warmup_steps: no pacing in the first steps (BACC v2 also skips a 5-minute warm-up);
        # min_count: steps needed in BOTH groups (with / without a miss) before p0, p1 are trusted
        # by the ceiling; with a handful of steps their ratio swings and forces alpha to its floor.
        self.eps, self.T, self.w = eps, period_steps, window
        self.eta_t, self.eta_r, self.a_min, self.a_max, self.a_cal_min = eta_tight, eta_relax, a_min, a_max, a_cal_min
        self.d, self.enabled, self.ceiling = startup_steps, enabled, ceiling
        self.warmup, self.min_count = warmup_steps, min_count
        self.alpha_fixed = eps if alpha_fixed is None else alpha_fixed
        self.a_pace = eps
        self.v = []
        self.K = {0: 0, 1: 0}
        self.U = {0: 0, 1: 0}

    def record(self, violated, missed):
        """One monitoring step: SLA violated? some active service short of demand?"""
        self.v.append(bool(violated))
        j = int(bool(missed))
        self.U[j] += 1
        self.K[j] += int(bool(violated))

    def p(self, j):                                      # Jeffreys posterior mean (BACC v2 eq. (11))
        return (self.K[j] + 0.5) / (self.U[j] + 1)

    def alpha(self):
        if not self.enabled:
            return self.alpha_fixed, {}
        t = len(self.v)
        info = {}
        if t <= self.warmup:
            return max(self.a_cal_min, self.a_pace), {"warmup": True}
        if t:
            r_rol, r_cum = float(np.mean(self.v[-self.w:])), float(np.mean(self.v))
            e = max(r_rol, r_cum) - self.eps
            eta = self.eta_t if e > 0 else self.eta_r
            self.a_pace = float(np.clip(self.a_pace - eta * e, self.a_min, self.a_max))
            info.update(R_rol=r_rol, R_cum=r_cum, e=e)
        a_ceil = 1.0
        if self.ceiling and t < self.T and min(self.U.values()) >= self.min_count:
            rho = float(self.v[-1]) * min(self.d, self.T - t)
            u = float(np.clip((math.floor(self.eps * self.T) - sum(self.v) - rho) / (self.T - t), 0, 1))
            p0, p1 = self.p(0), self.p(1)
            if u < self.eps and p1 > p0:
                a_ceil = float(np.clip((u - p0) / (p1 - p0), 0, 1))
            info.update(u=u, p0=p0, p1=p1)
        a = max(self.a_cal_min, min(self.a_pace, a_ceil))
        info.update(a_pace=self.a_pace, a_ceil=a_ceil)
        return a, info


class CapacityCorrector:
    """Online correction of the per-replica capacity, mu_eff = phi * mu (extension added 13/11/2026;
    the ORIGINAL design = mode "off", phi = 1; full copy of the previous file: pipeline/aci_v1_20261112.py).

    Why. The demand label n = max(1, ceil(lambda / mu)) is only as good as mu. mu is profiled one service
    at a time with the others over-provisioned (thesis Sec. 3.2), so it is optimistic when all services
    run near capacity together. Violations then happen at steps where no service is short of its (too
    low) demand -- the p0 term of Eq. (3.3). ACI sees no miss there and the pacer can only lower alpha,
    which adds nothing, so neither loop can add replicas (online run 10/11: p0 = 0.33).

    Update, only at steps where no active service is short (shortage is the calibrator's job):
        phi <- clip(phi - eta * (v - target), phi_min, phi_max)
    i.e. the stochastic-approximation form of the ACI update (Gibbs & Candes 2021, eq. (2)) applied to
    the violation indicator: in the long run the violation rate at steps without shortage (p0) is driven
    to `target` as long as phi stays inside [phi_min, phi_max]. Updating the capacity profile from
    observed violations follows BACCv2's profile update (unpublished manuscript, Sec. III-C3a).
    phi_min caps the extra cost (phi = 0.5 doubles the demand of a loaded service). If p0 stays above
    `target` even at phi_min, the violations do not depend on the load per replica (e.g. node-level
    interference): phi then sits at phi_min, which flags the budget as infeasible (Eq. (3.1)).

    mode: "off" -> phi = 1 always (ORIGINAL); "fixed" -> phi = phi0 always; "online" -> update above.
    """

    def __init__(self, mode="off", phi0=1.0, eta=0.02, target=0.025, phi_min=0.5, phi_max=1.0,
                 warmup_steps=10):
        if mode not in ("off", "fixed", "online"):
            raise ValueError(mode)
        self.mode, self.eta, self.target = mode, eta, target
        self.phi_min, self.phi_max, self.warmup = phi_min, phi_max, warmup_steps
        self.phi = 1.0 if mode == "off" else float(np.clip(phi0, phi_min, phi_max))
        self.t = 0
        self.z, self.Z = 0, 0                  # violations / steps without shortage (= p0 estimate)

    def record(self, violated, missed):
        """One monitoring step: SLA violated? some active service short of demand?"""
        self.t += 1
        if missed:
            return
        self.Z += 1
        self.z += int(bool(violated))
        if self.mode == "online" and self.t > self.warmup:
            self.phi = float(np.clip(self.phi - self.eta * (float(bool(violated)) - self.target),
                                     self.phi_min, self.phi_max))

    def p0(self):
        return (self.z + 0.5) / (self.Z + 1)   # Jeffreys posterior mean, as for p0 in SLAPacer

    def at_floor(self):
        return self.mode == "online" and self.phi <= self.phi_min + 1e-9


def scale_for_capacity(yhat, active, phi):
    """Forecast under mu_eff = phi * mu. The forecaster predicts n = ceil(lambda / mu); for a loaded
    service n ~ lambda / mu, so its demand under mu_eff is ~ n / phi. Applied to V_act only: a service
    whose demand is one replica because its load is tiny keeps one replica whatever phi.
    phi = 1 returns the forecast unchanged (ORIGINAL behaviour)."""
    y = np.asarray(yhat, float).copy()
    if phi != 1.0:
        act = np.asarray(active, bool)
        y[act] = y[act] / phi
    return y


class Scheduler:
    def __init__(self, n, r_min=1, r_max=10, down_steps=3, max_down=1):
        self.r_min, self.r_max, self.down_steps, self.max_down = r_min, r_max, down_steps, max_down
        self.planned = None
        self.low = np.zeros(n, int)

    def apply(self, target, current):
        target = np.clip(np.asarray(target), self.r_min, self.r_max).astype(int)
        if self.planned is None:
            self.planned = np.asarray(current, int).copy()
        new = self.planned.copy()
        up = target > self.planned
        new[up] = target[up]                                       # scale out at once
        down = target < self.planned
        self.low = np.where(down, self.low + 1, 0)
        go = down & (self.low >= self.down_steps)
        new[go] = np.maximum(target[go], self.planned[go] - self.max_down)
        self.low[go] = 0
        self.planned = new
        return new


def decide(yhat, aci, sched, current, r_min=1, r_max=10, inactive_rule="deepscaler"):
    """Replicas for one decision. Active: floor of the calibrated upper bound.
    Services outside V_act (no calibration data, demand ~always 1):
      inactive_rule="deepscaler" (default since 11/11/2026): DeepScaler's own rounding floor(yhat + 0.65)
        (authors' predict_scale.py; pipeline/autoscaler.scale_to_pods). A forecast of 1.02 -> 1 replica.
      inactive_rule="ceil" (ORIGINAL design text, tex Sec. 2.5): ceil(yhat). A forecast slightly above 1
        gives 2 replicas for every idle service (seen online 10/11: 7 idle services at 2 pods)."""
    up = aci.upper(yhat)
    act = aci.active
    yh = AppACI._yh(yhat)
    other = np.ceil(yh) if inactive_rule == "ceil" else np.floor(yh + 0.65)
    target = np.where(act, np.where(np.isinf(up), r_max, np.floor(np.nan_to_num(up, posinf=r_max))), other)
    return sched.apply(np.clip(target, r_min, r_max), current)


def active_set(n_hist, rho=0.05):
    """Services whose demand exceeds 1 replica in at least rho of the calibration steps."""
    return (np.asarray(n_hist) > 1).mean(0) >= rho
