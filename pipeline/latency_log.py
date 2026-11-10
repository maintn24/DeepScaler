"""Client-side, per-request end-to-end latency log for Locust (imported by every locustfile).

Why: the SLA of DeepScaler is defined on the *end-to-end* response time (paper Sec. III-A, eq. (1):
T_violation = time the end-to-end response time violates the SLA). The p95 read from Istio
(`histogram_quantile` on istio_request_duration_milliseconds_bucket) is interpolated inside the
histogram bucket that contains it; with Istio's default buckets (..., 100, 250, ...) a 200 ms SLO
falls inside (100, 250] and a p95 of ~120 ms can be reported as >200 ms. This log keeps the exact
latency of every request as seen by the load generator, so the per-step p95 and the share of slow
requests can be computed without interpolation (pipeline/client_latency.py).

Enabled only when LAT_LOG is set:
  LAT_LOG=Logs/RUN_latency.csv locust -f pipeline/locustfile.py ...
Writes LAT_LOG            : t_end (unix s), method, name, rt_ms, ok
       LAT_LOG.cpu.csv    : t, cpu_percent, users  (Locust's own CPU every 5 s; >90 % means the load
                            generator itself is saturated and client latencies are inflated)
"""
import csv
import os
import time

from locust import events

LAT_LOG = os.environ.get("LAT_LOG")
_state = {"f": None, "w": None}


@events.init.add_listener
def _on_init(environment, **_kw):
    if not LAT_LOG:
        return
    os.makedirs(os.path.dirname(LAT_LOG) or ".", exist_ok=True)
    f = open(LAT_LOG, "w", newline="", buffering=1 << 16)
    w = csv.writer(f)
    w.writerow(["t_end", "method", "name", "rt_ms", "ok"])
    _state.update(f=f, w=w)

    import gevent

    def cpu_loop():
        with open(LAT_LOG + ".cpu.csv", "w") as g:
            g.write("t,cpu_percent,users\n")
            while True:
                r = environment.runner
                if r is not None:
                    g.write(f"{time.time():.3f},{getattr(r, 'current_cpu_usage', float('nan')):.1f},"
                            f"{getattr(r, 'user_count', 0)}\n")
                    g.flush()
                gevent.sleep(5)

    gevent.spawn(cpu_loop)


@events.request.add_listener
def _on_request(request_type, name, response_time, exception=None, **_kw):
    w = _state["w"]
    if w is not None:            # fired when the response (incl. followed redirects) has arrived
        w.writerow([f"{time.time():.3f}", request_type, name, f"{response_time:.1f}", 0 if exception else 1])


@events.quitting.add_listener
def _on_quit(**_kw):
    if _state["f"] is not None:
        _state["f"].flush()
        _state["f"].close()
        _state.update(f=None, w=None)
