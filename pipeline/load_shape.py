"""FIFA-98 trace load shape shared by every locustfile (sendFlow/random-100max.req, the trace of the
authors' sendFlow/load_generator.py; paper Sec. V-A / Fig. 8: one workload intensity for all benchmarks).

  TRACE     users per tick file           (default sendFlow/random-100max.req)
  TICK      seconds per trace value       (default 5)
  SCALE     multiply user counts          (default 1.0)
  RUN_TIME  stop after N seconds          (default: loop the trace forever)
"""
import os

from locust import LoadTestShape

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TRACE = os.environ.get("TRACE", os.path.join(_ROOT, "sendFlow", "random-100max.req"))
TICK = float(os.environ.get("TICK", 5))
SCALE = float(os.environ.get("SCALE", 1.0))
RUN_TIME = float(os.environ.get("RUN_TIME", 0))


class FifaTraceShape(LoadTestShape):
    def __init__(self):
        super().__init__()
        with open(TRACE) as f:
            self.users = [max(1, int(round(int(l) * SCALE))) for l in f if l.strip()]

    def tick(self):
        t = self.get_run_time()
        if RUN_TIME and t >= RUN_TIME:
            return None                          # stops the test
        i = int(t // TICK) % len(self.users)
        return self.users[i], 100
