"""Locust workload for Train-Ticket (paper Sec. V-A, Fig. 7c) = the DeepScaler authors' own
generator sendFlow/load_generator_train.py, reused unchanged:
  * TaskSet TrainTicketUserTasks: 14 user flows with weight 1 each (login, then admin config/price,
    travel plan, query+preserve, cancel, collect, enter station, consign, rebook, food,
    order+pay, route, travel left), wait_time constant(1) as in the authors' WebsiteUser.
  * Load shape: the same FIFA trace (sendFlow/random-100max.req, 5 s per line) through
    pipeline/load_shape.py; the authors' StagesShape reads it from a hard-coded path on their machine.
The authors' file imports `atomic_queries` (train-ticket-auto-query) for two names it never uses;
a stub module is installed so it imports without that repository.

  DS_APP=trainticket locust -f pipeline/locust_trainticket.py --headless --host http://localhost:32677
"""
import os
import sys
import types

from locust import HttpUser, constant

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "sendFlow"))
sys.modules.setdefault("atomic_queries", types.SimpleNamespace(_query_advanced_ticket=None, _login=None))

from load_generator_train import TrainTicketUserTasks  # noqa: E402  (authors' flows)
from load_shape import FifaTraceShape  # noqa: E402,F401  (picked up by Locust)
import latency_log  # noqa: E402,F401  (per-request latency log when LAT_LOG is set)
# CÁCH CŨ (trước 10/11/2026): không có dòng import trên. Bỏ LAT_LOG thì hành vi y như cũ.


class TrainTicketUser(HttpUser):
    wait_time = constant(1)
    tasks = [TrainTicketUserTasks]
