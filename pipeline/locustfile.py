"""Locust workload for Online Boutique replaying the FIFA-98 trace (sendFlow/random-100max.req).

Same user behaviour and load shape as sendFlow/load_generator.py, but without
hard-coded absolute paths; the trace loops until RUN_TIME seconds have passed
(RUN_TIME=0 -> forever). Locust ignores --run-time when a LoadTestShape is used,
so the duration must be enforced here by returning None from tick().

  locust -f pipeline/locustfile.py --headless --host http://<node-ip>:31000
  TRACE=sendFlow/random-100max.req TICK=5 SCALE=1.0 locust -f pipeline/locustfile.py ...
"""
import os
import random

from locust import HttpUser, LoadTestShape, TaskSet, constant, task

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import latency_log  # noqa: E402,F401  (per-request latency log when LAT_LOG is set)
# CÁCH CŨ (trước 10/11/2026): không có dòng import trên. Bỏ LAT_LOG thì hành vi y như cũ.

TRACE = os.environ.get("TRACE", os.path.join(os.path.dirname(__file__), "..", "sendFlow", "random-100max.req"))
TICK = float(os.environ.get("TICK", 5))          # seconds per trace value
SCALE = float(os.environ.get("SCALE", 1.0))      # multiply user counts (cluster size)
RUN_TIME = float(os.environ.get("RUN_TIME", 0))  # seconds; 0 = run until Ctrl+C

PRODUCTS = ['0PUK6V6EV0', '1YMWWN1N4O', '2ZYFJ3GM2N', '66VCHSJNUP', '6E92ZMYYFZ',
            '9SIQT8TOJO', 'L9ECAV7KIM', 'LS4PSXUNUM', 'OLJCESPC7Z']


class BoutiqueTasks(TaskSet):
    @task(1)
    def index(self):
        self.client.get("/")

    @task(2)
    def set_currency(self):
        self.client.post("/setCurrency", {"currency_code": random.choice(["EUR", "USD", "JPY", "CAD"])})

    @task(10)
    def browse_product(self):
        self.client.get("/product/" + random.choice(PRODUCTS), name="/product/[id]")

    @task(2)
    def view_cart(self):
        self.client.get("/cart")

    @task(3)
    def add_to_cart(self):
        product = random.choice(PRODUCTS)
        self.client.get("/product/" + product, name="/product/[id]")
        self.client.post("/cart", {"product_id": product, "quantity": random.choice([1, 2, 3, 4, 5, 10])})

    @task(1)
    def checkout(self):
        self.add_to_cart()
        self.client.post("/cart/checkout", {
            "email": "someone@example.com", "street_address": "1600 Amphitheatre Parkway",
            "zip_code": "94043", "city": "Mountain View", "state": "CA", "country": "United States",
            "credit_card_number": "4432-8015-6152-0454", "credit_card_expiration_month": "1",
            "credit_card_expiration_year": "2039", "credit_card_cvv": "672"})


# Request mix (paper Sec. V-C / Fig. 11: "tuning different weights of user request types").
# Default = weights of the authors' sendFlow/load_generator.py; override e.g.
#   TASK_WEIGHTS="index=1,set_currency=2,browse_product=4,view_cart=2,add_to_cart=6,checkout=4"
_DEFAULT_W = {"index": 1, "set_currency": 2, "browse_product": 10, "view_cart": 2, "add_to_cart": 3, "checkout": 1}
_W = dict(_DEFAULT_W)
for kv in filter(None, os.environ.get("TASK_WEIGHTS", "").split(",")):
    k, v = kv.split("=")
    assert k.strip() in _W, f"unknown task {k}; choose from {list(_W)}"
    _W[k.strip()] = int(v)
# a new TaskSet class so that Locust's metaclass turns the {function: weight} dict into its task list
WeightedBoutiqueTasks = type("WeightedBoutiqueTasks", (BoutiqueTasks,),
                             {"tasks": {getattr(BoutiqueTasks, k): w for k, w in _W.items() if w > 0}})


# KEEPALIVE=0 (thesis, 11/11/2026): close the TCP connection after every request. The entry service is a
# NodePort (frontend-external in the authors' boutique.yaml), so kube-proxy balances per CONNECTION. With
# keep-alive each Locust user stays pinned to the frontend pod it first reached: users created while there
# was 1 pod keep loading that pod after scale-out (seen 10/11: one frontend pod 43.7 % CPU-throttled, the
# others 1-14 %), so added replicas barely help and SLA violations appear at any replica count.
# Default KEEPALIVE=1 = ORIGINAL behaviour (requests.Session keep-alive), used by every run before 11/11.
KEEPALIVE = os.environ.get("KEEPALIVE", "1") != "0"


class BoutiqueUser(HttpUser):
    wait_time = constant(1)
    tasks = [WeightedBoutiqueTasks]

    def on_start(self):
        if not KEEPALIVE:
            self.client.headers.update({"Connection": "close"})


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
