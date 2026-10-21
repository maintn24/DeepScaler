"""Locust workload for Online Boutique replaying the FIFA-98 trace (sendFlow/random-100max.req).

Same user behaviour and load shape as sendFlow/load_generator.py, but without
hard-coded absolute paths, and the trace loops forever (stop with Ctrl+C / --run-time).

  locust -f pipeline/locustfile.py --headless --host http://<node-ip>:31000
  TRACE=sendFlow/random-100max.req TICK=5 SCALE=1.0 locust -f pipeline/locustfile.py ...
"""
import os
import random

from locust import HttpUser, LoadTestShape, TaskSet, constant, task

TRACE = os.environ.get("TRACE", os.path.join(os.path.dirname(__file__), "..", "sendFlow", "random-100max.req"))
TICK = float(os.environ.get("TICK", 5))          # seconds per trace value
SCALE = float(os.environ.get("SCALE", 1.0))      # multiply user counts (cluster size)

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


class BoutiqueUser(HttpUser):
    wait_time = constant(1)
    tasks = [BoutiqueTasks]


class FifaTraceShape(LoadTestShape):
    def __init__(self):
        super().__init__()
        with open(TRACE) as f:
            self.users = [max(1, int(round(int(l) * SCALE))) for l in f if l.strip()]

    def tick(self):
        i = int(self.get_run_time() // TICK) % len(self.users)
        return self.users[i], 100
