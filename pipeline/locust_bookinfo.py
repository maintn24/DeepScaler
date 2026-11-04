"""Locust workload for BookInfo (paper Sec. V-A, Fig. 7a) with the same FIFA trace as Online-Boutique.

Requests mimic a book-store visitor: mostly the product page (productpage -> details + reviews ->
ratings), plus the REST API of productpage that hits a single backend.

  DS_APP=bookinfo locust -f pipeline/locust_bookinfo.py --headless --host http://localhost:30012
"""
import os
import random
import sys

from locust import HttpUser, constant, task

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_shape import FifaTraceShape  # noqa: E402,F401  (picked up by Locust)


class BookInfoUser(HttpUser):
    wait_time = constant(1)

    @task(10)
    def product_page(self):
        self.client.get("/productpage", params={"u": random.choice(["normal", "test"])}, name="/productpage")

    @task(2)
    def product_details(self):
        self.client.get("/api/v1/products/0", name="/api/v1/products/[id]")

    @task(2)
    def product_reviews(self):
        self.client.get("/api/v1/products/0/reviews", name="/api/v1/products/[id]/reviews")

    @task(1)
    def product_ratings(self):
        self.client.get("/api/v1/products/0/ratings", name="/api/v1/products/[id]/ratings")

    @task(1)
    def products(self):
        self.client.get("/api/v1/products")
