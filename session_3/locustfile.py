# locustfile.py
from locust import HttpUser, task, between
import random


class MLAPIUser(HttpUser):
    """Simulates one concurrent user hitting the prediction endpoint."""

    wait_time = between(0.5, 2.0)  # random wait between requests

    @task(weight=10)  # 10x more common than /health
    def predict(self):
        payload = {
            "distance_km": round(random.uniform(0.5, 30.0), 2),
            "passengers": random.randint(1, 4),
            "hour_of_day": random.randint(0, 23),
        }
        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
        ) as resp:
            if resp.status_code != 201:
                resp.failure(f"Expected 201, got {resp.status_code}")
            elif resp.json().get("duration_min", -1) < 0:
                resp.failure("Negative duration in response")

    @task(weight=1)
    def health(self):
        self.client.get("/health")


# ── Run from terminal ──────────────────────────────────
# locust -f locustfile.py --host http://localhost:8000
#        --users 100           ← total concurrent users
#        --spawn-rate 10       ← add 10 users/sec until 100
#        --run-time 2m         ← stop after 2 minutes
#        --headless            ← no UI, print results to CSV
#        --csv=results/load_test
