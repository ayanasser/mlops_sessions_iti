"""Online experimentation patterns for the ride-duration API.

Two ways to compare a new model against production in live traffic, each on its
own endpoint:

* ``POST /predict``         — **A/B testing**: split users deterministically so a
                              given user always sees the same variant, and log
                              which one served the request for offline analysis.
* ``POST /predict/shadow``  — **shadow mode**: always answer with the production
                              model, but *also* run the candidate on the same
                              input and log the delta. The candidate's output is
                              never returned to the caller.

Both share the model's real contract: exactly two features, in order,
``distance_km`` then ``passengers`` (the pickle reports ``n_features_in_ == 2``;
any ``hour_of_day`` in a payload is ignored — that 3-feature form was a bug).

Run locally (cwd = session_3 root):
    uvicorn ab_testing:app --reload
    # or:  python ab_testing.py

Point the candidate at a different artifact to make the comparison meaningful:
    CANDIDATE_MODEL_PATH=models/rf_model_v1.3.pkl python ab_testing.py
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os

import joblib
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ab_testing")

#: Feature order the model was trained on. Do not reorder.
FEATURES = ["distance_km", "passengers"]

#: Control = current production model; candidate = the new version under test.
#: Both default to the session-2 artifact so the demo runs out of the box; in a
#: real experiment you point CANDIDATE_MODEL_PATH at the new model.
CONTROL_MODEL_PATH = os.environ.get("CONTROL_MODEL_PATH", "models/rf_model.pkl")
CANDIDATE_MODEL_PATH = os.environ.get("CANDIDATE_MODEL_PATH", "models/rf_model.pkl")

app = FastAPI(title="ride-duration experimentation demo")

model_a = joblib.load(CONTROL_MODEL_PATH)    # control  / production
model_b = joblib.load(CANDIDATE_MODEL_PATH)  # treatment / candidate


class PredictRequest(BaseModel):
    """A ride to score. Extra fields (e.g. ``hour_of_day``) are ignored."""

    distance_km: float = Field(..., ge=0)
    passengers: int = Field(..., ge=1)

    def to_features(self) -> list[list[float]]:
        """Shape the request into the 2-D array sklearn's ``predict`` expects."""
        return [[getattr(self, name) for name in FEATURES]]


# ── A/B testing — route by user segment ───────────────────────────────────
def get_model_for_user(user_id: str):
    """Deterministic 50/50 split: the same user always gets the same variant."""
    bucket = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
    return model_a if bucket < 50 else model_b


@app.post("/predict")
async def predict(req: PredictRequest, request: Request):
    """Serve the variant assigned to this user and log which one it was."""
    user_id = request.headers.get("X-User-ID", "anonymous")
    model = get_model_for_user(user_id)
    prediction = float(model.predict(req.to_features())[0])

    variant = "A" if model is model_a else "B"
    logger.info("ab_event user_id=%s variant=%s prediction=%.2f",
                user_id, variant, prediction)
    return {"duration_min": round(prediction, 2), "variant": variant}


# ── Shadow mode — mirror traffic, discard the candidate's response ─────────
@app.post("/predict/shadow")
async def predict_shadow(req: PredictRequest):
    """Answer with production; run the candidate in the background and log the gap."""
    features = req.to_features()

    # Production model — this is the response the caller receives.
    prod_result = float(model_a.predict(features)[0])

    # Candidate model — runs fire-and-forget; its result is LOGGED, never returned.
    async def shadow_eval() -> None:
        shadow_result = float(model_b.predict(features)[0])
        logger.info("shadow_compare prod=%.2f shadow=%.2f diff=%.2f",
                    prod_result, shadow_result, abs(prod_result - shadow_result))

    asyncio.create_task(shadow_eval())

    return {"duration_min": round(prod_result, 2)}


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
