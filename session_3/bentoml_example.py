import bentoml
import numpy as np
from pydantic import BaseModel

# ── Step 1: Save trained model to BentoML model store ─
import mlflow.sklearn

sk_model = mlflow.sklearn.load_model("models:/RideDurationModel/Production")

bento_model = bentoml.sklearn.save_model(
    "ride_duration",  # model name in BentoML store
    sk_model,
    signatures={
        "predict": {"batchable": True, "batch_dim": 0},
    },
)
print(f"Saved: {bento_model.tag}")  # ride_duration:abc123


# ── Step 2: Define the BentoML Service ────────────────
class PredictRequest(BaseModel):
    distance_km: float
    passengers: int
    hour_of_day: int


class PredictResponse(BaseModel):
    duration_min: float


runner = bentoml.sklearn.get("ride_duration:latest").to_runner()

svc = bentoml.Service("ride_api", runners=[runner])


@svc.api(
    input=bentoml.io.JSON(pydantic_model=PredictRequest),
    output=bentoml.io.JSON(pydantic_model=PredictResponse),
)
async def predict(req: PredictRequest) -> PredictResponse:
    features = np.array([[req.distance_km, req.passengers, req.hour_of_day]])
    result = await runner.predict.async_run(features)
    return PredictResponse(duration_min=float(result[0]))


# ── Step 3: Run & containerise ────────────────────────
# bentoml serve service:svc --reload          # dev
# bentoml build                               # package as Bento
# bentoml containerize ride_api:latest        # → Docker image
