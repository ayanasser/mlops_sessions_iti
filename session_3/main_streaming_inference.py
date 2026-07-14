# main.py  (deployed as GCP Cloud Function)
import json
import base64
import mlflow.sklearn
import os
from google.cloud import firestore

MODEL = None  # module-level: loaded once per function container (cold start)


def get_model():
    global MODEL
    if MODEL is None:
        MODEL = mlflow.sklearn.load_model(os.environ["MODEL_URI"])
    return MODEL


def predict_on_pubsub(event, context):
    """Triggered automatically by Pub/Sub for every message published."""
    model = get_model()

    # Pub/Sub messages are base64-encoded JSON
    payload = json.loads(base64.b64decode(event["data"]).decode("utf-8"))
    features = [payload["distance_km"], payload["passengers"], payload["hour_of_day"]]

    prediction = model.predict([features])[0]
    result = {
        "ride_id": payload["ride_id"],
        "prediction": round(float(prediction), 2),
    }

    # Write prediction to Firestore
    db = firestore.Client()
    db.collection("ride-predictions").add(result)

    return {"statusCode": 200, "ride_id": result["ride_id"]}


# ── Deploy with gcloud CLI ─────────────────────────────
# gcloud functions deploy predict_on_pubsub \
#   --runtime python311 \
#   --trigger-topic ride-events \
#   --set-env-vars MODEL_URI=models:/RideDurationModel/Production
