# src/batch_score.py
from google.cloud import storage
import pandas as pd
import mlflow.sklearn
from datetime import date
import io


def load_production_model():
    """Always load from MLflow Production stage — no hardcoded paths."""
    return mlflow.sklearn.load_model("models:/RideDurationModel/Production")


def batch_score(bucket_name: str, input_blob: str, output_blob: str) -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # ── 1. Read input data from GCS ────────────────────
    data = bucket.blob(input_blob).download_as_bytes()
    df = pd.read_parquet(io.BytesIO(data))
    print(f"Scoring {len(df):,} rows …")

    # ── 2. Load model & score ──────────────────────────
    model = load_production_model()
    features = df[["distance_km", "passengers", "hour_of_day"]].values
    df["pred"] = model.predict(features)
    df["run_date"] = date.today().isoformat()

    # ── 3. Write predictions back to GCS ──────────────
    out_buffer = io.BytesIO()
    df.to_parquet(out_buffer, index=False)
    bucket.blob(output_blob).upload_from_string(out_buffer.getvalue())
    print(f"Predictions written to gs://{bucket_name}/{output_blob}")


if __name__ == "__main__":
    batch_score(
        bucket_name="mlops-gcs-ride-duration",
        input_blob=f"scoring/input/{date.today()}.parquet",
        output_blob=f"scoring/output/{date.today()}.parquet",
    )
