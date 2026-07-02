# Start MLflow tracking server via docker-compose (already in your docker-compose.yml)
# docker compose up -d mlflow  →  visit http://localhost:5000
"""MLflow tracking example — train a ride-duration model and log everything.

Start the MLflow tracking server first (it ships in this folder's
``docker-compose.yml``):

    docker compose up -d mlflow      #  → UI at http://localhost:5000

Then run this script:

    python mflow_example.py

It generates synthetic data, trains a RandomForest regressor (see
``src/train.py``), and logs the parameters, metrics, and the fitted model to
MLflow under the ``ride-duration-model`` experiment. Open http://localhost:5000
to browse the run.
"""

from __future__ import annotations

import os

import mlflow
import mlflow.sklearn

from src.train import DEFAULT_PARAMS, evaluate, generate_data, split_data, train_model

# Point at the tracking server started via docker compose. Override with the
# MLFLOW_TRACKING_URI env var (e.g. to log to a remote server).
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("ride-duration-model")

# ── Data ───────────────────────────────────────────────
# Synthetic ride data, split into train / validation partitions.
X, y = generate_data(n_samples=2_000, noise=1.0, seed=42)
X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.2, seed=42)

with mlflow.start_run(run_name="rf-baseline") as run:

    # ── Hyperparameters ────────────────────────────────────
    params = DEFAULT_PARAMS
    mlflow.log_params(params)

    # ── Train ──────────────────────────────────────────────
    model = train_model(X_train, y_train, params)

    # ── Metrics ────────────────────────────────────────────
    metrics = evaluate(model, X_val, y_val)      # rmse, mae, r2
    mlflow.log_metrics(metrics)
    mlflow.log_metric("val_size", len(y_val))

    # ── Artifact: log the trained model ────────────────────
    # `input_example` lets MLflow infer and store the model signature.
    mlflow.sklearn.log_model(
        model,
        name="model",
        input_example=X_train[:5],
        registered_model_name="RideDurationModel",
    )

    print(f"Run ID: {run.info.run_id}  |  MAE: {metrics['mae']:.3f}  |  R2: {metrics['r2']:.3f}")
    print(f"Open the MLflow UI to inspect this run: {TRACKING_URI}")
