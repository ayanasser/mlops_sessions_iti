"""MLflow tracking example — train a ride-duration model and log everything.


Start the MLflow tracking server first (it ships in this folder's
``docker-compose.yml``):

    docker compose up -d mlflow      #  → UI at http://localhost:5000

Then run this script:

    python mflow_example.py

Open http://localhost:5000 and click into the ``ride-duration-model``
experiment to browse the run, its charts, artifacts, and the registered model.
"""

from __future__ import annotations

import os

# Use a non-interactive matplotlib backend: we only save figures to disk / hand
# them to MLflow, we never pop up a window. This MUST be set before pyplot is
# imported.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.train import (
    DEFAULT_PARAMS,
    evaluate,
    generate_data,
    split_data,
    train_model,
)

# Point at the tracking server started via docker compose. Override with the
# MLFLOW_TRACKING_URI env var (e.g. to log to a remote server).
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# Name under which the model lives in the Model Registry (a separate concept
# from the experiment: an experiment groups *runs*, the registry versions
# *models* promoted out of those runs).
REGISTERED_MODEL_NAME = "RideDurationModel"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("ride-duration-model")

# ── Data ───────────────────────────────────────────────────────────────────
# Synthetic ride data, split into train / validation partitions.
X, y = generate_data(n_samples=2_000, noise=1.0, seed=42)
X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.2, seed=42)


with mlflow.start_run(run_name="rf-baseline") as run:
    # ── 1. Hyperparameters ──────────────────────────────────────────────────
    # log_params writes the run's configuration. These show up in the run's
    # "Parameters" table and become filterable/sortable columns in the
    # experiment view — the main way you compare runs against each other.
    params = DEFAULT_PARAMS
    mlflow.log_params(params)

    # Tags are free-form key/value metadata (not numeric like metrics, not
    # config like params). Handy for labelling runs so you can search them later
    # e.g. `tags.dataset = "synthetic-v1"` in the UI search bar.
    mlflow.set_tags(
        {
            "model_family": "random_forest",
            "dataset": "synthetic-v1",
            "stage": "experiment",
        }
    )

    # ── 2. Train the baseline model ─────────────────────────────────────────
    model = train_model(X_train, y_train, params)

    # ── 3. Metrics ──────────────────────────────────────────────────────────
    # Scalar metrics (single value each) → shown in the "Metrics" table.
    metrics = evaluate(model, X_val, y_val)  # -> {"rmse", "mae", "r2"}
    mlflow.log_metrics(metrics)
    mlflow.log_metric("val_size", len(y_val))

    # ── 4. Metric visualisation, technique A: STEP metrics (a line chart) ────
    # A metric logged repeatedly with an increasing `step` becomes a *series*,
    # which MLflow plots as an interactive line chart. Here we build a "learning
    # curve": retrain the forest with a growing number of trees and record the
    # validation error at each size. In the UI this answers "do more trees keep
    # helping, or have we plateaued?" at a glance.
    for n_estimators in (10, 25, 50, 100, 200, 400):
        sweep_params = {**params, "n_estimators": n_estimators}
        sweep_model = train_model(X_train, y_train, sweep_params)
        sweep_metrics = evaluate(sweep_model, X_val, y_val)
        # Same metric NAME, different `step` → one line per metric on the chart.
        # `step` is just the x-axis position (here: the tree count).
        mlflow.log_metric("lc_val_rmse", sweep_metrics["rmse"], step=n_estimators)
        mlflow.log_metric("lc_val_mae", sweep_metrics["mae"], step=n_estimators)
        mlflow.log_metric("lc_val_r2", sweep_metrics["r2"], step=n_estimators)

    # ── 5. Metric visualisation, technique B: FIGURE artifacts (PNGs) ────────
    # `mlflow.log_figure` uploads a matplotlib Figure as an image artifact that
    # renders inline in the UI's artifact browser. Great for diagnostics that
    # aren't a single number: fit quality, error distribution, what the model
    # considers important.
    preds = model.predict(X_val)
    residuals = y_val - preds

    # (a) Predicted vs. actual — points hug the diagonal when the model fits.
    fig_pva, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_val, preds, s=8, alpha=0.4)
    lims = [min(y_val.min(), preds.min()), max(y_val.max(), preds.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="perfect")  # y = x reference
    ax.set_xlabel("actual duration (min)")
    ax.set_ylabel("predicted duration (min)")
    ax.set_title("Predicted vs. actual")
    ax.legend()
    mlflow.log_figure(fig_pva, "plots/predicted_vs_actual.png")
    plt.close(fig_pva)  # free the figure so we don't leak memory across runs

    # (b) Residual histogram — should be roughly centred on 0 and bell-shaped.
    fig_res, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=40)
    ax.axvline(0, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("residual (actual − predicted, min)")
    ax.set_ylabel("count")
    ax.set_title("Residual distribution")
    mlflow.log_figure(fig_res, "plots/residuals.png")
    plt.close(fig_res)

    # (c) Feature importances — which inputs drive the prediction.
    from src.train import FEATURES  # ["distance_km", "passengers"]

    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    fig_imp, ax = plt.subplots(figsize=(6, 4))
    ax.bar([FEATURES[i] for i in order], importances[order])
    ax.set_ylabel("importance")
    ax.set_title("Feature importances")
    mlflow.log_figure(fig_imp, "plots/feature_importances.png")
    plt.close(fig_imp)

    # ── 6. Log AND register the model ───────────────────────────────────────
    # Passing `registered_model_name` does two things in one call:
    #   • logs the fitted model as a run artifact (with a signature inferred
    #     from `input_example`), and
    #   • creates/increments a version of that name in the Model Registry.
    # `log_model` returns a ModelInfo that tells us which registry version was
    # just created, so we can annotate it below.
    model_info = mlflow.sklearn.log_model(
        model,
        name="model",
        input_example=X_train[:5],
        registered_model_name=REGISTERED_MODEL_NAME,
    )

    run_id = run.info.run_id


# ── 7. Enrich the registered model version ──────────────────────────────────
# These calls target the *registry* (not the run), so they live outside the
# `with` block. MlflowClient is the lower-level API for registry management.
client = MlflowClient()

# The version number MLflow assigned to the model we just registered.
version = model_info.registered_model_version

# A human-readable description shown on the model-version page.
client.update_model_version(
    name=REGISTERED_MODEL_NAME,
    version=version,
    description=(
        "RandomForest ride-duration regressor trained on synthetic-v1 data. "
        f"Validation MAE={metrics['mae']:.3f}, R2={metrics['r2']:.3f}."
    ),
)

# Tags on the version — searchable and useful for governance.
client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "validated", "true")
client.set_model_version_tag(REGISTERED_MODEL_NAME, version, "source_run", run_id)

# An *alias* is a moving pointer to a specific version (aliases replaced the old
# "stages" like Staging/Production in MLflow 2.9+). Downstream code can then load
# `models:/RideDurationModel@champion` and always get whichever version we've
# blessed — without hard-coding a version number.
client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", version)


# ── Summary ──────────────────────────────────────────────────────────────────
print(f"Run ID: {run_id}  |  MAE: {metrics['mae']:.3f}  |  R2: {metrics['r2']:.3f}")
print(f"Registered '{REGISTERED_MODEL_NAME}' version {version} (alias: @champion)")
print(f"Open the MLflow UI to inspect this run: {TRACKING_URI}")
print("  • Learning-curve chart:  run → 'Model metrics' tab (lc_val_* series)")
print("  • Diagnostic plots:      run → 'Artifacts' → plots/")
print(f"  • Registered model:      Models → {REGISTERED_MODEL_NAME}")
print(f"Load the blessed model later with: models:/{REGISTERED_MODEL_NAME}@champion")
