"""Weights & Biases tracking example — the W&B counterpart to ``mflow_example.py``.

Same synthetic ride-duration model as the MLflow example (reused from
``src.train``), but tracked with W&B instead: a baseline run that logs params /
metrics / a model artifact, followed by a Bayesian hyperparameter sweep.

Authentication — pick one before running:

    wandb login                     # log to wandb.ai (needs a free account + API key)
    export WANDB_MODE=offline       # no account/network: writes runs to ./wandb/

Then run:

    python "w&b_example.py"

With ``WANDB_MODE=offline`` you can later push the saved runs with:

    wandb sync wandb/offline-run-*
"""

from __future__ import annotations

import joblib
import wandb

from src.train import (
    evaluate,
    generate_data,
    split_data,
    train_model,
)

PROJECT = "ride-duration"

# ── Data ───────────────────────────────────────────────────────────────────
# Synthetic ride data, split into train / validation partitions (module-level
# so both the baseline run and the sweep's train_fn can reuse it).
X, y = generate_data(n_samples=2_000, noise=1.0, seed=42)
X_train, X_val, y_train, y_val = split_data(X, y, test_size=0.2, seed=42)


# ── Basic logging ──────────────────────────────────────────────────────────
run = wandb.init(project=PROJECT, name="rf-baseline")

# wandb.config is the run's hyperparameter record (the analog of MLflow's
# log_params); it becomes a filterable column in the W&B dashboard.
params = {"n_estimators": 100, "max_depth": 6, "random_state": 42}
wandb.config.update(params)

model = train_model(X_train, y_train, params)
metrics = evaluate(model, X_val, y_val)  # -> {"rmse", "mae", "r2"}

# wandb.log records metrics for the run (numeric → charts in the UI).
wandb.log({**metrics, "val_size": len(y_val)})

# ── Log the model as a W&B Artifact ──────────────────────────────────────────
# Persist the fitted model to disk, then attach it to the run as a versioned
# artifact (W&B's equivalent of an MLflow model artifact).
joblib.dump(model, "model.pkl")
artifact = wandb.Artifact("ride-duration-model", type="model")
artifact.add_file("model.pkl")
run.log_artifact(artifact)
run.finish()

print(f"Baseline run logged  |  MAE={metrics['mae']:.3f}  R2={metrics['r2']:.3f}")


# ── Hyperparameter sweep (Bayesian search) ───────────────────────────────────
def train_fn() -> None:
    """One sweep trial: W&B injects the sampled params via ``wandb.config``."""
    with wandb.init() as sweep_run:
        cfg = dict(sweep_run.config)
        cfg.setdefault("random_state", 42)
        sweep_model = train_model(X_train, y_train, cfg)
        sweep_metrics = evaluate(sweep_model, X_val, y_val)
        wandb.log(sweep_metrics)  # logs "mae" (the sweep's optimisation target)


sweep_config = {
    "method": "bayes",
    "metric": {"name": "mae", "goal": "minimize"},
    "parameters": {
        "n_estimators": {"values": [50, 100, 200]},
        "max_depth": {"min": 3, "max": 10},
    },
}

# Sweeps are orchestrated by the W&B *server*, so they require a login and do
# NOT work in offline mode. Skip gracefully when we can't reach the backend so
# the baseline run above still completes on its own.
if wandb.setup().settings._offline:
    print(
        "Skipping sweep: WANDB_MODE=offline. Run `wandb login` and re-run "
        "without WANDB_MODE=offline to execute the Bayesian sweep."
    )
else:
    sweep_id = wandb.sweep(sweep_config, project=PROJECT)
    wandb.agent(sweep_id, function=train_fn, count=20)  # run 20 trials
    print(f"Sweep {sweep_id} complete.")
