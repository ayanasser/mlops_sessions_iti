import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ── Basic logging ──────────────────────────────────────
run = wandb.init(project="ride-duration", name="rf-baseline")

params = {"n_estimators": 100, "max_depth": 6}
wandb.config.update(params)

model = RandomForestRegressor(**params)
model.fit(X_train, y_train)
mae = mean_absolute_error(y_val, model.predict(X_val))

wandb.log({"mae": mae, "val_size": len(y_val)})

# ── Log the model as a W&B Artifact ───────────────────
artifact = wandb.Artifact("ride-duration-model", type="model")
artifact.add_file("model.pkl")
run.log_artifact(artifact)
run.finish()

# ── Hyperparameter sweep (Bayesian search) ────────────
sweep_config = {
    "method": "bayes",
    "metric": {"name": "mae", "goal": "minimize"},
    "parameters": {
        "n_estimators": {"values": [50, 100, 200]},
        "max_depth":    {"min": 3, "max": 10},
    },
}
sweep_id = wandb.sweep(sweep_config, project="ride-duration")
wandb.agent(sweep_id, function=train_fn, count=20)   # run 20 trials
