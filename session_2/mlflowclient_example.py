from mlflow import MlflowClient

client = MlflowClient("http://localhost:5000")

# ── List all registered versions ──────────────────────
for v in client.search_model_versions("name='RideDurationModel'"):
    print(f"v{v.version}  stage={v.current_stage}")

# ── Promote to Staging ────────────────────────────────
client.transition_model_version_stage(
    name="RideDurationModel", version="3", stage="Staging",
)

# ── Load Production model for serving ────────────────
import mlflow.sklearn
model = mlflow.sklearn.load_model("models:/RideDurationModel/Production")
