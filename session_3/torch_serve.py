# ── Step 1: Save model as TorchScript ────────────────
import torch


class RideDurationNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


model = RideDurationNet()
# ... train ...
scripted = torch.jit.script(model)  # TorchScript = serializable
scripted.save("ride_duration.pt")

# ── Step 2: Write a custom handler ────────────────────
# handler.py
from ts.torch_handler.base_handler import BaseHandler
import torch
import json


class RideHandler(BaseHandler):
    def preprocess(self, data):
        """Convert raw JSON request → tensor."""
        rows = [json.loads(d["body"]) for d in data]
        feats = [[r["distance_km"], r["passengers"], r["hour_of_day"]] for r in rows]
        return torch.tensor(feats, dtype=torch.float32)

    def postprocess(self, output):
        """Convert model output → list of dicts."""
        return [{"duration_min": round(v.item(), 2)} for v in output.squeeze()]


# ── Step 3: Package into a .mar archive ───────────────
# torch-model-archiver \
#   --model-name ride_duration \
#   --version 1.0 \
#   --serialized-file ride_duration.pt \
#   --handler handler.py \
#   --export-path model_store/

# ── Step 4: Start TorchServe ──────────────────────────
# torchserve --start \
#   --model-store model_store/ \
#   --models ride_duration=ride_duration.mar \
#   --ts-config config.properties
