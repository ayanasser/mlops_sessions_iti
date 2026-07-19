"""
LEVEL 4 — A purpose-built serving runtime (BentoML).

Compare this file against level_2_batching.py. They do the same thing.

  Level 2, by hand                     Level 4, BentoML
  -----------------------------------  ------------------------------------
  ~90 lines between SERVING LAYER      batchable=True
  markers: queue, window, futures,     max_batch_size=32
  backpressure, partial failure        max_latency_ms=8
  uvicorn --workers 1 (batcher state)  workers=1 in @bentoml.service
  hand-rolled /metrics                 Prometheus at /metrics, free
  MODEL_VERSION = "v1" constant        model store: versioned, hot-swappable
  write your own Dockerfile            bentoml build && bentoml containerize

The five [COST-n] problems annotated in Level 2 are now three config lines.
That is the entire argument for not writing it yourself.

Setup:  python -c "import bentoml, torchvision; \\
          bentoml.pytorch.save_model('resnet50', \\
            torchvision.models.resnet50(weights='DEFAULT').eval())"
Run:    bentoml serve level_4_bentoml_service:ResNet50 --port 8004
Then:   python bench.py --port 8004 -c 32 -n 200
"""



import bentoml
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models import ResNet50_Weights

CATEGORIES = ResNet50_Weights.DEFAULT.meta["categories"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_preprocess = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@bentoml.service(
    name="resnet50_classifier",
    # workers=1 for the same reason Level 2 needed --workers 1: one process
    # owns the accelerator. Scale with replicas, not workers.
    workers=1,
    resources={"cpu": "2"},
    traffic={"timeout": 60},
)
class ResNet50:
    # Versioned model from the BentoML store. "latest" is resolvable to a
    # specific hash -- this is the model versioning Level 1 did not have.
    # Running v1 and v2 side by side is two service definitions, not two
    # servers you operate by hand.
    bento_model = bentoml.models.get("resnet50:latest")

    def __init__(self) -> None:
        self.model = bentoml.pytorch.load_model(self.bento_model).eval().to(DEVICE)

    # ---- THE ENTIRE LEVEL 2 SERVING LAYER, AS A DECORATOR -----------------
    @bentoml.api(
        batchable=True,          # [COST-1] queue + window: handled
        batch_dim=0,             # [COST-2] response correlation: handled
        max_batch_size=32,       # [COST-4] backpressure: handled
        max_latency_ms=8,        # the batch window
    )
    @torch.inference_mode()
    def predict_batch(self, images: list[np.ndarray]) -> list[list[dict]]:
        """BentoML collects concurrent single-item calls into `images` for us,
        runs this once, and fans the results back to the right clients."""
        batch = torch.from_numpy(np.stack(images)).to(DEVICE)
        probs = torch.softmax(self.model(batch), dim=-1)
        top5 = torch.topk(probs, k=5, dim=-1)
        return [
            [{"label": CATEGORIES[i], "score": round(float(s), 4)}
             for s, i in zip(scores, idxs)]
            for scores, idxs in zip(top5.values.cpu(), top5.indices.cpu())
        ]

    @bentoml.api(route="/predict")
    async def predict(self, file: Image.Image) -> dict:
        """Single-image entrypoint. Annotating the param as PIL.Image is all it
        takes for BentoML to accept a multipart upload and hand us a decoded
        image -- no UploadFile, no io.BytesIO, no manual 400 on bad input.

        It then hands off to the batchable API above, so concurrent callers
        still get merged into one forward pass. Preprocessing still costs what
        Level 3 measured: no framework makes JPEG decode free.
        """
        tensor = _preprocess(file.convert("RGB")).numpy()
        preds = await self.predict_batch.to_async([tensor])
        return {"version": self.bento_model.tag.version, "predictions": preds[0]}
