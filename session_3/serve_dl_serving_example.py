"""
CLASSICAL DL SERVING — FastAPI + hand-written micro-batcher (ResNet50 image classifier)

Point of this file: with a single-forward-pass model, YOU own the GPU loop.
FastAPI only does HTTP. Everything below the line marked "SERVING LAYER"
is what TorchServe / BentoML / Triton would have given you for free.

Run:  uvicorn serve_dl_resnet:app --host 0.0.0.0 --port 8000 --workers 1
      ^ workers MUST be 1: the batcher holds in-process state + the GPU.
        Scale with replicas (pods), not uvicorn workers.
"""

import asyncio
import io
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import torch
import torchvision.transforms as T
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_BATCH_SIZE = 16          # cap: bigger batch = more throughput, worse tail latency
BATCH_WINDOW_MS = 8          # how long we're willing to wait to fill a batch
QUEUE_MAXSIZE = 256          # backpressure: reject rather than queue forever


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
class Classifier:
    def __init__(self) -> None:
        weights = ResNet50_Weights.DEFAULT
        self.categories = weights.meta["categories"]
        self.model = resnet50(weights=weights).eval().to(DEVICE)

        # graph-level optimization: channels_last + fp16 + compile
        self.model = self.model.to(memory_format=torch.channels_last)
        if DEVICE == "cuda":
            self.model = self.model.half()
            self.model = torch.compile(self.model, mode="reduce-overhead")

        self.preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def encode(self, raw: bytes) -> torch.Tensor:
        """CPU-side preprocessing. Runs in a threadpool, off the event loop."""
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return self.preprocess(img)          # -> [3, 224, 224]

    @torch.inference_mode()
    def predict_batch(self, tensors: list[torch.Tensor]) -> list[list[dict]]:
        """THE forward pass. One call, N images. This is the whole point of batching."""
        batch = torch.stack(tensors).to(DEVICE, memory_format=torch.channels_last)
        if DEVICE == "cuda":
            batch = batch.half()

        logits = self.model(batch)                       # [N, 1000]
        probs = torch.softmax(logits.float(), dim=-1)
        top5 = torch.topk(probs, k=5, dim=-1)

        out = []
        for scores, idxs in zip(top5.values.cpu(), top5.indices.cpu()):
            out.append([
                {"label": self.categories[i], "score": round(float(s), 4)}
                for s, i in zip(scores, idxs)
            ])
        return out


# ---------------------------------------------------------------------------
# SERVING LAYER  <-- everything from here to `app` is what a serving framework
#                    hands you out of the box. Written by hand to show the cost.
# ---------------------------------------------------------------------------
@dataclass
class Job:
    tensor: torch.Tensor
    # The future MUST come from the *running* loop. Binding it at import time
    # (asyncio.get_event_loop().create_future) attaches every future to a loop
    # uvicorn never runs -> "got Future attached to a different loop" on request 1.
    future: asyncio.Future = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )
    enqueued_at: float = field(default_factory=time.perf_counter)


@dataclass
class Metrics:
    requests: int = 0
    rejected: int = 0
    batches: int = 0
    batched_items: int = 0
    queue_time_s: float = 0.0
    infer_time_s: float = 0.0


class Batcher:
    """Dynamic batching: drain the queue every BATCH_WINDOW_MS, run one forward pass.

    Safe here ONLY because every request costs the same amount of compute.
    A batch of 16 images takes ~as long as a batch of 1. That assumption is
    exactly what breaks for autoregressive LLMs (see serve_llm_vllm.py).
    """

    def __init__(self, model: Classifier, metrics: Metrics) -> None:
        self.model = model
        self.metrics = metrics
        self.queue: asyncio.Queue[Job] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def submit(self, tensor: torch.Tensor) -> list[dict]:
        job = Job(tensor=tensor)
        try:
            self.queue.put_nowait(job)
        except asyncio.QueueFull:
            self.metrics.rejected += 1
            raise HTTPException(status_code=503, detail="overloaded, retry")
        return await job.future

    async def _collect(self) -> list[Job]:
        """Block for the first job, then greedily grab whatever arrives in the window."""
        jobs = [await self.queue.get()]
        deadline = time.perf_counter() + BATCH_WINDOW_MS / 1000
        while len(jobs) < MAX_BATCH_SIZE:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                jobs.append(await asyncio.wait_for(self.queue.get(), timeout=remaining))
            except asyncio.TimeoutError:
                break
        return jobs

    async def _loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            jobs = await self._collect()
            now = time.perf_counter()
            self.metrics.queue_time_s += sum(now - j.enqueued_at for j in jobs)

            t0 = time.perf_counter()
            try:
                # GPU work off the event loop, otherwise HTTP stalls during inference
                results = await loop.run_in_executor(
                    None, self.model.predict_batch, [j.tensor for j in jobs]
                )
                for job, res in zip(jobs, results):
                    if not job.future.done():
                        job.future.set_result(res)
            except Exception as exc:                      # one bad batch != dead server
                for job in jobs:
                    if not job.future.done():
                        job.future.set_exception(exc)

            self.metrics.infer_time_s += time.perf_counter() - t0
            self.metrics.batches += 1
            self.metrics.batched_items += len(jobs)


# ---------------------------------------------------------------------------
# HTTP LAYER — this is the only part FastAPI is actually responsible for
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.metrics = Metrics()
    app.state.model = Classifier()
    app.state.batcher = Batcher(app.state.model, app.state.metrics)
    app.state.batcher.start()
    # warm-up: trigger compile + cudnn autotune before traffic arrives
    await app.state.batcher.submit(torch.zeros(3, 224, 224))
    yield
    await app.state.batcher.stop()


app = FastAPI(title="ResNet50 classifier", lifespan=lifespan)


@app.post("/predict")
async def predict(file: UploadFile):
    raw = await file.read()
    loop = asyncio.get_running_loop()
    try:
        tensor = await loop.run_in_executor(None, app.state.model.encode, raw)
    except Exception:
        raise HTTPException(status_code=400, detail="could not decode image")

    app.state.metrics.requests += 1
    t0 = time.perf_counter()
    preds = await app.state.batcher.submit(tensor)
    return JSONResponse({
        "predictions": preds,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    })


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "device": DEVICE}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    m: Metrics = app.state.metrics
    avg_batch = m.batched_items / m.batches if m.batches else 0
    avg_queue = m.queue_time_s / m.batched_items if m.batched_items else 0
    avg_infer = m.infer_time_s / m.batches if m.batches else 0
    return (
        f"inference_requests_total {m.requests}\n"
        f"inference_rejected_total {m.rejected}\n"
        f"inference_batches_total {m.batches}\n"
        f"inference_batch_size_avg {avg_batch:.2f}\n"
        f"inference_queue_seconds_avg {avg_queue:.5f}\n"
        f"inference_forward_seconds_avg {avg_infer:.5f}\n"
    )