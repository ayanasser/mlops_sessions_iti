# Model serving, level by level

Standalone, runnable scripts showing the same models served six different ways —
from a notebook cell to a C++ inference server. Every number in this file was
**measured on the machine described at the bottom**; nothing is copied from a blog
post, and every claim that could *not* be measured here (GPU, TensorRT, Triton) is
labelled as such.

## The one idea

Serving has **three layers**. Most teams only realise they own all three after
something falls over in production.

| Layer | Responsibility | Who owns it |
|---|---|---|
| **1 — Transport** | HTTP, routing, validation, docs | FastAPI, from Level 1 |
| **2 — Scheduling** | batching, queueing, backpressure, versioning | **you**, until Level 4 |
| **3 — Runtime** | graph optimization, precision, device concurrency | **you**, until Level 3 |

> FastAPI is not wrong. It is **incomplete**. It owns layer 1 and leaves 2 and 3 to you.

---

## Files

**One model — ResNet-50 — served six ways.** Every level uses the same weights, the
same image and the same `bench.py`, so any difference you measure is the serving
architecture and nothing else. Run them top to bottom.

| Level | Serving approach | File | Port | Runs here? |
|---|---|---|---|---|
| 0 | In-process, no server | `level_0_notebook.py` | — | yes |
| 1 | FastAPI, one request at a time | `level_1_fastapi.py` | 8001 | yes |
| 2 | FastAPI + hand-written dynamic batcher | `level_2_batching.py` | 8002 | yes |
| 3 | Runtime optimization (compile / ONNX / INT8) | `level_3_optimize.py` | — | yes |
| 4 | BentoML | `level_4_save_model.py` → `level_4_bentoml_service.py` | 8004 | yes |
| 4 | Triton (NVIDIA Dynamo-Triton) | `level_4_triton_export.py` → `level_4_triton/` | 8000 | export yes, serve needs amd64 |

Plus `bench.py`, the concurrent load generator used for every measurement below.

Use the session venv: `../.venv/bin/python`, `../.venv/bin/uvicorn`, `../.venv/bin/bentoml`.

---

# Level 0 — The notebook

```bash
python level_0_notebook.py
```

Correct prediction (`Samoyed 0.4554`), then the diagnostics that matter:

```
model load       0.37 s    paid again on every kernel restart
forward pass    29.47 ms   the only part that is "the model"
batch size          1      the GPU wants 32-64
callable by         1 person over 0 network interfaces
process         pid 9934   dies with this terminal
```

**Why anyone does this:** it is the fastest way to know whether the model works at
all. Exploration should be cheap.

**Why you cannot ship it:** the failure is not accuracy — the model is fine. The
prediction lives in one kernel session on one laptop, and nothing else in the world
can ask for one.

**Use when:** exploring, debugging, teaching. Never for anything with a caller.

---

# Level 1 — FastAPI alone

```bash
uvicorn level_1_fastapi:app --port 8001
python bench.py --port 8001 -c 16 -n 60
curl -s localhost:8001/metrics
```

**Benefits — real ones:** a genuine HTTP endpoint anyone on the network can call,
automatic OpenAPI docs at `/docs`, Pydantic request validation, async I/O. **For many
projects this is genuinely enough**, and stopping here is a legitimate engineering
decision, not laziness.

**Measured, 16 concurrent clients:**

```
throughput      35.3 req/s
p50            420.0 ms
p99           1162.6 ms
inference_batch_size_avg  1.00      <- problem 1
preprocess_seconds_avg    0.01209   <- problem 2: 12.1ms preprocessing ...
forward_seconds_avg       0.01659   <-            ... vs 16.6ms of model
```

**Why it is not enough — four problems, each visible on `/metrics`:**

1. **Batch size is always 1.** A GPU is built for 32–64 images at once; utilization
   sits at 8–15%. You are paying for a GPU and using a tenth of it.
2. **Preprocessing is on the request path.** JPEG decode + resize + normalize costs
   12.1 ms against 16.6 ms for the model itself. The handler is `async` but does
   blocking CPU work, so it holds the event loop that should be accepting connections.
3. **No model versioning.** `MODEL_VERSION` is a string constant: v2 means a restart,
   rollback means a redeploy, a canary means two services.
4. **Eager FP32 PyTorch.** Op by op, no fusion, through the Python interpreter.

**Use when:** low or predictable traffic, CPU inference, internal tools, or a
prototype whose latency budget is generous. Move on when the GPU bill or p99 hurts.

---

# Level 2 — FastAPI + a hand-written dynamic batcher

```bash
uvicorn level_2_batching:app --port 8002 --workers 1   # workers MUST be 1
python bench.py --port 8002 -c 16 -n 60
```

```
Request A -+
Request B -+--> [ queue: wait 8ms OR until N items ] --> model(batch) --> split
Request C -+
```

**Why this is the highest-leverage change available:** it is the only fix that
addresses *device utilization* rather than shaving constants. The device was idle
waiting for work; batching gives it work.

**Measured — same load, same hardware, same model:**

| | Level 1 | Level 2 | Change |
|---|---|---|---|
| throughput | 35.3 req/s | **58.8 req/s** | **1.67×** |
| p50 | 420 ms | 246 ms | 1.7× better |
| p99 | 1163 ms | **331 ms** | **3.5× better** |
| avg batch size | 1.00 | 5.64 | — |

### Honest caveat on the "5–10×" claim

On a **GPU**, dynamic batching commonly buys 5–10×. On the **CPU** most laptops have,
we measured **1.67×**. That gap is the mechanism, not a broken demo:

> Batching is a **device-utilization** fix. It wins when the device is *idle*. A GPU at
> 8–15% has enormous headroom; a CPU is already saturated by one image, so a batch of
> 32 is genuinely 32× the work.

`level_3_optimize.py` measures this directly: eager PyTorch at batch 32 runs at
**0.20× the per-image throughput of batch 1** — a memory-bandwidth cliff (~2.5 s per
batch). Hence `MAX_BATCH_SIZE = 32 if CUDA else 8`. Not a style choice.

Note *where* the CPU win comes from: mostly the **tail**. Batching serialises access
to the model instead of letting 16 concurrent forward passes fight over the same
threads, so p99 improves far more than throughput.

**What it cost:** the ~90 lines between the `SERVING LAYER` markers, each hard part
labelled inline:

- `[COST-1]` async queue with a timeout window
- `[COST-2]` correlating each response back to the right request (futures)
- `[COST-3]` variable input shapes — fine for ResNet, breaks for detection/ASR
- `[COST-4]` backpressure when the queue outruns the device
- `[COST-5]` partial batch failure — if the batcher task dies, every later request
  hangs forever on a future nobody will resolve

Two classic bugs are pre-seeded as comments: binding the future to the wrong event
loop, and running the forward pass *on* the event loop instead of a threadpool.

> This is where self-rolled serving stacks develop their worst bugs — and it is
> exactly what every purpose-built framework gives you as a config line.

**Use when:** you need the batching win but cannot adopt a framework (unusual
runtime, hard dependency limits, or you must understand it before you buy it).

---

# Level 3 — Optimize the runtime

```bash
python level_3_optimize.py          # --quick skips the INT8 build
```

Three independent levers, **multiplicative** with Level 2's batching win:

| Lever | What it does | Typical gain | Accuracy cost |
|---|---|---|---|
| **A — Graph optimization** | fuse Conv+BN+ReLU, fold constants, plan memory | 2–3× | none |
| **B — Reduced precision** | FP32 → FP16 → INT8 | 2–4× | <1% top-1 for ResNet PTQ |
| **C — Device concurrency** | 2–4 model copies overlapping compute and transfer | 20–50% | none |

**Measured, images/sec, this CPU:**

| variant | bs=1 | bs=8 | bs=32 |
|---|---|---|---|
| pytorch eager fp32 | 60.0 | 81.5 | 11.9 |
| + torch.compile | 74.1 | 93.0 | 12.8 |
| onnxruntime fp32 | 44.8 | 28.6 | 37.5 |
| **onnxruntime int8** | **80.7** | 78.2 | **67.3 (5.7×)** |

Read the **columns**, not just the rows. The striking result is `bs=32`: eager
PyTorch **collapses** to 11.9 img/s while ONNX INT8 holds 67.3. **The runtime you
choose decides whether batching helps you at all.**

`channels_last` is applied **only on CUDA** here. On CPU it is a ~2× *pessimization*.
A lever that helps on one device and hurts on another is the normal case — which is
why you measure on your target hardware instead of copying a flag list.

On an NVIDIA GPU the ladder continues past what this machine can run:

```
pytorch eager fp32   ->  baseline
+ onnxruntime        ->  ~2x      (works on CPU too)
+ tensorrt fp16      ->  ~4x
+ tensorrt int8      ->  ~7-10x
```

### The punchline

```
JPEG decode + resize + normalize    11.49 ms   1 image, Python
forward pass, onnxruntime int8      12.40 ms   fastest runtime here
preprocessing is 0.9x the cost of the model
```

We spent all of Level 3 optimizing the forward pass, and preprocessing now costs
about as much as the entire model. On a GPU with TensorRT the forward drops to ~3 ms
while that JPEG decode stays ~12 ms — the model becomes a rounding error.

> **Optimize what is actually slow, not what you assume is slow.**

Fixes: NVIDIA DALI, GPU-side decode (nvJPEG), or moving preprocessing into a pipeline
stage that scales independently of the accelerator.

**Use when:** always, eventually. Lever A is free accuracy-wise and should be default.
Lever B needs a calibration set and an accuracy check. Lever C needs a serving runtime
that supports it (Level 4).

---

# Level 4 — BentoML (ResNet-50, images)

```bash
python level_4_save_model.py
bentoml serve level_4_bentoml_service:ResNet50 --port 8004
python bench.py --port 8004 --field images -c 16 -n 60
```

| Level 2, by hand | Level 4, BentoML |
|---|---|
| ~90 lines of serving layer | `batchable=True` |
| queue, window, futures, backpressure | `max_batch_size=8` |
| partial-batch failure handling | `max_latency_ms=<SLA>` |
| `uvicorn --workers 1` | `workers=1` |
| hand-rolled `/metrics` | Prometheus, free |
| `MODEL_VERSION = "v1"` | model store, hot-swappable |
| write your own Dockerfile | `bentoml build && bentoml containerize` |
| `UploadFile` + `io.BytesIO` + try/except | annotate the param `PIL.Image` |

**Measured: 39.7 req/s, p50 390 ms, p99 468 ms, 60/60 successful.**

### Two things worth being honest about

**1. BentoML is slower here than our hand-written Level 2** (39.7 vs 58.8 req/s).
Real and reproducible: the framework pays for multipart parsing, schema validation and
per-request tracing, and in this service preprocessing runs serially inside the batched
call while Level 2 farms it to a threadpool. You are trading raw throughput on a toy
benchmark for correctness, observability, versioning and packaging you did not write
and do not maintain. On a GPU — where batching is 5–10×, not 1.7× — that trade looks
very different.

**2. Two config knobs silently destroy the thing you configured them for.** Both were
hit while building this, both are commented in the source:

- `max_latency_ms` is **not** Level 2's `BATCH_WINDOW_MS`. It is a latency **SLA**.
  BentoML measures how long batches actually take, tunes the wait window itself, and
  *sheds* requests it predicts will miss the deadline. Setting it to `8` "to match
  Level 2" made the server 503 **51 of 60 requests** — a batch of 8 takes ~100 ms on CPU.
- `traffic.concurrency` is the admission limit and must be **≥ `max_batch_size`**, or
  requests are rejected before enough accumulate to fill a batch, and you have paid for
  a batching system that never batches.

You declare the deadline; the framework picks the window. That is the real shift in
thinking at Level 4 — and it is why the batchable API must **be** the HTTP endpoint:
batching is applied at request dispatch, so calling a batchable method internally from
another handler gets you no batching at all.

---# Level 4 — Triton (the same ResNet-50, as configuration)

```bash
python level_4_triton_export.py     # ONNX export, verified against PyTorch
```

BentoML served the PyTorch module directly. Triton wants a portable graph, so the
model is exported to ONNX first — that conversion step is the main difference in
effort between the two, and what you get for it is a C++ server with no GIL anywhere
on the request path.

The export **verifies itself at three batch sizes**, because checking only `bs=1`
would miss a broken dynamic batch axis — the most common way this export goes wrong:

```
  batch    max abs diff   top-1 agree
      1        3.10e-06          True
      8        5.13e-06          True
     32        6.91e-06          True
conversion verified across batch sizes
```

A silently-wrong export is a nasty production bug: healthy server, great latency,
subtly wrong answers. Always diff against the source model before shipping.

## The model repository *is* the versioning system

```
level_4_triton/model_repository/
  resnet50/
    config.pbtxt
    1/model.onnx        <- written by the export script
    2/model.onnx        <- next version, hot-loaded without a restart
```

## `config.pbtxt` is the entire serving layer

| Level 2, by hand | `config.pbtxt` |
|---|---|
| `asyncio.Queue` + window + futures (~90 lines) | `dynamic_batching { ... }` |
| `MAX_BATCH_SIZE` | `preferred_batch_size` |
| `BATCH_WINDOW_MS = 8` | `max_queue_delay_microseconds: 8000` |
| `QUEUE_MAXSIZE` + 503 | `default_queue_policy { max_queue_size, REJECT }` |
| `MODEL_VERSION = "v1"` constant | `version_policy` + version directories |
| not possible without a rewrite | `instance_group { count: 2 }` (Lever C) |
| a separate export + rebuild | `optimization { ... tensorrt FP16 }` (Levers A + B) |

Prometheus metrics come free on `:8002/metrics`, and
`nv_inference_request_success / nv_inference_exec_count` **is** your average batch
size — the number Level 2 had to compute by hand.

## It does not run on this machine, and that is stated honestly

**Triton images are `linux/amd64` only; this machine is `aarch64` (Apple Silicon).**
The ~15 GB image was deliberately **not** pulled — that is your disk and bandwidth,
and under emulation with no GPU it would be useless for benchmarking anyway. Run it
on a Linux x86 box or a cloud GPU VM.

Everything Triton-specific here is nonetheless correct and ready: the `config.pbtxt`
tensor names and shapes were **cross-checked against the actual export**
(`input [3,224,224]` → `output [1000]`, batch dim implicit), and the weights were
verified against PyTorch. See [`level_4_triton/README.md`](level_4_triton/README.md)
for the request format and the full comparison.

## BentoML or Triton?

`level_4_bentoml_service.py` serves this model in ~60 lines of Python with no
conversion step. Choose Triton when you need what it uniquely gives you:

- **No GIL on the request path** — the server is C++
- **Multiple frameworks in one process** — TensorRT + ONNX + PyTorch behind one endpoint
- **Model ensembles** — preprocess → infer → postprocess as a server-side DAG, which
  is the proper fix for the preprocessing bottleneck Level 3 uncovered
- **Concurrent model execution** — `instance_group.count`, i.e. Lever C
- **Hot reload** — drop in a `2/` directory, no restart

For one model in a Python shop, BentoML is the better trade. For a GPU fleet serving
several models under a latency SLA, Triton is what the trade was designed for.

---

# Every serving approach: what it buys, and when to use it

| Approach | Key benefit | Main cost | Use when |
|---|---|---|---|
| **Notebook / in-process** | zero setup, instant feedback | not reachable, not reproducible | exploring, debugging |
| **Batch / offline scoring** | no latency budget at all; simplest thing that works | results are stale by design | predictions can be precomputed (see `src/batch_scoring.py`) |
| **FastAPI alone** | real endpoint, docs, validation, tiny dependency surface | batch size 1, no versioning, you own layers 2–3 | low/predictable traffic, CPU, internal tools |
| **FastAPI + manual batching** | the 5–10× GPU win, full control | ~90 lines of concurrency you now maintain and debug | you need batching but cannot adopt a framework |
| **Runtime optimization** (ONNX / TensorRT / OpenVINO / `torch.compile`) | 2–10× on the *same* hardware; often the only lever on CPU | export step, a runtime to validate, possible accuracy loss | always, eventually — start with Lever A |
| **TorchServe** | PyTorch-native, `.mar` archives, management API | **maintenance mode** — a stepping stone, not a destination | existing TorchServe estate only |
| **BentoML** | Python-native; batching, versioning, Prometheus, containerization included; multi-model composition | some throughput overhead; framework opinions | most Python teams shipping one or a few models — **the default recommendation** |
| **Triton / NVIDIA Dynamo-Triton** | C++, no GIL; many backends in one process; ensembles; concurrent execution; hot reload | conversion step, `config.pbtxt`, heavy runtime | GPU fleets, multiple models, strict latency SLAs |
| **vLLM / TGI (LLM-specific)** | continuous batching + PagedAttention for autoregressive decode | LLM-only | serving an LLM — see `../serve_llm_vllm_example.py` |
| **Serverless (Cloud Run / Functions)** | scale to zero, no servers to run | cold starts, no GPU batching, time limits | spiky or low traffic; event-driven scoring |
| **Nginx / gateway canary** | traffic splitting, A/B, gradual rollout | another component to operate | validating v2 against v1 — see `../nginx.conf`, `../ab_testing.py` |

## Why classic batching does not transfer to LLMs

Levels 2–4 batch safely because **every request costs the same compute**: a batch of 32
images takes about as long as a batch of 1. That assumption breaks for autoregressive
models, where each sequence decodes a different number of steps and a fixed batch runs
at the speed of its slowest member. That is the problem vLLM's continuous batching and
PagedAttention exist to solve — a different scheduler, not a bigger batch.

## A decision path

1. **Can predictions be precomputed?** → batch scoring. Stop.
2. **Is traffic low and latency generous?** → FastAPI alone. Stop.
3. **Is the device underutilised at p99?** → add batching. Prefer a framework over
   hand-rolling it.
4. **Still too slow or too expensive?** → Level 3 runtime work, then re-measure.
   **Profile first** — it is often preprocessing, not the model.
5. **Several models, multiple frameworks, or a GPU fleet?** → Triton.
6. **Serving an LLM?** → vLLM/TGI, not any of the above.

---

# Summary of measurements

**ResNet-50, images, 60 requests at 16 concurrent:**

| Level | Throughput | p50 | p99 | Notes |
|---|---|---|---|---|
| 1 — FastAPI | 35.3 req/s | 420 ms | 1163 ms | batch always 1.00 |
| 2 — hand batcher | **58.8 req/s** | 246 ms | **331 ms** | avg batch 5.64 |
| 4 — BentoML | 39.7 req/s | 390 ms | 468 ms | + versioning, metrics, packaging |

Level 3 is offline (no server), so it is not in the table; its win **multiplies** with
whatever the table shows. Triton is not in the table either — see the platform caveat.

> **How to read a comparison like this.** Every row uses the same model, the same
> image, the same client and the same concurrency. That matters more than it sounds:
> comparing 1 client against 32 clients on the *same* service measures **concurrency
> scaling, not batching**, and it is the easiest way to convince yourself a change
> helped when it did not. If you change the load, you have not measured the change.

### Reproduction environment

Apple Silicon, macOS (Darwin 25.5), **CPU only — no CUDA**. Python 3.12, torch 2.13.0,
torchvision 0.28.0, onnxruntime 1.27.0, bentoml 1.4.39, scikit-learn 1.9.0,
skl2onnx 1.20.0, mlflow 3.14.0. 5 torch threads on 11 cores. Image: `assets/dog.jpg`
(1546×1213).

Numbers move ±15% run to run on a thermally throttled laptop. Every GPU-specific claim
(5–10× batching, the TensorRT ladder, `channels_last`, Lever C) is **marked as
unmeasured** rather than presented as a result. Re-run each level before quoting it.

### Dependencies added for these scripts

```bash
pip install onnxruntime onnx httpx bentoml skl2onnx
```

### Cleanup

```bash
lsof -tiTCP:8001 -tiTCP:8002 -tiTCP:8004 | xargs -r kill -9
rm -f assets/*.onnx                       # regenerated by the level 3 scripts
bentoml models list                       # delete: bentoml models delete <tag>
```

> Note: the parent `.gitignore` excludes `*.onnx`, so the exported artifacts (including
> `level_4_triton/model_repository/resnet50/1/model.onnx`) will not be committed. They
> are regenerated by `level_3_optimize.py` and `level_4_triton_export.py`, but add an
> exception if you want the Triton repository to be self-contained for students.
