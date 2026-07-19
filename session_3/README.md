# Ride Duration — Orchestration & Deployment (Session 3)

Session 2 stopped at a *trained, registered* model. **Session 3 is about everything
that happens after the model exists**: scheduling the pipeline that keeps it fresh,
choosing how predictions reach consumers, proving the service survives real
traffic, and serving it behind a production API.

The model itself is unchanged — the same synthetic **ride-duration** regressor
(a scikit-learn `RandomForest`, `models/rf_model.pkl`) carried over from session 2.
What changes is the machinery around it.

**This README follows the order you'd actually build the stack:**

1. [Orchestration with Airflow](#1-orchestration-with-airflow) — schedule and retry the retraining pipeline
2. [Deployment strategies](#2-deployment-strategies) — [web service](#the-three-shapes), [batch scoring](#21-batch-scoring), [streaming inference](#22-streaming-inference)
3. [Load testing with Locust](#3-load-testing-with-locust) — find the breaking point before users do
4. [BentoML serving](#4-bentoml-serving) — model → production API → Docker image
5. [TorchServe](#5-torchserve) — the PyTorch-native serving path

## Project structure

```
session_3/
├── docker-compose.yaml               # Airflow 3.x local stack (LocalExecutor, Postgres)
├── Dockerfile                        # extends apache/airflow with the DAG runtime deps
├── requirements.txt                  # deps baked into the Airflow image
├── .env                              # Airflow local-dev env (UID, Fernet key, admin login)
├── pyproject.toml                    # project deps + optional extras (tracking/bentoml/torch/load)
├── Dags/
│   └── example_1.py                  # retraining DAG: extract → train → evaluate
├── src/
│   ├── config.py                     # YAML config loader
│   ├── prepare.py                    # seeds data/processed/train.parquet (synthetic)
│   ├── train.py                      # fits the RandomForest → models/rf_model.pkl
│   ├── register_model.py             # logs the model to MLflow + moves the @production alias
│   └── batch_scoring.py              # ── batch strategy: parquet in → predictions out
├── bentoml_example.py                # BentoML service (save → serve → build → containerize)
├── torch_serve.py                    # TorchScript export + custom handler + .mar packaging
├── locustfile.py                     # load-test scenario (weighted /predict vs /health)
├── config/config.yaml                # data + model hyperparameters
├── models/rf_model.pkl               # the DVC-tracked model (from session 2)
└── data/, logs/, plugins/            # mounted into the Airflow containers
```

## Setup

```bash
# Base training/scoring stack
pip install -e .

# Add only the extras you need for the section you're on:
pip install -e ".[tracking]"   # MLflow registry (batch scoring / BentoML load from it)
pip install -e ".[bentoml]"    # BentoML serving
pip install -e ".[torch]"      # TorchServe + torch-model-archiver
pip install -e ".[load]"       # Locust
pip install -e ".[airflow]"    # only to parse/lint DAGs locally; they RUN in the image
```

Python **3.10–3.12** (Airflow 3.x and torch have no 3.13 wheels yet).

---

## 1. Orchestration with Airflow

### What it is

**Apache Airflow** is a workflow orchestrator. A model isn't a one-off script —
it's a *recurring pipeline*: pull fresh data → train → evaluate → promote if
better. Airflow turns that chain into a **DAG** (Directed Acyclic Graph) of tasks
with a schedule, and takes over the boring, critical parts: running it on time,
retrying the flaky step, showing you which task failed and why, and never
re-running a task whose upstream dependency didn't succeed.

[`Dags/example_1.py`](Dags/example_1.py) is the retraining DAG: `extract` pulls
new rides, `train` runs training inside an MLflow run and pushes the run ID to
**XCom**, `evaluate` pulls that run ID back and compares the metric against the
current production model.

```python
dag = DAG(
    dag_id="ride_duration_retrain",
    schedule="@weekly",           # every Monday 00:00 UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,                # don't backfill every missed week since 2024
    default_args={"retries": 2, "retry_delay": timedelta(minutes=5)},
)

t_extract >> t_train >> t_evaluate   # the dependency chain
```

### Why use it

- **Scheduling without cron sprawl.** One `schedule="@weekly"` instead of a crontab entry nobody can find.
- **Retries and alerting are free.** `retries: 2, retry_delay: 5min` handles the transient timeout that would otherwise page you.
- **Dependencies are explicit.** `train` cannot run if `extract` failed. A cron chain of scripts gives you no such guarantee.
- **Observability.** The UI shows every run, every task, every log line, and lets you re-run a single failed task instead of the whole pipeline.
- **Backfills.** Flip `catchup=True` and Airflow replays history for you.
- **XCom passes small values between tasks** (here: the MLflow run ID) without you inventing a temp-file protocol.

### The local stack

[`docker-compose.yaml`](docker-compose.yaml) runs Airflow 3.x with the
**LocalExecutor** — tasks are subprocesses of the scheduler, so there's no
Redis/Celery/Flower to babysit. Services: `postgres` (metadata DB),
`airflow-apiserver` (UI + REST API — Airflow 3 renamed the old *webserver*),
`airflow-scheduler`, `airflow-dag-processor` (a standalone component in Airflow 3),
and `airflow-triggerer`.

The DAG's runtime deps (`mlflow`, scikit-learn, pandas, …) are
baked into the image by [`Dockerfile`](Dockerfile) + [`requirements.txt`](requirements.txt),
rather than reinstalled on every container start via `_PIP_ADDITIONAL_REQUIREMENTS`.

### Most important commands

```bash
# ── Start the stack ───────────────────────────────────
docker compose build                     # build the extended Airflow image
docker compose up -d                     # postgres + apiserver + scheduler + dag-processor + triggerer
open http://localhost:8080               # login: airflow / airflow (from .env)

docker compose ps                        # service health
docker compose logs -f airflow-scheduler # follow the scheduler
docker compose down -v                   # stop and wipe the metadata DB

# ── Seed the local training data (once) ───────────────
docker compose run --rm airflow-cli python src/prepare.py

# ── Airflow CLI (the `debug` profile) ─────────────────
docker compose --profile debug run --rm airflow-cli airflow dags list
docker compose --profile debug run --rm airflow-cli airflow dags list-import-errors
docker compose --profile debug run --rm airflow-cli airflow tasks list ride_duration_retrain

# Unpause and trigger a run by hand
docker compose --profile debug run --rm airflow-cli airflow dags unpause ride_duration_retrain
docker compose --profile debug run --rm airflow-cli airflow dags trigger ride_duration_retrain

# Test ONE task in isolation — no scheduler, no DB state written
docker compose --profile debug run --rm airflow-cli \
  airflow tasks test ride_duration_retrain train 2024-01-01
```

> `airflow tasks test` is the command you'll actually live in while developing a
> DAG: it executes a single task's Python immediately and prints the traceback.

---

## 2. Deployment strategies

### The three shapes

Before writing any serving code, answer one question: **how fresh does a
prediction need to be?** The honest answer decides the architecture, and it is
usually *far less fresh* than people assume.

| | **Batch scoring** | **Streaming inference** | **Web service (online / REST)** |
|---|---|---|---|
| **Latency** | hours / days | seconds | milliseconds |
| **Trigger** | schedule (Airflow) | an event arrives | a user request |
| **Cost** | lowest — one job, no idle server | low — pay per invocation | highest — always-on |
| **Use when** | predictions are read from a table | reacting to events as they happen | a user is waiting for the answer |
| **In this repo** | [`src/batch_scoring.py`](src/batch_scoring.py) | *(conceptual — see §2.2)* | [BentoML §4](#4-bentoml-serving) / [TorchServe §5](#5-torchserve) |

The **web service** column — a synchronous REST endpoint a client calls and waits
on — is what sections [4](#4-bentoml-serving) and [5](#5-torchserve) build. This
section covers the other two columns: batch and streaming.

### 2.1 Batch scoring

#### What it is

Score a whole file of rows on a schedule and write the predictions somewhere your
consumers already read from (a bucket, a warehouse table). Nobody calls an API;
they `SELECT` yesterday's predictions.

[`src/batch_scoring.py`](src/batch_scoring.py) reads an input parquet, loads the
**production** model from the MLflow registry *by alias*, scores every row, stamps
a `run_date`, and writes the result back out as parquet (to a local path or any
object store your consumers read from).

The model gets its `@production` alias from [`src/register_model.py`](src/register_model.py),
which logs the trained model and then moves the alias onto the new version. (MLflow 3
removed the old `Production`/`Staging` *stages*, so an **alias** is now the supported
mechanism — that's why it's `models:/RideDurationModel@production`, not `.../Production`.)

```python
def load_production_model():
    """Always load from the MLflow registry by alias — no hardcoded paths."""
    return mlflow.sklearn.load_model("models:/RideDurationModel@production")

df["pred"]     = model.predict(features)
df["run_date"] = date.today().isoformat()
df.to_parquet(output_path)
```

#### Why use it

- **Cheapest thing that works.** No server sitting idle at 3 a.m. waiting for traffic.
- **Throughput over latency.** One vectorised `model.predict()` over 10M rows beats 10M HTTP round-trips by orders of magnitude.
- **Trivially retryable.** The job is idempotent — same input file, same output file. Re-run it and nothing breaks.
- **Failure is not an outage.** A late batch is an inconvenience; a down API is an incident.
- **The registry decouples deploys.** Moving the `@production` alias onto a new model version changes tomorrow's predictions with *zero* code changes — note the deliberate absence of a hardcoded model path.

#### Most important commands

```bash
# Run the scoring job locally (needs the [tracking] extra + an MLflow server)
export MLFLOW_TRACKING_URI=http://localhost:5000

python src/register_model.py            # (re)train + move the @production alias
python src/batch_scoring.py             # score the input parquet → output parquet

# Inspect input/output
ls -lh data/scoring/input/
ls -lh data/scoring/output/
```

In production you don't run this by hand — you make it the last task of the
Airflow DAG from section 1:

```python
PythonOperator(task_id="batch_score", python_callable=batch_score, dag=dag)
```

### 2.2 Streaming inference

#### What it is

An event lands on a message topic (Kafka, RabbitMQ, a cloud pub/sub); a consumer
wakes up, scores it, and writes the prediction to a store. No polling, no fixed
schedule — the *event* is the trigger. This is the middle ground between batch
(too slow to react) and an always-on web service (a server you pay for even when
idle).

The consumer is a small loop: pull a message, decode it, run `model.predict`,
persist the result. The pattern that matters is **load the model once**, not per
message:

```python
MODEL = None   # module-level: loaded ONCE per process, not per message

def get_model():
    global MODEL
    if MODEL is None:                                  # first message only
        MODEL = joblib.load("models/rf_model.pkl")
    return MODEL

def handle(message):
    payload    = json.loads(message)
    features   = [[payload["distance_km"], payload["passengers"]]]
    prediction = float(get_model().predict(features)[0])
    save({"ride_id": payload["ride_id"], "prediction": round(prediction, 2)})
```

That `global MODEL` is the single most important line. Loading the model inside
the handler would re-load it **on every message**. Loading it once means the warm
process reuses it, and only startup pays the cost.

#### Why use it

- **Reacts in seconds, not hours** — the gap batch scoring can't close.
- **Producer and consumer are decoupled.** The app publishes a ride event and moves on. If the scorer is down, messages queue on the broker instead of erroring out.
- **The broker gives you back-pressure and retries.** Run more consumers to scale throughput; a raised exception means the message is *redelivered*, not lost.
- **No fixed capacity to size.** Consumers process at their own pace; the queue absorbs bursts.

> A concrete cloud implementation of this (a Pub/Sub-triggered function) lived here
> in an earlier version and was removed along with the rest of the GCP/Terraform
> material. The pattern above is what any broker-backed consumer looks like.

---

## 3. Load testing with Locust

### What it is

Your API answers one request in 40 ms. What does it do with **500 concurrent
users**? *"It'll probably be fine"* is a guess, and it's the guess that turns into
an incident. **Locust** answers it empirically: you describe one simulated user in
Python, it spawns thousands of them and reports latency percentiles and failure
rates.

[`locustfile.py`](locustfile.py) simulates a client that hits `/predict` far more
often than `/health`, with realistic think-time between calls, and validates the
*content* of each response — not just its status code.

```python
class MLAPIUser(HttpUser):
    wait_time = between(0.5, 2.0)      # think-time, so the load is realistic

    @task(weight=10)                    # 10x more common than /health
    def predict(self):
        payload = {"distance_km": ..., "passengers": ...}
        with self.client.post("/predict", json=payload, catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"Unexpected status {resp.status_code}")
            elif resp.json().get("duration_min", -1) < 0:
                resp.failure("Negative duration in response")   # a 200 that's still wrong
```

### Why use it

- **Scenarios are plain Python.** Random payloads, weighted task mixes, auth flows, stateful sessions — anything you can write, you can simulate. No XML, no clicking through a GUI recorder.
- **`catch_response` catches *semantic* failures.** A `200 OK` carrying a negative duration is a bug; Locust marks it as a failure. HTTP-only tools would report 100% success.
- **Percentiles, not averages.** p95/p99 is what your users feel. The mean hides the tail.
- **Finds the knee in the curve.** Ramp users up until p95 latency spikes — that's your real capacity, and now you can size the VM in section 6 with a number instead of a vibe.
- **`--headless --csv` makes it a CI gate.** Fail the build if p95 regresses.

> Point the check at the status code your server actually returns — FastAPI and
> BentoML default to `200`; adjust the assertion if your endpoint returns `201`.

### Most important commands

```bash
# ── Interactive: web UI at http://localhost:8089 ──────
locust -f locustfile.py --host http://localhost:8000

# ── Headless: scripted run, results to CSV ────────────
locust -f locustfile.py --host http://localhost:8000 \
  --users 100 \          # total concurrent users
  --spawn-rate 10 \      # add 10 users/sec until 100
  --run-time 2m \        # then stop
  --headless \
  --csv=results/load_test

# ── CI gate: fail the build on regressions ────────────
locust -f locustfile.py --host http://localhost:8000 \
  --users 200 --spawn-rate 20 --run-time 3m --headless \
  --only-summary \
  --exit-code-on-error 1
```

Read the summary right to left: **failure count first** (any non-zero means stop
tuning and start fixing), then **p95/p99**, then RPS. A high RPS with a 12-second
p99 is not a passing test.

---

## 4. BentoML serving

### What it is

Between `model.pkl` and *a production API* sits a pile of undifferentiated work:
an HTTP layer, request validation, batching, health checks, a Dockerfile, the
right base image. **BentoML** generates all of it from a model reference and a
Python function — this is the **web service** shape from the strategy table.

[`bentoml_example.py`](bentoml_example.py) walks the full path: pull the production
model out of the MLflow registry, save it into BentoML's model store, wrap it in a
Service, then package it as a Docker image.

```python
bento_model = bentoml.sklearn.save_model(
    "ride_duration", sk_model,
    signatures={"predict": {"batchable": True, "batch_dim": 0}},   # ← adaptive batching
)

runner = bentoml.sklearn.get("ride_duration:latest").to_runner()
svc    = bentoml.Service("ride_api", runners=[runner])

@svc.api(input=bentoml.io.JSON(pydantic_model=PredictRequest),
         output=bentoml.io.JSON(pydantic_model=PredictResponse))
async def predict(req: PredictRequest) -> PredictResponse:
    result = await runner.predict.async_run(features)
    return PredictResponse(duration_min=float(result[0]))
```

### Why use it

- **Adaptive batching for free.** `"batchable": True` makes BentoML collect concurrent requests into one `predict()` call — the single biggest throughput win available to an ML API, and it's one dict key. Re-run section 3's Locust test with it on and off; the difference is not subtle.
- **Pydantic in, Pydantic out.** Malformed requests are rejected with a clean 422 before they ever touch the model.
- **Runners scale independently of the API.** Model inference runs in its own worker process, so a slow prediction doesn't block the event loop serving `/health`.
- **`bentoml containerize` writes the Dockerfile.** Correct Python version, correct deps, correct model — no hand-maintained image.
- **The model store is versioned.** Every `save_model` gets a tag (`ride_duration:abc123`); rollback is redeploying a previous tag.
- **Batteries included:** OpenAPI docs at `/docs`, Prometheus metrics at `/metrics`, health checks — all out of the box.

### Most important commands

```bash
# ── Save the model into the BentoML store ─────────────
python bentoml_example.py
bentoml models list                      # see the tags
bentoml models get ride_duration:latest

# ── Dev server (hot reload) ───────────────────────────
bentoml serve bentoml_example:svc --reload
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"distance_km": 8.4, "passengers": 2}'

# ── Package & containerize ────────────────────────────
bentoml build                            # → a versioned Bento
bentoml list                             # list built Bentos
bentoml containerize ride_api:latest     # → a Docker image

docker run -p 3000:3000 ride_api:latest  # ready for section 6
```

---

## 5. TorchServe

### What it is

The PyTorch-native serving path, maintained alongside PyTorch itself. Where
BentoML is framework-agnostic, **TorchServe** is built around one assumption —
your model is a `torch.nn.Module` — and gets to specialise hard on that:
TorchScript compilation, GPU batching, multi-model hosting, and model versioning
in one binary.

[`torch_serve.py`](torch_serve.py) shows the four steps:

**1. Export to TorchScript** — a serialized graph the server can load without your Python class:
```python
scripted = torch.jit.script(model)
scripted.save("ride_duration.pt")
```

**2. Write a handler** — the only glue you own: JSON → tensor, tensor → JSON:
```python
class RideHandler(BaseHandler):
    def preprocess(self, data):
        rows  = [json.loads(d["body"]) for d in data]     # `data` is a BATCH
        feats = [[r["distance_km"], r["passengers"]] for r in rows]
        return torch.tensor(feats, dtype=torch.float32)

    def postprocess(self, output):
        return [{"duration_min": round(v.item(), 2)} for v in output.squeeze()]
```
Note the shape of `preprocess`: it receives a **list** of requests, not one.
TorchServe batched them for you, and the handler is written to stay vectorised.

**3–4. Archive and serve** — bundle weights + handler into one `.mar` file, then start the server.

### Why use it

- **TorchScript decouples serving from your source tree.** The `.pt` file carries the graph; the server doesn't need to import your model class.
- **Server-side batching is built in** (`batch_size` / `max_batch_delay`) — same throughput win as BentoML's adaptive batching, tuned for GPU.
- **The `.mar` is a single versioned deployable.** Weights, handler, and config in one artifact you can store in a registry.
- **Multi-model, one server.** Register N models on one GPU and route by URL — you don't pay for a GPU per model.
- **Management API at :8081** — register, scale workers, and unregister models at runtime, no restart.
- **First-class GPU support**, plus a metrics endpoint on :8082.

**Pick TorchServe over BentoML** when you're all-in on PyTorch, need GPU batching,
or want to host several models on one box. **Pick BentoML** when you have
scikit-learn/XGBoost/mixed frameworks or want the shortest path to a Docker image.

### Most important commands

```bash
# ── 1. Export TorchScript ─────────────────────────────
python torch_serve.py                    # → ride_duration.pt

# ── 2. Package into a .mar archive ────────────────────
torch-model-archiver \
  --model-name ride_duration \
  --version 1.0 \
  --serialized-file ride_duration.pt \
  --handler handler.py \
  --export-path model_store/

# ── 3. Start / stop the server ────────────────────────
torchserve --start \
  --model-store model_store/ \
  --models ride_duration=ride_duration.mar \
  --ts-config config.properties
torchserve --stop

# ── 4. Inference (:8080) ──────────────────────────────
curl -X POST http://localhost:8080/predictions/ride_duration \
  -H "Content-Type: application/json" \
  -d '{"distance_km": 8.4, "passengers": 2}'

# ── Management API (:8081) — no restart needed ────────
curl http://localhost:8081/models                                   # list
curl -X POST "http://localhost:8081/models?url=ride_duration.mar"   # register
curl -X PUT  "http://localhost:8081/models/ride_duration?min_worker=4"  # scale workers
curl -X DELETE http://localhost:8081/models/ride_duration/1.0       # unregister

# ── Metrics (:8082) ───────────────────────────────────
curl http://localhost:8082/metrics
```

---

## Where this leaves you

You now have the deployment shapes for the same model, and the tools to choose
between them: **Airflow** schedules the retraining, **batch scoring** serves the
cheap/high-throughput case, **streaming inference** serves the event-driven case,
**BentoML/TorchServe** serve the online web-service case, and **Locust** tells you
which of them will hold up under real traffic.

The lesson worth keeping: the model was the easy part.
