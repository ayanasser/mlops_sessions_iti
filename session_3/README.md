# Ride Duration — Orchestration & Deployment (Session 3)

Session 2 stopped at a *trained, registered* model. Session 3 is about everything
that happens **after** the model exists: scheduling the pipeline that keeps it
fresh, choosing how predictions reach consumers, packaging the service as a
container, and proving it survives real traffic.

The model itself is unchanged — the same synthetic **ride-duration** regressor
(`src/train.py`) from session 2. What changes is the machinery around it.

**This README follows the order you'd actually build the stack:**

1. [Orchestration with Airflow](#1-orchestration-with-airflow) — schedule and retry the retraining pipeline
2. [Deployment strategies](#2-deployment-strategies) — [batch scoring](#21-batch-scoring) vs. [streaming inference](#22-streaming-inference)
3. [Load testing with Locust](#3-load-testing-with-locust) — find the breaking point before users do
4. [BentoML serving](#4-bentoml-serving) — model → production API → Docker image

## Project structure

```
session_3/
├── docker-compose.yaml          # Airflow 3.3 local stack (LocalExecutor, Postgres)
├── Dockerfile                   # extends apache/airflow:3.3.0 with the DAG runtime deps
├── requirements.txt             # deps baked into the Airflow image
├── .env                         # Airflow local-dev env (UID, Fernet key, admin login)
├── pyproject.toml               # project deps + optional extras (tracking/gcp/bentoml/load)
├── Dags/
│   └── example_1.py             # retraining DAG: extract → train → evaluate
├── src/
│   ├── config.py                # YAML config loader
│   ├── prepare.py               # seeds data/processed/train.parquet (synthetic)
│   ├── train.py                 # fits the RandomForest → models/rf_model.pkl
│   └── batch_scoring.py         # ── batch strategy: GCS parquet in → predictions out
├── main_streaming_inference.py  # ── streaming strategy: Pub/Sub-triggered Cloud Function
├── bentoml_example.py           # BentoML service (save → serve → build → containerize)
├── locustfile.py                # load-test scenario (90% /predict, 10% /health)
├── config/
│   └── config.yaml              # data + model hyperparameters
└── data/, logs/, plugins/       # mounted into the Airflow containers
```

## Setup

```bash
# Base training/scoring stack
pip install -e .

# Add only the extras you need for the section you're on:
pip install -e ".[tracking]"   # MLflow registry (batch, streaming, BentoML all load from it)
pip install -e ".[gcp]"        # google-cloud-storage + firestore
pip install -e ".[bentoml]"    # BentoML serving
pip install -e ".[load]"       # Locust
pip install -e ".[airflow]"    # only to parse/lint DAGs locally; they RUN in the image
```

Python **3.10–3.12** (Airflow 3.3 has no 3.13 wheels yet).

---

## 1. Orchestration with Airflow

### What it is

A model isn't a one-off script — it's a *recurring pipeline*: pull fresh data →
train → evaluate → promote if better. **Airflow** turns that chain into a DAG
(Directed Acyclic Graph) of tasks with a schedule, and takes over the boring,
critical parts: running it on time, retrying the flaky step, showing you which
task failed and why, and never re-running a task whose upstream dependency
didn't succeed.

[`Dags/example_1.py`](Dags/example_1.py) is the retraining DAG: `extract` pulls
new rides from GCS, `train` runs `src/train.py` inside an MLflow run and pushes
the run ID to **XCom**, `evaluate` pulls that run ID back and compares MAE
against the current Production model.

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
- **Retries and alerting are free.** `retries: 2, retry_delay: 5min` handles the transient GCS timeout that would otherwise page you.
- **Dependencies are explicit.** `train` cannot run if `extract` failed. A cron chain of scripts gives you no such guarantee.
- **Observability.** The UI shows every run, every task, every log line, and lets you re-run a single failed task instead of the whole pipeline.
- **Backfills.** Flip `catchup=True` and Airflow replays history for you.
- **XCom passes small values between tasks** (here: the MLflow run ID) without you inventing a temp-file protocol.

### The local stack

[`docker-compose.yaml`](docker-compose.yaml) runs Airflow 3.3 with the
**LocalExecutor** — tasks are subprocesses of the scheduler, so there's no
Redis/Celery/Flower to babysit. Services: `postgres` (metadata DB),
`airflow-apiserver` (UI + REST API — Airflow 3 renamed the old *webserver*),
`airflow-scheduler`, `airflow-dag-processor` (a standalone component in Airflow 3),
and `airflow-triggerer`.

The DAG's runtime deps (`google-cloud-storage`, `mlflow`, scikit-learn, …) are
baked into the image by [`Dockerfile`](Dockerfile) + [`requirements.txt`](requirements.txt),
rather than reinstalled on every container start via `_PIP_ADDITIONAL_REQUIREMENTS`.

### Most important commands

```bash
# ── Start the stack ───────────────────────────────────
docker compose build                     # build the extended Airflow image
docker compose up -d                     # postgres + apiserver + scheduler + dag-processor + triggerer
open http://localhost:8080               # login: airflow / airflow

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

Before writing any serving code, answer one question: **how fresh does a
prediction need to be?** The honest answer decides the architecture, and it is
usually *far less fresh* than people assume.

| | Batch scoring | Streaming inference | Online (REST) |
|---|---|---|---|
| **Latency** | hours / days | seconds | milliseconds |
| **Trigger** | schedule (Airflow) | an event arrives | a user request |
| **Cost** | lowest — one job, no idle server | low — pay per invocation | highest — always-on |
| **Use when** | predictions are consumed from a table | reacting to events as they happen | a user is waiting for the answer |

Section 4 covers the *online* column. This section covers the other two.

### 2.1 Batch scoring

#### What it is

Score a whole file of rows on a schedule and write the predictions somewhere
your consumers already read from (a bucket, a warehouse table). Nobody calls an
API; they `SELECT` yesterday's predictions.

[`src/batch_scoring.py`](src/batch_scoring.py) reads a parquet from GCS, loads
the **Production** model from the MLflow registry, scores every row, stamps a
`run_date`, and writes the result back to GCS.

```python
def load_production_model():
    """Always load from MLflow Production stage — no hardcoded paths."""
    return mlflow.sklearn.load_model("models:/RideDurationModel/Production")

df["pred"]     = model.predict(features)
df["run_date"] = date.today().isoformat()
bucket.blob(output_blob).upload_from_string(out_buffer.getvalue())
```

#### Why use it

- **Cheapest thing that works.** No server sitting idle at 3 a.m. waiting for traffic.
- **Throughput over latency.** Vectorised `model.predict()` over 10M rows beats 10M HTTP round-trips by orders of magnitude.
- **Trivially retryable.** The job is idempotent — same input file, same output file. Re-run it and nothing breaks.
- **Failure is not an outage.** A late batch is an inconvenience; a down API is an incident.
- **The registry decouples deploys.** Promoting a new model to `Production` in MLflow changes tomorrow's predictions with *zero* code changes — note the deliberate absence of a hardcoded model path.

#### Most important commands

```bash
# Run the scoring job locally (needs the [gcp,tracking] extras + object-store creds)
export MLFLOW_TRACKING_URI=http://localhost:5000
python src/batch_scoring.py
```

In production you don't run this by hand — you make it the last task of the
Airflow DAG from section 1:

```python
PythonOperator(task_id="batch_score", python_callable=batch_score, dag=dag)
```

### 2.2 Streaming inference

#### What it is

An event lands on a message topic; a function wakes up, scores it, and writes the
prediction to a store. No polling, no server you manage, no fixed schedule —
the *event* is the trigger.

[`main_streaming_inference.py`](main_streaming_inference.py) is a **GCP Cloud
Function** triggered by every message published to the `ride-events` Pub/Sub
topic. It decodes the base64 payload, scores it, and appends the result to a
Firestore collection.

```python
MODEL = None   # module-level: loaded ONCE per container, not per message

def get_model():
    global MODEL
    if MODEL is None:                                  # cold start only
        MODEL = mlflow.sklearn.load_model(os.environ["MODEL_URI"])
    return MODEL

def predict_on_pubsub(event, context):
    payload = json.loads(base64.b64decode(event["data"]).decode("utf-8"))
    ...
    db.collection("ride-predictions").add(result)
```

That `global MODEL` is the single most important line in the file. Loading the
model inside the handler would re-download it from the registry **on every
message**. Loading it at module level means the warm container reuses it, and
only a cold start pays the cost.

#### Why use it

- **Reacts in seconds, not hours** — the gap batch scoring can't close.
- **Scales to zero.** No messages, no containers, no bill.
- **Auto-scales up.** Pub/Sub backpressure spawns more function instances; you wrote no scaling logic.
- **Producer and consumer are decoupled.** The app publishes a ride event and moves on. If the model is down, messages queue in Pub/Sub instead of erroring out.
- **Pub/Sub retries on failure** — a raised exception means the message is redelivered, not lost.

#### Testing the handler locally

The handler is a plain function — you don't need a deployed topic to exercise it,
just a base64-encoded payload shaped like a Pub/Sub event:

```bash
export MODEL_URI=models:/RideDurationModel/Production
python -c '
import base64, json, main_streaming_inference as m
msg = {"ride_id":"r-123","distance_km":8.4,"passengers":2,"hour_of_day":17}
event = {"data": base64.b64encode(json.dumps(msg).encode())}
m.predict_on_pubsub(event, None)
'
```

---

## 3. Load testing with Locust

### What it is

Your API answers one request in 40 ms. What does it do with **500 concurrent
users**? *"It'll probably be fine"* is a guess, and it's the guess that turns
into an incident. **Locust** answers it empirically: you describe one simulated
user in Python, it spawns thousands of them and reports latency percentiles and
failure rates.

[`locustfile.py`](locustfile.py) simulates a client that hits `/predict` ten
times as often as `/health`, with realistic think-time between calls, and
validates the *content* of each response — not just its status code.

```python
class MLAPIUser(HttpUser):
    wait_time = between(0.5, 2.0)      # think-time, so the load is realistic

    @task(weight=10)                    # 10x more common than /health
    def predict(self):
        payload = {"distance_km": ..., "passengers": ..., "hour_of_day": ...}
        with self.client.post("/predict", json=payload, catch_response=True) as resp:
            if resp.status_code != 201:
                resp.failure(f"Expected 201, got {resp.status_code}")
            elif resp.json().get("duration_min", -1) < 0:
                resp.failure("Negative duration in response")   # a 200 that's still wrong
```

### Why use it

- **Scenarios are plain Python.** Random payloads, weighted task mixes, auth flows, stateful sessions — anything you can write, you can simulate. No XML, no clicking through a GUI recorder.
- **`catch_response` catches *semantic* failures.** A `200 OK` carrying a negative duration is a bug; Locust marks it as a failure. HTTP-only tools would report 100% success.
- **Percentiles, not averages.** p95/p99 is what your users feel. The mean hides the tail.
- **Finds the knee in the curve.** Ramp users up until p95 latency spikes — that's your real capacity, and now you can size the box you deploy on with a number instead of a vibe.
- **`--headless --csv` makes it a CI gate.** Fail the build if p95 regresses.

> **Note:** the scenario asserts `201`. FastAPI/BentoML return `200` by default,
> so point this at an endpoint that really returns `201` — or change the check to
> `!= 200`.

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
right CUDA base image. **BentoML** generates all of it from a model reference
and a Python function.

[`bentoml_example.py`](bentoml_example.py) walks the full path: pull the
production model out of the MLflow registry, save it into BentoML's model store,
wrap it in a Service, then package it as a Docker image. Each of the five things
a serving framework buys you is marked `[1]`–`[5]` in the file.

```python
@bentoml.service(
    workers=1,                                    # [5] concurrency: process-level
    traffic={"concurrency": 64, "timeout": 30},   # [5] admission + [2] queue bound
)
class RideDuration:
    bento_model = bentoml.models.BentoModel(MODEL_TAG)   # [4] versioning

    def __init__(self) -> None:                          # runs once per worker
        self.model = joblib.load(self.bento_model.path_of("model.pkl"))

    @bentoml.api(batchable=True, batch_dim=0,            # [1] dynamic batching
                 max_batch_size=64, max_latency_ms=2_000)
    def predict(self, inputs: list[PredictRequest]) -> list[PredictResponse]:
        features = np.array([[r.distance_km, r.passengers] for r in inputs])  # [3] pre
        preds = self.model.predict(features)             # ONE call for all N rows
        return [PredictResponse(duration_min=round(float(p), 2), ...)         # [3] post
                for p in preds]
```

### Why use it

- **[1] Dynamic batching for free.** `batchable=True` makes BentoML merge concurrent requests into one `predict()` call — invisible to callers, who each send and receive a single row. This is the single biggest throughput win available to an ML API. Re-run section 3's Locust test with it on and off; the difference is not subtle.
- **[2] Request queueing.** Traffic past `concurrency` waits instead of being refused, and `timeout` bounds that wait so an overloaded service sheds load rather than melting down.
- **[3] Pydantic in, Pydantic out.** Malformed requests are rejected with a clean 4xx before they ever touch the model — and post-processing (rounding, derived fields) happens once, server-side, instead of in every client.
- **[4] The model store is versioned.** Every save mints an immutable tag (`ride_duration:rlkriaedugl2xuiy`); rollback is redeploying a previous tag. Pin it in production — `:latest` means a colleague's save silently changes what you serve.
- **[5] Concurrency control.** `workers` sets process-level parallelism, `traffic.concurrency` sets admission. Note the constraint: `concurrency` must be ≥ `max_batch_size`, or the dispatcher can never hold enough requests to fill a batch.
- **`bentoml containerize` writes the Dockerfile.** Correct Python version, correct deps, correct model — no hand-maintained image.
- **Batteries included:** OpenAPI docs at `/docs`, Prometheus metrics at `/metrics`, health checks — all out of the box.

> **Version note:** this targets the `@bentoml.service` class API (BentoML 1.2+,
> tested on 1.4.39). The legacy `bentoml.io` + `bentoml.Service(runners=[...])`
> API was **removed** in 1.2 — `to_runner()` and `bentoml.io` no longer exist.
> Likewise, the model is loaded by MLflow *alias* (`models:/Name@production`),
> since MLflow 3 removed the `Production`/`Staging` stages.

### Most important commands

```bash
# ── Save the model into the BentoML store ─────────────
export MLFLOW_TRACKING_URI=http://localhost:5000   # else falls back to models/rf_model.pkl
python bentoml_example.py
bentoml models list                      # see the tags
bentoml models get ride_duration:latest

# ── Dev server (hot reload) ───────────────────────────
bentoml serve bentoml_example:RideDuration --port 8005 --reload

# NOTE the wire format: a batchable API takes {"<param>": [ <one item> ]}
curl -X POST http://localhost:8005/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"distance_km": 8.4, "passengers": 2}]}'
# → [{"duration_min": 17.68, "eta_band": "10-20 min", "model_version": "...", "batch_size": 1}]

curl -X POST http://localhost:8005/model_info -d '{}'   # [4] what's actually live?
open http://localhost:8005/docs

# ── Package & containerize ────────────────────────────
bentoml build                                    # → a versioned Bento
bentoml list                                     # list built Bentos
bentoml containerize ride_duration_api:latest    # → a Docker image

docker run -p 3000:3000 ride_duration_api:latest # the same image you'd ship anywhere
```

---

## Where this leaves you

You now have all three deployment shapes for the same model, and the tools to
choose between them: **Airflow** schedules the retraining, **batch scoring**
serves the cheap/high-throughput case, **streaming inference** serves the
event-driven case, **BentoML** serves the online case, and **Locust** tells you
which of them will hold up.

The lesson worth keeping: the model was the easy part.
