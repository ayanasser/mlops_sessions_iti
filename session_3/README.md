# Ride Duration — Orchestration & Deployment (Session 3)

Session 2 stopped at a *trained, registered* model. Session 3 is about everything
that happens **after** the model exists: scheduling the pipeline that keeps it
fresh, choosing how predictions reach consumers, proving the service survives
real traffic, and putting it on a cloud VM.

The model itself is unchanged — the same synthetic **ride-duration** regressor
(`src/train.py`) from session 2. What changes is the machinery around it.

**This README follows the order you'd actually build the stack:**

1. [Orchestration with Airflow](#1-orchestration-with-airflow) — schedule and retry the retraining pipeline
2. [Deployment strategies](#2-deployment-strategies) — [batch scoring](#21-batch-scoring) vs. [streaming inference](#22-streaming-inference)
3. [Load testing with Locust](#3-load-testing-with-locust) — find the breaking point before users do
4. [BentoML serving](#4-bentoml-serving) — model → production API → Docker image
5. [TorchServe](#5-torchserve) — the PyTorch-native serving path
6. [GCP & cloud basics](#6-gcp--cloud-basics) — ship the container to a real VM, [by hand](#61-the-manual-flow-gcloud-cli) then [with Terraform](#62-the-same-thing-with-terraform)

## Project structure

```
session_3/
├── docker-compose.yaml          # Airflow 3.3 local stack (LocalExecutor, Postgres)
├── Dockerfile                   # extends apache/airflow:3.3.0 with the DAG runtime deps
├── requirements.txt             # deps baked into the Airflow image
├── .env                         # Airflow local-dev env (UID, Fernet key, admin login)
├── pyproject.toml               # project deps + optional extras (tracking/gcp/bentoml/torch/load)
├── Dags/
│   └── example_1.py             # retraining DAG: extract → train → evaluate
├── src/
│   ├── config.py                # YAML config loader
│   ├── prepare.py               # seeds data/processed/train.parquet (synthetic)
│   ├── train.py                 # fits the RandomForest → models/rf_model.pkl
│   └── batch_scoring.py         # ── batch strategy: GCS parquet in → predictions out
├── main_streaming_inference.py  # ── streaming strategy: Pub/Sub-triggered Cloud Function
├── bentoml_example.py           # BentoML service (save → serve → build → containerize)
├── torch_serve.py               # TorchScript export + custom handler + .mar packaging
├── locustfile.py                # load-test scenario (90% /predict, 10% /health)
├── infra/                       # ── Terraform: the gcloud commands, as code
│   ├── versions.tf              #    provider + (optional) GCS remote state
│   ├── variables.tf             #    every knob: project, image, port, CIDRs
│   ├── main.tf                  #    static IP, service account, firewall x2, VM
│   ├── startup.sh               #    boot script: install Docker, pull, run
│   ├── outputs.tf               #    external IP, api_url, ready-to-paste curl
│   └── terraform.tfvars.example #    copy → terraform.tfvars, fill in 2 values
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
pip install -e ".[torch]"      # TorchServe + torch-model-archiver
pip install -e ".[load]"       # Locust
pip install -e ".[airflow]"    # only to parse/lint DAGs locally; they RUN in the image
```

Python **3.10–3.12** (Airflow 3.3 and torch have no 3.13 wheels yet).

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

Sections 4–5 cover the *online* column. This section covers the other two.

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
# Run the scoring job locally (needs GCP creds + the [gcp,tracking] extras)
export GOOGLE_APPLICATION_CREDENTIALS=~/keys/sa.json
export MLFLOW_TRACKING_URI=http://localhost:5000
python src/batch_scoring.py

# Inspect input/output in the bucket
gsutil ls gs://mlops-gcs-ride-duration/scoring/input/
gsutil ls gs://mlops-gcs-ride-duration/scoring/output/
gsutil cp gs://mlops-gcs-ride-duration/scoring/output/$(date +%F).parquet .
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

#### Most important commands

```bash
# ── Create the topic (one time) ───────────────────────
gcloud pubsub topics create ride-events

# ── Deploy the function ───────────────────────────────
gcloud functions deploy predict_on_pubsub \
  --runtime python311 \
  --trigger-topic ride-events \
  --set-env-vars MODEL_URI=models:/RideDurationModel/Production

# ── Publish a test event ──────────────────────────────
gcloud pubsub topics publish ride-events \
  --message='{"ride_id":"r-123","distance_km":8.4,"passengers":2,"hour_of_day":17}'

# ── Watch the logs / read the predictions ─────────────
gcloud functions logs read predict_on_pubsub --limit 20
gcloud firestore documents list ride-predictions   # or read from the Console
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
- **Finds the knee in the curve.** Ramp users up until p95 latency spikes — that's your real capacity, and now you can size the VM in section 6 with a number instead of a vibe.
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
Production model out of the MLflow registry, save it into BentoML's model store,
wrap it in a Service, then package it as a Docker image.

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

- **Adaptive batching for free.** `"batchable": True` makes BentoML collect concurrent requests into one `predict()` call. This is the single biggest throughput win available to an ML API, and it's one dict key. Re-run section 3's Locust test with it on and off — the difference is not subtle.
- **Pydantic in, Pydantic out.** Malformed requests are rejected with a clean 422 before they ever touch the model.
- **Runners scale independently of the API.** Model inference runs in its own worker process, so a slow prediction doesn't block the event loop serving `/health`.
- **`bentoml containerize` writes the Dockerfile.** Correct Python version, correct deps, correct model — no hand-maintained image.
- **The model store is versioned.** Every `save_model` gets a tag (`ride_duration:abc123`); rollback is redeploying a previous tag.
- **Batteries included:** OpenAPI docs at `/docs`, Prometheus metrics at `/metrics`, health checks — all out of the box.

> **Version note:** [`pyproject.toml`](pyproject.toml) pins `bentoml>=1.1,<1.2`.
> The script uses the legacy `bentoml.io` + `bentoml.Service(runners=[...])` API,
> which BentoML 1.2 replaced with the `@bentoml.service` class-based API.

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
  -d '{"distance_km": 8.4, "passengers": 2, "hour_of_day": 17}'

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
        feats = [[r["distance_km"], r["passengers"], r["hour_of_day"]] for r in rows]
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
  -d '{"distance_km": 8.4, "passengers": 2, "hour_of_day": 17}'

# ── Management API (:8081) — no restart needed ────────
curl http://localhost:8081/models                                   # list
curl -X POST "http://localhost:8081/models?url=ride_duration.mar"   # register
curl -X PUT  "http://localhost:8081/models/ride_duration?min_worker=4"  # scale workers
curl -X DELETE http://localhost:8081/models/ride_duration/1.0       # unregister

# ── Metrics (:8082) ───────────────────────────────────
curl http://localhost:8082/metrics
```

---

## 6. GCP & cloud basics

### What it is

Everything above ran on your laptop. This section is the last mile: **build the
image, push it to a registry, and run it on a VM with a public IP** — the
simplest deployment that a real user can actually reach.

The flow is deliberately boring, and that's the point:

```
docker build → docker push → create VM → open firewall → SSH → docker pull → docker run
```

We do it **twice**: first by hand with `gcloud`, so you see every moving part;
then with **Terraform** ([`infra/`](infra/)), which is what you'd actually keep.
Do the manual pass first — Terraform is much easier to read once you know which
command each resource is standing in for.

### Why use it

- **The container is the contract.** The image that passed Locust on your laptop is bit-for-bit the image running on the VM. "Works on my machine" stops being a sentence anyone says.
- **A single VM is the right first step.** Understand a `docker run` on one box before you reach for Kubernetes, Cloud Run, or Vertex AI — those solve scaling problems you don't have yet.
- **Redeploying is `pull` + `run`.** No build tooling on the server, no source code on the server.
- **`--restart unless-stopped` gives you free resilience.** The container comes back after a crash *and* after a VM reboot.
- **The firewall is deny-by-default.** Nothing is reachable until you open a port and tag the VM — a good default worth internalising early.
- **Everything is a CLI call, so everything is scriptable** — which is exactly the observation Terraform is built on.

### 6.1 The manual flow (`gcloud` CLI)

```bash
# ── On your local machine: push image to Docker Hub ──
docker build -t yourusername/ride-api:latest .
docker push yourusername/ride-api:latest

# ── Create a GCP VM (one time, via gcloud CLI) ────────
gcloud compute instances create ride-api-vm \
  --machine-type=e2-small \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --tags=http-server \
  --zone=us-central1-a

# ── Open firewall port 8000 (one time) ────────────────
gcloud compute firewall-rules create allow-8000 \
  --allow tcp:8000 --target-tags http-server

# ── SSH into the VM ───────────────────────────────────
gcloud compute ssh ride-api-vm --zone=us-central1-a

# ── On the VM: install Docker (one time) ──────────────
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER && newgrp docker

# ── On the VM: deploy / redeploy ──────────────────────
docker pull yourusername/ride-api:latest
docker stop ride-api 2>/dev/null || true
docker rm   ride-api 2>/dev/null || true
docker run -d --name ride-api --restart unless-stopped \
  -p 8000:8000 -e MODEL_PATH=/models/v1/model.pkl \
  yourusername/ride-api:latest

# ── Get public IP and verify ──────────────────────────
gcloud compute instances describe ride-api-vm \
  --zone=us-central1-a --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
curl http://<EXTERNAL_IP>:8000/health
```

#### Reading the important flags

| Flag | Why it's there |
|---|---|
| `--tags=http-server` | The label the firewall rule targets. No tag → no traffic, even with the rule created. |
| `--target-tags http-server` | Binds the rule to *only* the VMs carrying that tag, not the whole network. |
| `-d` | Detached — the container survives your SSH session ending. |
| `--restart unless-stopped` | Restarts on crash and on VM reboot; still respects a deliberate `docker stop`. |
| `-p 8000:8000` | Publishes the container port on the host. Without it the firewall rule opens a port nothing is listening on. |
| `-e MODEL_PATH=...` | Config via environment — the same image serves any model version. |
| `docker stop \|\| true` | First deploy has no container to stop; `\|\| true` keeps the script from failing on it. |

#### Everyday `gcloud` commands

```bash
# ── Auth & project setup (one time) ───────────────────
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud config set compute/zone us-central1-a

# ── Inspect what's running ────────────────────────────
gcloud compute instances list
gcloud compute firewall-rules list
gcloud compute ssh ride-api-vm --command='docker ps'
gcloud compute ssh ride-api-vm --command='docker logs --tail 50 ride-api'

# ── Stop paying for it ────────────────────────────────
gcloud compute instances stop ride-api-vm      # keeps the disk, stops the bill for CPU
gcloud compute instances delete ride-api-vm    # gone for good
```

> **Cost:** an `e2-small` running 24/7 is a few dollars a month — small, not free.
> `gcloud compute instances stop` when you're done with the demo.

#### Where the manual flow breaks down

It works, and that's the trap. The problems only show up later:

- **It exists nowhere but in your shell history.** Six months on, nobody knows whether that firewall rule was `8000` or `8080`, or why the VM is an `e2-small`.
- **It isn't repeatable.** Building the staging environment means re-typing all of it and hoping you don't fluff a flag. The drift between environments is where the 2 a.m. bugs live.
- **It has no idea what it already did.** Re-run `instances create` and it errors; you have to remember what exists.
- **Tearing down is manual too** — and the resource you forget is the one that bills you.
- **There's no review.** Nobody diffs a shell command before it opens a port to `0.0.0.0/0`.

### 6.2 The same thing, with Terraform

#### What it is

**Terraform** is the same deployment written as *declared state* instead of
typed commands. You describe what should exist; Terraform diffs that against
what does exist and makes only the changes needed. [`infra/`](infra/) is a
faithful port of the block above — plus the three things the manual flow quietly
skipped.

| Manual command | Terraform resource in [`infra/main.tf`](infra/main.tf) |
|---|---|
| `gcloud compute instances create` | `google_compute_instance.ride_api` |
| `gcloud compute firewall-rules create allow-8000` | `google_compute_firewall.allow_app` |
| SSH in, `curl get.docker.com \| sh`, `docker pull`, `docker run` | `metadata.startup-script` → [`infra/startup.sh`](infra/startup.sh) |
| `gcloud compute instances describe ... --format='get(...natIP)'` | `output.external_ip` in [`infra/outputs.tf`](infra/outputs.tf) |
| *(nothing — the IP was ephemeral)* | `google_compute_address.ride_api` — a **static** IP |
| *(nothing — the VM used the default SA)* | `google_service_account.ride_api` — **least privilege** |
| *(nothing — port 22 was public)* | `google_compute_firewall.allow_ssh_iap` — **SSH via IAP only** |

Those last three rows are the real argument for IaC. They're all things you'd
*intend* to do manually and never get around to; in Terraform they're four lines
you write once.

The whole deploy — install Docker, pull, run — moves into the VM's
`startup-script` metadata, so it runs automatically on first boot. **You never SSH
in to deploy.** The `templatefile()` call injects your image and port into it:

```hcl
metadata = {
  startup-script = templatefile("${path.module}/startup.sh", {
    docker_image   = var.docker_image
    container_name = var.container_name
    app_port       = var.app_port
    model_path     = var.model_path
  })
}
```

#### Why use it

- **The infrastructure is in code review.** `--allow tcp:8000 --source-ranges 0.0.0.0/0` slips through a shell. It does not slip through a pull request.
- **`terraform plan` is a dry run.** You see *exactly* what will change — created, updated, or **destroyed** — before anything happens. There is no `gcloud` equivalent.
- **Idempotent.** Run `apply` ten times; if nothing changed, nothing happens. Re-running the manual flow throws "already exists" errors.
- **`terraform destroy` is complete.** It knows every resource it made, so the demo VM, its IP, its firewall rules and its service account all go together. Nothing lingers on the bill.
- **A new environment is a new `.tfvars`.** Staging = the same code, a different variable file. That's how you kill environment drift.
- **It's the documentation, and it can't go stale** — because it's also the thing that runs.
- **State is shared.** Uncomment the GCS backend in [`versions.tf`](infra/versions.tf) and your teammate's `plan` sees what you deployed.

#### Most important commands

```bash
cd infra
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars → set gcp_project and docker_image

gcloud auth application-default login   # Terraform's credentials

# ── The core loop ─────────────────────────────────────
terraform init         # download the google provider (once, and after backend changes)
terraform fmt          # canonical formatting
terraform validate     # syntax + type check — no cloud calls, no credentials needed
terraform plan         # DRY RUN: what would change? read this before every apply
terraform apply        # make it so (prompts for confirmation)

# ── Read the outputs ──────────────────────────────────
terraform output                        # all of them
terraform output -raw external_ip       # just the IP, for scripting
curl "$(terraform output -raw api_url)/health"

# ── Redeploy a new image version ──────────────────────
terraform apply -var="docker_image=yourusername/ride-api:v2"

# ── Inspect state ─────────────────────────────────────
terraform state list                    # every resource Terraform manages
terraform show                          # full current state

# ── Tear it ALL down (VM + IP + firewall + SA) ───────
terraform destroy
```

> **Always read the `plan` output.** The line that matters is the summary:
> `Plan: 1 to add, 0 to change, 0 to destroy.` If **destroy** is non-zero and you
> didn't expect it, stop — Terraform is about to replace something. That
> confirmation prompt is the whole safety model.

#### Gotchas worth knowing

- **`terraform.tfvars` is gitignored** — it names your project and image. Commit [`terraform.tfvars.example`](infra/terraform.tfvars.example) instead.
- **`*.tfstate` is gitignored too, and must stay that way.** It holds the full resource graph including anything sensitive. Use the GCS backend for anything shared.
- **State is the source of truth.** If you `gcloud compute instances delete` a VM Terraform manages, Terraform still thinks it exists until the next `plan` reconciles it. Pick one tool per resource and stick with it.
- **`allowed_source_ranges` defaults to `0.0.0.0/0`** — the entire internet, matching the manual flow. Narrow it to your own IP (`curl -s ifconfig.me`) the moment this serves a real model.
- **`app_port` sets the firewall rule *and* the `docker run -p` in one place.** In the manual flow those were two commands that could silently disagree.

---

## Where this leaves you

You now have all four deployment shapes for the same model, and the tools to
choose between them: **Airflow** schedules the retraining, **batch scoring**
serves the cheap/high-throughput case, **streaming inference** serves the
event-driven case, **BentoML/TorchServe** serve the online case, **Locust** tells
you which of them will hold up, and **GCP + Terraform** put it in front of a real
user — reproducibly.

The lesson worth keeping: the model was the easy part.
