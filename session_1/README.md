# Ride Duration API

A minimal [FastAPI](https://fastapi.tiangolo.com/) service that predicts ride
duration (in minutes) from a trip's distance and passenger count. Built as an
MLOps course example.

## Project structure

```
session_1/
├── fastapi_example.py    # FastAPI app + entry point
├── litestar_example.py   # Litestar app (same model, DI-based)
├── pytorch_to_onnx.py    # Export a PyTorch model to ONNX + validate
├── pyproject.toml        # Project metadata + Python dependencies
├── src/
│   ├── __init__.py
│   └── model.py          # RideDurationModel (placeholder heuristic)
└── README.md
```

> **Note:** `src/model.py` currently holds a simple placeholder model
> (distance ÷ average speed + per-passenger overhead). Swap in a real trained
> model when available — keep the `predict(features)` signature.

## Requirements

- Python 3.10+

## Setup

Create a virtual environment and install the project (dependencies are
declared in `pyproject.toml`):

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .                   # installs deps from pyproject.toml (editable)
```

The `-e` flag installs the project in *editable* mode, so changes to the source
are picked up without reinstalling. Drop `-e` for a plain install (`pip install .`).

## Running the server

This repo ships two equivalent implementations of the same API — one with
FastAPI, one with [Litestar](https://litestar.dev/). Pick either.

### FastAPI (port 8000)

**As a Python script:**

```bash
python fastapi_example.py
```

**With uvicorn (adds auto-reload for development):**

```bash
uvicorn fastapi_example:app --host 127.0.0.1 --port 8000 --reload
```

The server starts at **http://127.0.0.1:8000**. Stop it with `Ctrl+C`.

### Litestar (port 8001)

**As a Python script:**

```bash
python litestar_example.py
```

**With the Litestar CLI (adds auto-reload for development):**

```bash
litestar --app litestar_example:app run --host 127.0.0.1 --port 8001 --reload
```

The server starts at **http://127.0.0.1:8001**. Stop it with `Ctrl+C`.

> Both apps expose the same `/health` and `/predict` endpoints — just swap the
> port (`8000` → `8001`) in the examples below.

## API endpoints

| Method | Path        | Description                          |
|--------|-------------|--------------------------------------|
| GET    | `/health`   | Health check                         |
| POST   | `/predict`  | Predict ride duration                |

### `POST /predict`

**Request body:**

| Field        | Type  | Required | Default | Description         |
|--------------|-------|----------|---------|---------------------|
| `distance`   | float | yes      | —       | Trip distance (km)  |
| `passengers` | int   | no       | `1`     | Number of passengers|

**Example:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"distance": 10, "passengers": 2}'
```

**Response:**

```json
{"duration_min": 21.0, "status": "ok"}
```

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status": "healthy"}
```

## Interactive documentation

Both frameworks auto-generate OpenAPI docs once the server is running.

**FastAPI** (port 8000):

- **Swagger UI:** http://127.0.0.1:8000/docs — try requests in the browser
- **ReDoc:** http://127.0.0.1:8000/redoc — clean reference view
- **OpenAPI schema (JSON):** http://127.0.0.1:8000/openapi.json

**Litestar** (port 8001):

- **Swagger UI:** http://127.0.0.1:8001/schema/swagger — try requests in the browser
- **ReDoc:** http://127.0.0.1:8001/schema/redoc — clean reference view
- **OpenAPI schema (JSON):** http://127.0.0.1:8001/schema/openapi.json

## Docker

The [`Dockerfile`](Dockerfile) builds the **FastAPI** app as a slim,
production-style image using a **multi-stage build**: a `builder` stage installs
the dependencies into an isolated virtualenv, and a minimal `runtime` stage
copies only that venv plus the app code onto `python:3.12-slim`. The container
runs as a non-root user and serves on port **8000** (bound to `0.0.0.0`).

**Build the image:**

```bash
docker build -t ride-duration-api:session1 .
```

**Run the container:**

```bash
docker run --rm -p 8000:8000 ride-duration-api:session1
```

The API is then available at **http://127.0.0.1:8000** (same endpoints and docs
as above). Stop it with `Ctrl+C`.

## Docker Compose

[`docker-compose.yml`](docker-compose.yml) runs the API alongside an
[MLflow](https://mlflow.org/) tracking server, wiring together a small local
MLOps stack:

| Service  | Image / build      | Port   | Purpose                              |
|----------|--------------------|--------|--------------------------------------|
| `api`    | built from `Dockerfile` | `8000` | Ride Duration API                    |
| `mlflow` | `ghcr.io/mlflow/mlflow` | `5000` | Experiment tracking + artifact store |

The `api` service mounts a local [`models/`](models/) directory read-only at
`/models` (via the `MODEL_PATH` env var) and waits for `mlflow` to start.
MLflow persists its artifacts in the named `mlflow-data` volume.

**Start the stack:**

```bash
docker compose up --build
```

- API: **http://127.0.0.1:8000**
- MLflow UI: **http://127.0.0.1:5000**

**Run in the background / tear down:**

```bash
docker compose up -d --build   # detached
docker compose down            # stop and remove containers
docker compose down -v         # also remove the mlflow-data volume
```

> Drop trained model files into `models/v1/` (e.g. `model.pkl`) to make them
> available inside the container at `/models/v1/`.

## Running tests

Unit tests for the model live in [`tests/`](tests/) and use `pytest`. Install
the `dev` extra, then run them:

```bash
pip install -e ".[dev]"
pytest
```

Expected output: `5 passed`. The tests mock the model's internal estimator
(`RideDurationModel._model`) and exercise the prediction path plus threshold
clipping.

## Exporting to ONNX

[`pytorch_to_onnx.py`](pytorch_to_onnx.py) exports a small PyTorch model
(`RideDurationTorchModel` — a single linear layer initialized to match the
`RideDurationModel` heuristic) to the portable [ONNX](https://onnx.ai/) format
and validates the result.

These deps are heavy and unrelated to serving the API, so they live in an
optional `onnx` extra rather than the core dependencies. Install them with:

```bash
pip install -e ".[onnx]"
```

Run the export:

```bash
python pytorch_to_onnx.py
```

This writes **`model.onnx`** to the project root, runs `onnx.checker` to verify
the graph is valid, and prints the model's inputs/outputs:

```
Inputs:  ['features']
Outputs: ['duration']
```

> A couple of non-fatal warnings are expected (opset auto-bumped 17→18,
> `torchvision not installed`) — they don't affect the exported model.

## How `pyproject.toml` grew, stage by stage

This project started with a flat `requirements.txt` and migrated to
`pyproject.toml`, then grew as each new capability was added. Below is the exact
edit made to the file at every stage — a small tour of how to shape a
`pyproject.toml` around a project's needs.

### Stage 0 — the starting point: `requirements.txt`

Originally dependencies lived in a plain text file:

```text
fastapi
uvicorn
pydantic
```

No versions, no metadata, no way to declare optional/dev dependencies or build
the project as an installable package.

### Stage 1 — migrate to `pyproject.toml` (core dependencies)

We replaced `requirements.txt` with a `pyproject.toml` that declares project
metadata, a Python floor, pinned lower bounds, and a build backend. `litestar`
was added here because [`litestar_example.py`](litestar_example.py) needs it:

```toml
[project]
name = "ride-duration-api"
version = "0.1.0"
description = "A minimal FastAPI/Litestar service that predicts ride duration from trip distance and passenger count."
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "fastapi>=0.138",
    "uvicorn>=0.49",
    "pydantic>=2.13",
    "litestar>=2.24",
]

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["src"]      # makes `src` importable after `pip install`
```

Install everything (editable) with a single command:

```bash
pip install -e .
```

### Stage 2 — add an optional `onnx` extra (PyTorch → ONNX export)

[`pytorch_to_onnx.py`](pytorch_to_onnx.py) needs `torch` and `onnx`, which are
large and irrelevant to serving the API. Instead of bloating the core
dependencies, we added an **optional-dependency group** so they install only on
demand. (`onnxscript` was added after we found torch's exporter requires it.)

```toml
[project.optional-dependencies]
# Heavy deps used only by pytorch_to_onnx.py (ONNX export). Install with:
#   pip install -e ".[onnx]"
onnx = [
    "torch>=2.2",
    "onnx>=1.16",
    "onnxscript>=0.2",
]
```

Install the core project **plus** the ONNX tooling:

```bash
pip install -e ".[onnx]"
```

### Stage 3 — add a `dev` extra + pytest config (tests)

To run [`tests/`](tests/), we added a second optional group for test tooling and
a `[tool.pytest.ini_options]` block so `pytest` knows where the tests live:

```toml
[project.optional-dependencies]
# ... onnx group from Stage 2 ...
# Test/dev tooling. Install with:
#   pip install -e ".[dev]"
dev = [
    "pytest>=8.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Install the core project **plus** the dev tooling, then run the suite:

```bash
pip install -e ".[dev]"
pytest
```

### The result

The final [`pyproject.toml`](pyproject.toml) cleanly separates concerns: a small
core install for running the API, and opt-in extras for the heavy ONNX tooling
and the test suite. Combine extras when you need several at once:

```bash
pip install -e ".[onnx,dev]"
```

## Troubleshooting

- **IDE warns "Import could not be resolved":** your editor is using the system
  Python instead of the venv. In VS Code: `Cmd/Ctrl+Shift+P` →
  *Python: Select Interpreter* → choose `.venv/bin/python`.
- **`ModuleNotFoundError: No module named 'src'`:** run the server from the
  `session_1/` directory so `src` is importable.
