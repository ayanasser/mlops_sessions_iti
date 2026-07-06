# Ride Duration — Experiment Tracking (Session 2)

Session 2 of the MLOps course. It trains a small **ride-duration** regression
model on synthetic data and tracks the run — parameters, metrics, and the fitted
model — with two experiment trackers so you can compare them side by side:
[MLflow](https://mlflow.org/) and [Weights & Biases](https://wandb.ai/). The
model reuses the same distance/passenger relationship from session 1.

Around the training core it also wires up the surrounding MLOps toolchain: a DVC
pipeline, a FastAPI serving image, a GitHub Actions CI/CD workflow, and Terraform
for the AWS resources (S3 DVC remote + ECR).

## Project structure

```
session_2/
├── mlflow_example.py                # MLflow training + logging script (entry point)
├── mlflow_modelregiestry_example.py # load/promote a registered model via MlflowClient (aliases)
├── mlflow_s3_registry_example.py    # same, against an S3-backed tracking server
├── wandb_example.py                 # Weights & Biases counterpart to mlflow_example.py
├── serve.py               # FastAPI inference service (/health, /predict)
├── Dockerfile             # multi-stage image that serves the model
├── .dockerignore          # trims the Docker build context
├── pyproject.toml         # Project metadata + Python dependencies
├── docker-compose.yml     # MLflow tracking server, local volume artifacts (from session 1)
├── docker-compose.s3.yml  # MLflow tracking server, artifacts in an S3 bucket
├── .env.s3.example        # template env for docker-compose.s3.yml (copy to .env)
├── dvc.yaml               # DVC pipeline definition (prepare → train → evaluate)
├── dvc.lock               # DVC lock file (recorded stage input/output hashes)
├── .dvc/
│   └── config             # DVC remotes: local (default) + gcs + s3
├── config/
│   └── config.yaml        # pipeline parameters (data + model)
├── data/
│   └── raw/
│       ├── rides.csv      # raw synthetic dataset (pipeline input)
│       └── rides.csv.dvc  # DVC pointer for the raw dataset
├── src/
│   ├── __init__.py
│   ├── train.py           # data gen, train, evaluate + DVC `train` stage
│   ├── prepare.py         # DVC `prepare` stage (raw CSV → train/val parquet)
│   ├── evaluate.py        # DVC `evaluate` stage (model → metrics/scores.json)
│   └── config.py          # YAML config loader
├── tests/
│   ├── test_train.py      # pytest suite for src/train.py
│   └── test_integration.py# end-to-end training/eval integration test
├── infra/                 # Terraform IaC (AWS: S3 DVC remote + ECR repo)
│   ├── main.tf            # providers, S3 bucket, ECR repository, outputs
│   ├── variables.tf       # input variables
│   └── commands.md        # Terraform command cheatsheet
├── .pre-commit-config.yaml
└── README.md

# The CI/CD workflow lives at the REPO ROOT (one level up), not in session_2/:
mlops_sessions/
└── .github/workflows/
    └── ci_cd.yml          # lint + test, then build & push the Docker image
```

> `src/train.py` deliberately contains **no tracking calls** so it can be unit
> tested in isolation. It holds the shared model core (`generate_data`,
> `split_data`, `train_model`, `evaluate`) reused by every tracking script; all
> experiment tracking lives in `mlflow_example.py` / `wandb_example.py`.

## Requirements

- Python 3.10+
- Docker (for the MLflow tracking server)
- Terraform 1.x + AWS CLI (for the `infra/` provisioning)
- GitHub CLI (`gh`) — optional, for CI/PR workflows

## Installing the toolchain

The commands below target **macOS (Homebrew)**, with notes for Linux/Windows.

### Homebrew (macOS package manager)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Terraform

```bash
# macOS
brew tap hashicorp/tap
brew install hashicorp/tap/terraform

# Linux (Debian/Ubuntu)
wget -O- https://apt.releases.hashicorp.com/gpg | \
  sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] \
  https://apt.releases.hashicorp.com $(lsb_release -cs) main" | \
  sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install terraform

# Windows
choco install terraform        # or: winget install HashiCorp.Terraform

terraform version              # verify
```

### AWS CLI

```bash
# macOS
brew install awscli

# Linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip awscliv2.zip && sudo ./aws/install

# Windows
winget install Amazon.AWSCLI

aws --version                 # verify
aws configure                 # set credentials + default region
```

### Docker

Install **Docker Desktop** (macOS/Windows) from
<https://www.docker.com/products/docker-desktop/>, or Docker Engine on Linux:

```bash
# macOS
brew install --cask docker

# Linux (Debian/Ubuntu)
curl -fsSL https://get.docker.com | sh

docker --version              # verify
```

### GitHub CLI — optional

```bash
brew install gh               # macOS
sudo apt install gh           # Linux (Debian/Ubuntu)
winget install GitHub.cli     # Windows

gh --version
gh auth login                 # authenticate
```

### `act` — run GitHub Actions locally (optional)

Runs workflows from `.github/workflows/` on your machine (requires Docker):

```bash
brew install act              # macOS
# Linux: curl -fsSL https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

act --version
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"            # core deps + pytest
```

## 1. Start the MLflow tracking server

The MLflow service is carried over from session 1's `docker-compose.yml`:

```bash
docker compose up -d mlflow
```

- MLflow UI: **http://localhost:5000**
- Backend store: SQLite (`mlflow.db`)
- Artifacts: served by the tracking server and persisted in the `mlflow-data`
  volume

Tear down with `docker compose down` (add `-v` to also drop the volume).

### Storing artifacts in S3 instead of a local volume

[`docker-compose.s3.yml`](docker-compose.s3.yml) runs the same server but points
`--artifacts-destination` at an S3 bucket instead of the local Docker volume.
Thanks to `--serve-artifacts`, the server **proxies** S3 for you — client code and
`load_model()` calls stay byte-for-byte identical and never need AWS credentials
(only the server does).

```bash
cp .env.s3.example .env                       # fill in bucket + AWS keys
docker compose -f docker-compose.s3.yml up -d # UI still at http://localhost:5000
```

The registry metadata (versions, `@champion`) still lives in a database — here
SQLite bind-mounted to `mlflow-s3.db` on the host so it survives `down` (swap for
Postgres in production; a commented service shows how). Register a version with
`python mlflow_example.py` as usual, then load it back with
[`mlflow_s3_registry_example.py`](mlflow_s3_registry_example.py) — whose code is
unchanged from the local example, which is the whole point.

> The real `.env` holds secrets and is git-ignored; only `.env.s3.example` is
> committed. Set `MLFLOW_S3_ENDPOINT_URL` to target MinIO / R2 / LocalStack for a
> fully local, no-AWS demo.

## 2. Run the training script

```bash
python mlflow_example.py
```

This will:

1. Generate synthetic ride data and split it into train/validation sets
   (`src/train.py`).
2. Log the hyperparameters (`n_estimators`, `max_depth`, …) and free-form tags.
3. Train a `RandomForestRegressor`.
4. Log the metrics (`rmse`, `mae`, `r2`, `val_size`).
5. Log a **learning curve** as stepped metrics (RMSE/MAE/R² vs. tree count) —
   an interactive line chart in the UI.
6. Log **diagnostic figures** as PNG artifacts (predicted-vs-actual, residuals,
   feature importances).
7. Log **and register** the fitted model as `RideDurationModel`, then annotate
   the new version with a description, tags, and the moving alias `@champion`.

It prints the run ID, MAE, and the registered version, e.g.:

```
Run ID: a1b2c3...  |  MAE: 0.83  |  R2: 0.99
Registered 'RideDurationModel' version 1 (alias: @champion)
```

Open **http://localhost:5000** to browse the experiment
(`ride-duration-model`), compare runs, and inspect the registered model.

> The script reads the tracking URI from the `MLFLOW_TRACKING_URI` env var,
> defaulting to `http://localhost:5000`. Point it at a remote server by
> exporting that variable.

### Consume the registered model

Once a version exists, [`mlflow_modelregiestry_example.py`](mlflow_modelregiestry_example.py)
is the consumer side: it lists the registered versions, promotes one by moving
an alias (`@champion` → `@production`), loads it via
`models:/RideDurationModel@production`, and runs a few predictions.

```bash
python mlflow_modelregiestry_example.py
```

## Tracking with Weights & Biases (alternative)

[`wandb_example.py`](wandb_example.py) mirrors `mlflow_example.py` feature for
feature against [W&B](https://wandb.ai/) — config/tags, scalar metrics, a stepped
learning curve, diagnostic figures, a predictions `wandb.Table`, a versioned
model **artifact** with a `champion` alias, and a Bayesian hyperparameter
**sweep**. It's the easiest way to see how the two trackers map onto each other.

```bash
pip install -e ".[wandb]"          # wandb + matplotlib

# Pick an auth mode first:
wandb login                        # log to wandb.ai (free account + API key), or
export WANDB_MODE=offline          # no account/network → runs written to ./wandb/

python wandb_example.py
```

> In offline mode everything runs locally **except the sweep** (sweeps are
> orchestrated by the W&B server and need a login); the script skips it
> gracefully. Push saved offline runs later with `wandb sync wandb/offline-run-*`.

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

The suite exercises `src/train.py` only (data generation, split, training,
evaluation) — it does **not** require a running MLflow server.

## Serving the model (inference API)

[`serve.py`](serve.py) is a small FastAPI app that loads the fitted model and
exposes two endpoints:

| Method & path  | Purpose                                                    |
|----------------|------------------------------------------------------------|
| `GET  /health` | Liveness probe (used by the Docker `HEALTHCHECK` and CI)   |
| `POST /predict`| Predict duration from `{"distance_km": …, "passengers": …}`|

The model file is chosen by the `MODEL_PATH` env var (default
`models/rf_model.pkl`, produced by the DVC `train` stage). Run it locally:

```bash
pip install -e ".[serve]"                    # fastapi + uvicorn
uvicorn serve:app --port 8000
curl localhost:8000/health
curl -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"distance_km": 10, "passengers": 2}'   # -> {"duration_min": 21.08}
```

Or build and run the container (multi-stage image, runs as non-root, bakes in
the model):

```bash
docker build -t ride-api .
docker run -p 8000:8000 ride-api
```

## Continuous Integration / Delivery (GitHub Actions)

The pipeline is defined in **`.github/workflows/ci_cd.yml`** at the **repository
root** (`mlops_sessions/`), not inside `session_2/` — GitHub only discovers
workflows at the repo root. Every `run:` step is scoped to `session_2/` via
`defaults.run.working-directory`.

| Job              | Trigger                              | What it does                                   |
|------------------|--------------------------------------|------------------------------------------------|
| `lint-and-test`  | push to `main`/`develop`, PR to `main` | Ruff lint/format check, pytest + 80% coverage  |
| `build-and-push` | push to `main` only                  | Build the Docker image and push to Docker Hub  |

### 1. Required secrets

Add these as **repository secrets** — repo → **Settings** → **Secrets and
variables** → **Actions** → *New repository secret*. They live on GitHub and are
never committed. (This is the **repo's** Settings tab, not your personal account
settings.)

| Secret               | Needed for            | Where to get it                                                                 |
|----------------------|-----------------------|---------------------------------------------------------------------------------|
| `DOCKERHUB_USERNAME` | `build-and-push`      | Your Docker Hub username (also becomes the image namespace `<user>/ride-api`)   |
| `DOCKERHUB_TOKEN`    | `build-and-push`      | hub.docker.com → Account settings → **Personal access tokens** → *Read & Write* |
| `CODECOV_TOKEN`      | `lint-and-test` (opt) | app.codecov.io → your repo → Settings → upload token                            |

Notes:

- `DOCKERHUB_TOKEN` is a Docker Hub **access token** (looks like `dckr_pat_…`),
  **not** your password. Scope it *Read & Write* so CI can push.
- `CODECOV_TOKEN` is optional — the upload step has `fail_ci_if_error: false`, so
  CI stays green without it. `lint-and-test` needs **no** secrets to run.

Set them from the terminal with the GitHub CLI (must be authenticated as the
repo owner — check with `gh auth status`):

```bash
gh secret set DOCKERHUB_USERNAME --body "your-dockerhub-username"
gh secret set DOCKERHUB_TOKEN      # prompts, paste the token (hidden)
gh secret set CODECOV_TOKEN        # optional
gh secret list                     # verify
```

### 2. Build & verify the Docker image locally (before triggering CI)

The `build-and-push` job just automates these steps — running them by hand first
confirms the image builds, serves correctly, and that your Docker Hub token
works, before you rely on CI.

```bash
cd session_2

# 1. Build the image (Dockerfile + app + baked model all live in session_2/)
docker build -t ride-api:local .

# 2. Run it and smoke-test the same endpoints CI health-checks
docker run -d --name ride-api -p 8000:8000 ride-api:local
curl localhost:8000/health                        # {"status":"ok"}
curl -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"distance_km": 10, "passengers": 2}'        # {"duration_min": 21.08}
docker rm -f ride-api

# 3. (Optional) push by hand to confirm the token/permissions work
docker login -u <your-dockerhub-username>          # paste the dckr_pat_… token as the password
docker tag ride-api:local <your-dockerhub-username>/ride-api:test
docker push <your-dockerhub-username>/ride-api:test
```

In CI the same build runs via `docker/build-push-action`, tagging every image
two ways and reusing cached layers (`cache-from: …:latest`):

| Tag                       | Example                      | Purpose                        |
|---------------------------|------------------------------|--------------------------------|
| `…/ride-api:<short-sha>`  | `ayanasser/ride-api:a1b2c3d` | Immutable, one per commit      |
| `…/ride-api:latest`       | `ayanasser/ride-api:latest`  | Always the newest `main` build |

### 3. Trigger the pipeline

The workflow fires automatically on the events in its `on:` block — you trigger
it with ordinary git pushes / PRs, there is no button to click:

| Event                    | How you cause it                        | Jobs that run                      |
|--------------------------|-----------------------------------------|------------------------------------|
| **PR targeting `main`**  | open a PR from your branch              | `lint-and-test` only               |
| **Push to `develop`**    | `git push origin develop`               | `lint-and-test` only               |
| **Push to `main`**       | `git push origin main` / merge a PR     | `lint-and-test` → `build-and-push` |

Typical flow — branch → PR (tests) → merge (tests + image build & push):

```bash
# 1. Work on a branch
git checkout -b my-change
git add -A && git commit -m "describe the change"
git push -u origin my-change

# 2. Open a PR to main  → runs lint-and-test
gh pr create --base main --fill          # or use the link git prints on push

# 3. Merge it           → push to main re-runs tests, then build-and-push
gh pr merge --squash --delete-branch
```

> `build-and-push` is gated on `if: github.ref == 'refs/heads/main'`, so the
> image is published **only after merge to `main`**, never from a PR — unreviewed
> code never reaches your registry.

There's no `workflow_dispatch` trigger, so the Actions tab has no *Run workflow*
button. To enable manual runs, add `workflow_dispatch:` under `on:` in
`ci_cd.yml`:

```yaml
on:
  workflow_dispatch:        # adds a "Run workflow" button in the Actions tab
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
```

### 4. Watch the results

```bash
gh run list                  # recent runs + pass/fail status
gh run watch                 # live-tail the in-progress run
gh run view --log-failed     # show logs for only the failed steps
```

Or open the repo's **Actions** tab in the browser. After a successful `main`
build, the image appears on Docker Hub under `<your-username>/ride-api`.

### 5. Dry-run the workflow locally with `act` (optional)

From the **repo root** (where `act` finds `.github/workflows/`):

```bash
cd ..                                    # into mlops_sessions/
act -l                                   # list detected jobs
act pull_request -j lint-and-test        # run the test job in Docker
```

## DVC pipeline

The reproducible training pipeline is defined in [`dvc.yaml`](dvc.yaml) (at the
project root) as three stages:

| Stage      | Command              | Inputs                                   | Outputs                         |
|------------|----------------------|------------------------------------------|---------------------------------|
| `prepare`  | `python src/prepare.py`  | `data/raw/rides.csv`, `config/config.yaml` | `data/processed/{train,val}.parquet` |
| `train`    | `python src/train.py`    | `train.parquet`, `model.*` params        | `models/rf_model.pkl`           |
| `evaluate` | `python src/evaluate.py` | `rf_model.pkl`, `val.parquet`            | `metrics/scores.json`           |

All hyperparameters live in [`config/config.yaml`](config/config.yaml); the
`train` stage tracks `model.n_estimators` and `model.max_depth` as DVC params,
so changing them invalidates the cache and triggers a re-run. Every stage's
recorded input/output hashes live in [`dvc.lock`](dvc.lock) — commit it so the
pipeline is reproducible for everyone.

Because `dvc.yaml` sits at the project root, all commands run without a path
argument. Install DVC and run the pipeline from the project root:

```bash
pip install -e ".[dvc]"           # DVC + remote support

dvc init --subdir                 # one-time, initialises DVC in session_2/

dvc repro                         # run prepare → train → evaluate
dvc status                        # show which stages are stale (deps changed)
dvc dag                           # visualise the stage graph
dvc metrics show                  # print metrics/scores.json
```

Re-running `dvc repro` only re-executes stages whose dependencies changed.

### Tracking the raw data

The raw dataset is tracked by DVC (pointer file `data/raw/rides.csv.dvc`), not
by Git. When the CSV changes, re-add it and push the new version:

```bash
dvc add data/raw/rides.csv        # update the .dvc pointer + cache
git add data/raw/rides.csv.dvc    # commit the pointer, not the CSV
```

> A file cannot be tracked by both Git and DVC. If `dvc add` reports the file is
> "already tracked by SCM", stop Git tracking it first:
> `git rm -r --cached data/raw/rides.csv`.

### Remote storage

Remotes are defined in [`.dvc/config`](.dvc/config). Three are configured — a
local directory (the default), a GCS bucket, and an S3 bucket:

```bash
dvc remote list                   # local (default), gcs, s3

dvc push                          # push to the default (local) remote
dvc push -r gcs                   # push to the GCS bucket
dvc push -r s3                    # push to the S3 bucket
dvc pull                          # pull tracked data/models from the default remote
```

Set or change a remote (and keep it the default with `-d`):

```bash
dvc remote add -d s3 s3://mlops-dvc-artifacts-<project_name>
dvc remote modify s3 url s3://my-bucket/dvc-store   # change the URL
```

Credentials must **not** go in `.dvc/config` (it is committed to Git). Put them
in the git-ignored `.dvc/config.local`:

```bash
dvc remote modify --local s3 access_key_id     YOUR_KEY
dvc remote modify --local s3 secret_access_key YOUR_SECRET
# GCS typically just uses: gcloud auth application-default login
```

The `infra/` Terraform provisions an S3 bucket
(`mlops-dvc-artifacts-<project_name>`) you can point the `s3` remote at.

## Infrastructure (Terraform)

The `infra/` folder provisions the AWS resources this project depends on:

- **S3 bucket** — remote storage for DVC artifacts
  (`mlops-dvc-artifacts-<project_name>`).
- **ECR repository** — registry for the API Docker image
  (`<project_name>/ride-api`, with scan-on-push enabled).

```
infra/
├── main.tf          # providers, S3 bucket, ECR repository, outputs
├── variables.tf     # input variables (project_name)
└── commands.md      # Terraform command cheatsheet
```

Requires the [Terraform CLI](https://developer.hashicorp.com/terraform/install)
and AWS credentials (e.g. via `aws configure` or environment variables).

### Standard Terraform workflow

```bash
cd infra

# 1. Initialise — download providers, setup backend
terraform init

# 2. Format — auto-format all .tf files
terraform fmt

# 3. Validate — check syntax without calling AWS
terraform validate

# 4. Plan — preview what will change (read-only)
terraform plan -out=tfplan

# 5. Apply — create/update resources
terraform apply tfplan

# 6. Show outputs (e.g. to get ECR URL for CI)
terraform output ecr_repo_url

# 7. Destroy — tear everything down (be careful!)
terraform destroy
```
