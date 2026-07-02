# Ride Duration — MLflow Tracking (Session 2)

Session 2 of the MLOps course. It trains a small **ride-duration** regression
model on synthetic data and tracks the run — parameters, metrics, and the fitted
model — with [MLflow](https://mlflow.org/). The model reuses the same
distance/passenger relationship from session 1.

## Project structure

```
session_2/
├── mflow_example.py       # MLflow training + logging script (entry point)
├── pyproject.toml         # Project metadata + Python dependencies
├── docker-compose.yml     # MLflow tracking server (from session 1)
├── config/
│   ├── config.yaml        # pipeline parameters (data + model)
│   └── dvc.yaml           # DVC pipeline definition (prepare → train → evaluate)
├── data/
│   └── raw/rides.csv      # raw synthetic dataset (pipeline input)
├── src/
│   ├── __init__.py
│   ├── train.py           # data gen, train, evaluate + DVC `train` stage
│   ├── prepare.py         # DVC `prepare` stage (raw CSV → train/val parquet)
│   ├── evaluate.py        # DVC `evaluate` stage (model → metrics/scores.json)
│   └── config.py          # YAML config loader
├── tests/
│   └── test_train.py      # pytest suite for src/train.py
├── infra/                 # Terraform IaC (AWS: S3 DVC remote + ECR repo)
│   ├── main.tf            # providers, S3 bucket, ECR repository, outputs
│   ├── variables.tf       # input variables
│   └── commands.md        # Terraform command cheatsheet
├── .github/workflows/     # CI (lint + test)
├── .pre-commit-config.yaml
└── README.md
```

> `src/train.py` deliberately contains **no MLflow calls** so it can be unit
> tested in isolation. All experiment tracking lives in `mflow_example.py`.

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

## 2. Run the training script

```bash
python mflow_example.py
```

This will:

1. Generate synthetic ride data and split it into train/validation sets
   (`src/train.py`).
2. Log the hyperparameters (`n_estimators`, `max_depth`, …).
3. Train a `RandomForestRegressor`.
4. Log the metrics (`rmse`, `mae`, `r2`, `val_size`).
5. Log **and register** the fitted model as `RideDurationModel`.

It prints the run ID and MAE, e.g.:

```
Run ID: a1b2c3...  |  MAE: 0.83  |  R2: 0.99
```

Open **http://localhost:5000** to browse the experiment
(`ride-duration-model`), compare runs, and inspect the registered model.

> The script reads the tracking URI from the `MLFLOW_TRACKING_URI` env var,
> defaulting to `http://localhost:5000`. Point it at a remote server by
> exporting that variable.

## Running tests

```bash
pip install -e ".[dev]"
pytest
```

The suite exercises `src/train.py` only (data generation, split, training,
evaluation) — it does **not** require a running MLflow server.

## DVC pipeline

The reproducible training pipeline is defined in
[`config/dvc.yaml`](config/dvc.yaml) as three stages:

| Stage      | Command              | Inputs                                   | Outputs                         |
|------------|----------------------|------------------------------------------|---------------------------------|
| `prepare`  | `python src/prepare.py`  | `data/raw/rides.csv`, `config/config.yaml` | `data/processed/{train,val}.parquet` |
| `train`    | `python src/train.py`    | `train.parquet`, `model.*` params        | `models/rf_model.pkl`           |
| `evaluate` | `python src/evaluate.py` | `rf_model.pkl`, `val.parquet`            | `metrics/scores.json`           |

All hyperparameters live in [`config/config.yaml`](config/config.yaml); the
`train` stage tracks `model.n_estimators` and `model.max_depth` as DVC params,
so changing them invalidates the cache and triggers a re-run.

> `dvc.yaml` sits in `config/` but each stage uses `wdir: ..`, so every path is
> relative to the project root.

Install DVC and run the pipeline from the project root:

```bash
pip install -e ".[dvc]"           # DVC + S3 remote support

dvc init --subdir                 # one-time, initialises DVC in session_2/
dvc repro config/dvc.yaml         # run prepare → train → evaluate
dvc metrics show                  # print metrics/scores.json
dvc dag config/dvc.yaml           # visualise the stage graph
```

Re-running `dvc repro` only re-executes stages whose dependencies changed.

### Remote storage (optional)

The `infra/` Terraform provisions an S3 bucket for DVC artifacts. Wire it up
with:

```bash
dvc remote add -d storage s3://mlops-dvc-artifacts-<project_name>
dvc push                          # upload tracked data/models to S3
```

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
