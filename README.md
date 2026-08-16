# NYC Green Taxi Trip Duration MLOps

Portfolio project for AI Engineer, Data Science, and ML Engineer internship roles. The project predicts NYC Green Taxi trip duration in minutes and demonstrates a reproducible ML pipeline with DVC, MLflow, XGBoost, FastAPI, Docker, and Evidently.

[![Quality Gates](https://github.com/duylw/mlops-practice-01/actions/workflows/ci.yml/badge.svg)](https://github.com/duylw/mlops-practice-01/actions/workflows/ci.yml)
[![Continuous Delivery](https://github.com/duylw/mlops-practice-01/actions/workflows/cd.yml/badge.svg)](https://github.com/duylw/mlops-practice-01/actions/workflows/cd.yml)

## Architecture

```mermaid
flowchart LR
    A[NYC TLC Data] --> B[DVC Pipeline: Ingest & Split]
    B --> C[Sklearn Pipeline: Features + XGBoost]
    C --> D[MLflow Tracking & Registry: @champion]
    D --> E[FastAPI Serving Container]
    E --> F[Evidently AI Drift Monitoring]
    F -.->|Retrain Trigger| B
```


## Project Structure

```text
src/
  config/      path and environment settings
  data/        ingest, split, target preparation
  features/    stateless engineering and sklearn transformer
  training/    pipeline training, MLflow tracking, registry promotion
  monitoring/  Evidently batch monitoring for registered models
  serving/     FastAPI app, schemas, MLflow model loader
  utils/       geo and IO helpers
tests/         unit and schema tests
reports/       DVC metrics and model card
```

## Setup

```bash
uv sync
```

`uv sync` installs every local capability group. The serving Docker image installs only the base runtime and `serving` group:

```bash
uv sync --no-default-groups --group serving
```

Dependency groups:

| Group | Purpose |
|---|---|
| Base | MLflow model loading, sklearn pipeline, XGBoost, and shared configuration. |
| `serving` | FastAPI and Uvicorn. |
| `training` | DVC, ingestion, feature reports, and Optuna tuning. |
| `monitoring` | Evidently and Parquet monitoring input. |
| `dev` | Pytest, Ruff, and HTTP benchmark tests. |

Start MLflow before running the full DVC pipeline:

```bash
docker compose up mlflow-server
```

In another terminal:

```bash
uv run dvc repro
```

Tracked outputs:

- `reports/mlflow_run.json`
- `reports/metrics.json`
- `reports/registry.json`

The registered MLflow model contains both the fitted feature transformer and XGBoost estimator. There is no standalone preprocessor artifact.

## Training Pipeline

```bash
python -m src.data.ingest
python -m src.data.split
python -m src.training.train
python -m src.training.register_model
```

`src.training.train` fits and evaluates one sklearn Pipeline from raw chronological splits, then logs that complete pipeline to MLflow. Primary metrics are RMSE and MAE in minutes. DVC reads them from:

```bash
dvc metrics show
```

Latest local champion (chronological test split):

| Metric | Result |
|---|---:|
| RMSE | 9.10 minutes |
| MAE | 4.45 minutes |
| In-process pipeline latency | 0.007 ms/row |

The final configuration was selected with 20 Optuna trials on validation RMSE only; the test split was used once for final evaluation. Run tuning separately when experimenting:

```bash
uv run python -m src.training.tune --n-trials 20
```

Copy the selected parameters from `reports/tuning.json` to `params.yaml`, then run `uv run dvc repro` to train and register the candidate.

## Serving

After a model has been registered and promoted:

```bash
docker compose up --build
```

Health and metadata:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metadata
```

Prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @data_sample.json
```

Example request:

```json
{
  "trips": [
    {
      "VendorID": 2,
      "lpep_pickup_datetime": "2026-01-15 08:30:00",
      "PULocationID": 74,
      "DOLocationID": 42,
      "passenger_count": 1,
      "trip_type": 1
    }
  ]
}
```

### Serving Validation

```bash
uv run python scripts/benchmark_serving.py --request-count 20
```

The benchmark tests batch sizes `1`, `10`, and `100` at concurrency `1`, `5`, and `10`. It stores aggregate success rate, throughput, and p50/p95/p99 latency in ignored `reports/serving_benchmark.json`; request payloads are never saved.

Example response:

```json
{
  "predictions": [12.34],
  "model_name": "green_taxi_duration_model",
  "model_version": 1,
  "model_alias": "champion",
  "latency": 0.015
}
```

## CI/CD Pipeline & Quality Gates

The repository implements an automated **CI/CD/CT** lifecycle with GitHub Actions:

- **CI (`.github/workflows/ci.yml`)**: Triggered on all PRs and pushes to `main`.
  - **Quality Gates**: Linting (`ruff`), 19 unit tests (`pytest`), and syntax checks (`compileall`).
  - **Docker Dry-Run**: Builds the serving container using Docker Buildx to verify packaging integrity before merge.
- **CD (`.github/workflows/cd.yml`)**: Triggered on merge to `main` and release tags (`v*.*.*`).
  - Builds and releases multi-tag Docker images to **GitHub Container Registry (GHCR)**: `ghcr.io/duylw/mlops-serving-api:latest`.
  - Employs GitHub Actions cache (`type=gha`) for fast builds.

Run local verification suite:

```bash
uv run ruff check src tests scripts
uv run pytest
uv run python -m compileall src
```


## Batch Monitoring

Evidently evaluates a current Parquet batch against the DVC-managed training and
validation splits associated with the registered `champion` model. It does not
copy Parquet files into MLflow. Training logs Git/DVC lineage and split time-range tags;
retrain and register the champion once before the first monitoring run.

```bash
python -m src.monitoring.run \
  --current-path data/split/test_raw.parquet \
  --current-name test-2026-04
```

The command creates ignored HTML and JSON reports under `reports/monitoring/`
and logs them with Evidently-derived metrics to the
`green_taxi_duration_monitoring` MLflow experiment. A current file containing
only the six API input fields runs input drift monitoring without regression
quality metrics.

## Portfolio Notes

| Focus | What this project demonstrates |
|---|---|
| Data Science | Leakage-aware target preparation, chronological validation, Optuna tuning, and segment-level error analysis. |
| ML Engineering | DVC reproducibility, MLflow tracking/registry, one fitted sklearn pipeline, tests, and monitoring lineage. |
| AI Engineering | FastAPI prediction contract, Dockerized serving, model readiness checks, and request benchmark tooling. |

See `reports/model_card.md` for modeling details and `docs/portfolio-readiness/` for the completed local-first roadmap.

## Future Cloud Extension

Not implemented in this repository. A future iteration may add an S3 DVC remote, hosted MLflow, and CI/CD deployment. The current portfolio intentionally stays local-first so every demonstrated capability can be reproduced and explained without cloud credentials or cost.
