# Sklearn Only

A monorepo of small, self-contained scikit-learn projects. Each one is a full
ML pipeline — data → preprocessing → model selection → evaluation → serving —
built to a single consistent standard.

## Rules
- Rule #1: Every project must only contain scikit-learn models.
- Rule #2: No NLP projects.
- Rule #3: Every project must be a pipeline (data → preprocessing → model → evaluation → optionally deployment).

## Projects

| Project        | Task                         | Data source        | Serves            |
|----------------|------------------------------|--------------------|-------------------|
| `airplane`     | Regression (aircraft price)  | 3 remote CSVs      | predicted price   |
| `houseprice`   | Regression (house price)     | local CSV          | predicted price   |
| `utilization`  | Regression (credit util.)    | 3 remote CSVs      | predicted target  |
| `creditrisk`   | Classification (default)     | remote CSV         | default probability |
| `hmda`         | Classification (loan approval) | remote TSV       | approval probability |

## Shared architecture

Every project follows the same layout and conventions:

```
<project>/
├── helpers/
│   ├── config.py      # typed, immutable Config dataclass + get_config() + joblib IO
│   └── logger.py      # single configured logger
├── src/
│   ├── data/          # data_ingestion, feature_engineering (clean), data_transformation
│   └── models/        # model_trainer, model_evaluation, predict
├── tests/             # network-free pytest suite (synthetic fixtures)
├── app.py             # Flask serving app (/, /predict, /health)
├── main.py            # end-to-end training entry point
├── Dockerfile         # slim, non-root, gunicorn, healthcheck
├── requirements.txt   # runtime deps
└── requirements-dev.txt
```

Engineering choices applied across all projects:

- **Typed config, no YAML.** Configuration is a frozen `dataclass` validated at
  construction; `get_config(**overrides)` supports test overrides and the
  `MLFLOW_TRACKING_URI` / `MODEL_PATH` environment variables.
- **Function modules, not stateful classes.** Pipeline stages are plain functions
  taking `config`.
- **One persisted pipeline.** All preprocessing (log scaling, standardization,
  one-hot encoding) lives inside a single `sklearn.pipeline.Pipeline`, fit on the
  training split only and persisted with the model — training and serving apply
  identical transforms to raw inputs.
- **Experiment tracking.** Every candidate model is logged to a local MLflow
  (SQLite) store with params, metrics, and the fitted pipeline; the best model is
  chosen on a held-out split.
- **Reproducible outputs.** Each run writes `artifacts/best_model.pkl`,
  `artifacts/metrics.json`, and `artifacts/model_card.md`.
- **Tested & linted.** Network-free pytest suites and `ruff`, gated by GitHub
  Actions (`.github/workflows/ci.yml`).

## Quickstart

```bash
make install PROJECT=airplane     # deps
make train   PROJECT=airplane     # ingest -> train -> evaluate (writes artifacts/)
make serve   PROJECT=airplane     # gunicorn on :8080
make test-all                     # pytest across every project
make lint-all                     # ruff across every project
```

Browse experiments with `mlflow ui` from inside a project (uses
`sqlite:///mlflow.db`).

### Docker

```bash
make train PROJECT=airplane && make docker-build PROJECT=airplane
# or build & run every service (distinct ports 8081-8085):
docker compose up --build
```

## Note
This repository is no longer actively maintained, but the projects are kept
clean, tested, and reproducible.
