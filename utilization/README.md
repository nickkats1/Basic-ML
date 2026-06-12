# Utilization — Credit Utilization Regression

Predicts the log-odds of credit utilization from applicant and bureau data
using a scikit-learn pipeline with hyperparameter search, MLflow tracking, and
a Flask serving app.

## Architecture

```
config (typed dataclass)  ->  helpers/config.py
ingest (3 CSVs) -> clean/derive target -> split -> preprocess+train -> evaluate -> serve
```

The target `log_odds_utils` and the binary `homeownership` encoding are derived
in `feature_engineering.clean`; all column scaling/encoding lives inside a single
persisted `sklearn.pipeline.Pipeline`. The applicant identifier `ssn` is excluded
from the features.

> Note: `purchases` and `credit_limit` define the utilization ratio the target is
> derived from, so they are strongly predictive by construction.

## Setup

```bash
pip install -r requirements-dev.txt
```

## Train / serve / test

```bash
python main.py                         # ingest -> train -> evaluate
mlflow ui                              # browse runs (sqlite:///mlflow.db)
python app.py                          # serve on http://localhost:8080
gunicorn app:app -b 0.0.0.0:8080       # production
pytest                                 # network-free tests
ruff check .
```

Outputs: `artifacts/best_model.pkl`, `artifacts/metrics.json`,
`artifacts/model_card.md`.

## Docker

```bash
python main.py && docker build -t utilization .
docker run -p 8080:8080 utilization
```

`MLFLOW_TRACKING_URI` and `MODEL_PATH` env vars override the defaults.
