# House Price — Regression

Predicts house price from property attributes using a scikit-learn pipeline
with hyperparameter search, MLflow tracking, and a Flask serving app.

## Architecture

```
config (typed dataclass)  ->  helpers/config.py
ingest (local CSV) -> clean -> split -> preprocess+train -> evaluate -> serve
```

Reads `data/raw/dataset.csv` (falls back to the configured URL). All
preprocessing (log scaling of skewed size/distance features, standardization)
is embedded in a single persisted `sklearn.pipeline.Pipeline`.

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
python main.py && docker build -t houseprice .
docker run -p 8080:8080 houseprice
```

`MLFLOW_TRACKING_URI` and `MODEL_PATH` env vars override the defaults.
