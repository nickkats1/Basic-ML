# Airplane — Price Regression

Predicts aircraft price from specifications using a scikit-learn pipeline with
hyperparameter search, MLflow tracking, and a Flask serving app.

## Architecture

```
config (typed dataclass)  ->  helpers/config.py
ingest -> clean -> split -> preprocess+train (GridSearchCV) -> evaluate -> serve
```

All preprocessing (log scaling, standardization, encoding) is embedded in a
single persisted `sklearn.pipeline.Pipeline`, so training and serving apply
identical transforms to raw inputs. Every candidate model is logged to a local
MLflow (SQLite) store; the best pipeline is saved to `artifacts/best_model.pkl`.

## Setup

```bash
pip install -r requirements-dev.txt   # runtime + test/lint deps
```

## Train

```bash
python main.py            # ingest -> train -> evaluate -> sample prediction
mlflow ui                 # browse runs (sqlite:///mlflow.db)
```

Outputs: `artifacts/best_model.pkl`, `artifacts/metrics.json`,
`artifacts/model_card.md`.

## Serve

```bash
python app.py                          # http://localhost:8080
# or production:
gunicorn app:app -b 0.0.0.0:8080
curl localhost:8080/health
```

## Test & lint

```bash
pytest          # network-free unit + app tests
ruff check .
```

## Docker

```bash
python main.py                 # produce artifacts/best_model.pkl first
docker build -t airplane .
docker run -p 8080:8080 airplane
```

## Configuration

Edit `helpers/config.py` (the `Config` dataclass). `MLFLOW_TRACKING_URI` and
`MODEL_PATH` environment variables override the defaults at runtime.
