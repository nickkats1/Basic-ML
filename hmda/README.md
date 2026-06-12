# HMDA — Loan Approval Classification

Predicts mortgage loan approval from HMDA survey data using a scikit-learn
classification pipeline with hyperparameter search, MLflow tracking, and a Flask
serving app.

## Architecture

```
config (typed dataclass)  ->  helpers/config.py
ingest (TSV) -> rename/derive/encode -> split -> preprocess+train -> evaluate -> serve
```

`feature_engineering.clean` renames the raw `sNN` survey columns, derives the
binary `approved` target, and encodes categorical fields. All scaling lives
inside a single persisted `sklearn.pipeline.Pipeline`. The best model is chosen
by holdout ROC-AUC.

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

`/predict` returns the predicted probability of approval. Outputs:
`artifacts/best_model.pkl`, `artifacts/metrics.json`, `artifacts/model_card.md`.

## Docker

```bash
python main.py && docker build -t hmda .
docker run -p 8080:8080 hmda
```

`MLFLOW_TRACKING_URI` and `MODEL_PATH` env vars override the defaults.
