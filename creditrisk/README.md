# Credit Risk — Default Classification

Predicts loan default from applicant/bureau attributes using a scikit-learn
classification pipeline with hyperparameter search, MLflow tracking, and a Flask
serving app. Scikit-learn models only (per the repo rules).

## Architecture

```
config (typed dataclass)  ->  helpers/config.py
ingest -> clean -> split -> preprocess (scale + one-hot) + train -> evaluate -> serve
```

Mixed numeric/categorical features are handled by a `ColumnTransformer`
(StandardScaler for numerics, OneHotEncoder for categoricals) embedded in a
single persisted `sklearn.pipeline.Pipeline`. The best model is chosen by
holdout ROC-AUC.

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

`/predict` returns the predicted probability of default. Outputs:
`artifacts/best_model.pkl`, `artifacts/metrics.json`, `artifacts/model_card.md`.

## Docker

```bash
python main.py && docker build -t creditrisk .
docker run -p 8080:8080 creditrisk
```

`MLFLOW_TRACKING_URI` and `MODEL_PATH` env vars override the defaults.
