"""Classification training with GridSearchCV, MLflow tracking, and persistence.

Every candidate is a full ``Pipeline([preprocess, model])`` so the persisted
best estimator transforms raw inputs end-to-end. Each candidate's run (params,
metrics, fitted model) is logged to a local MLflow (SQLite) tracking store.
Scikit-learn models only (per the repository rules) — no XGBoost.
"""

from typing import Any

import mlflow
import mlflow.sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from helpers.config import Config, get_config, save_object
from helpers.logger import logger


def candidate_models() -> dict[str, tuple[Any, dict[str, list]]]:
    """Return the classifier zoo as ``name -> (estimator, param_grid)``.

    Param-grid keys are bare estimator parameters; they are prefixed with
    ``model__`` when applied to the pipeline.
    """
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=5000),
            {"C": [0.01, 0.1, 1.0, 10.0], "solver": ["liblinear", "lbfgs"]},
        ),
        "DecisionTreeClassifier": (
            DecisionTreeClassifier(random_state=42),
            {"max_depth": [None, 5, 10], "min_samples_leaf": [1, 2, 5]},
        ),
        "RandomForestClassifier": (
            RandomForestClassifier(random_state=42),
            {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
        ),
        "GradientBoostingClassifier": (
            GradientBoostingClassifier(random_state=42),
            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
        ),
        "KNeighborsClassifier": (
            KNeighborsClassifier(),
            {"n_neighbors": [5, 9, 15], "weights": ["uniform", "distance"]},
        ),
        "BaggingClassifier": (
            BaggingClassifier(random_state=42),
            {"n_estimators": [50, 100], "max_samples": [0.8, 1.0]},
        ),
    }


def train(
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    preprocessor: ColumnTransformer,
    config: Config | None = None,
) -> dict[str, Any]:
    """Tune every classifier, log to MLflow, and persist the best pipeline.

    Returns:
        A summary dict with the best model name, metrics, and per-model results.
    """
    config = config or get_config()
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    results: list[dict[str, Any]] = []
    best_auc = float("-inf")
    best_name: str | None = None
    best_pipeline: Pipeline | None = None

    for name, (estimator, grid) in candidate_models().items():
        pipeline = Pipeline(
            [("preprocess", clone(preprocessor)), ("model", estimator)]
        )
        param_grid = {f"model__{k}": v for k, v in grid.items()}
        search = GridSearchCV(
            pipeline, param_grid, cv=config.cv_folds, scoring="roc_auc", n_jobs=-1
        )

        with mlflow.start_run(run_name=name):
            search.fit(X_train, y_train)
            y_pred = search.predict(X_test)
            y_proba = search.predict_proba(X_test)[:, 1]

            auc = roc_auc_score(y_test, y_proba)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(
                {"roc_auc": auc, "accuracy": acc, "f1": f1, "cv_auc": search.best_score_}
            )
            mlflow.sklearn.log_model(search.best_estimator_, name="model")

            logger.info("%s -> auc=%.4f acc=%.4f f1=%.4f", name, auc, acc, f1)
            results.append(
                {
                    "model": name,
                    "roc_auc": auc,
                    "accuracy": acc,
                    "f1": f1,
                    "cv_auc": search.best_score_,
                    "best_params": search.best_params_,
                }
            )

            if auc > best_auc:
                best_auc, best_name, best_pipeline = auc, name, search.best_estimator_

    if best_pipeline is None:
        raise RuntimeError("No model was successfully trained.")

    save_object(config.model_path, best_pipeline)
    logger.info("best model: %s (holdout roc_auc=%.4f)", best_name, best_auc)
    return {"best_model": best_name, "best_roc_auc": best_auc, "results": results}


def run_training(config: Config | None = None) -> dict[str, Any]:
    """Run the full training flow end-to-end from configuration."""
    from src.data.data_ingestion import fetch_raw_data
    from src.data.data_transformation import build_preprocessor, split
    from src.data.feature_engineering import clean

    config = config or get_config()
    data = clean(fetch_raw_data(config), config)
    X_train, X_test, y_train, y_test = split(data, config)
    preprocessor = build_preprocessor(X_train, config)
    return train(X_train, X_test, y_train, y_test, preprocessor, config)
