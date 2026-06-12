"""Model training with GridSearchCV, MLflow tracking, and pipeline persistence.

Every candidate is a full ``Pipeline([preprocess, model])`` so the persisted
best estimator transforms raw inputs end-to-end. Each candidate's run (params,
metrics, fitted model) is logged to a local MLflow (SQLite) tracking store.
"""

from typing import Any

import mlflow
import mlflow.sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    BaggingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from helpers.config import Config, get_config, save_object
from helpers.logger import logger


def candidate_models() -> dict[str, tuple[Any, dict[str, list]]]:
    """Return the model zoo as ``name -> (estimator, param_grid)``.

    Param-grid keys are bare estimator parameters; they are prefixed with
    ``model__`` when applied to the pipeline.
    """
    return {
        "LinearRegression": (
            LinearRegression(),
            {"fit_intercept": [True, False], "positive": [True, False]},
        ),
        "Lasso": (
            Lasso(max_iter=10000),
            {"alpha": [1e-3, 1e-2, 0.1, 1.0, 5.0, 10.0]},
        ),
        "Ridge": (
            Ridge(),
            {"alpha": [1e-3, 1e-2, 0.1, 1.0, 5.0, 10.0, 50.0]},
        ),
        "DecisionTreeRegressor": (
            DecisionTreeRegressor(random_state=42),
            {"max_depth": [None, 5, 10, 15], "min_samples_leaf": [1, 2, 5]},
        ),
        "RandomForestRegressor": (
            RandomForestRegressor(random_state=42),
            {"n_estimators": [100, 200], "max_depth": [5, 10, None]},
        ),
        "GradientBoostingRegressor": (
            GradientBoostingRegressor(random_state=42),
            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
        ),
        "BaggingRegressor": (
            BaggingRegressor(random_state=42),
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
    """Tune every candidate, log to MLflow, and persist the best pipeline.

    Returns:
        A summary dict with the best model name, metrics, and per-model results.
    """
    config = config or get_config()
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    results: list[dict[str, Any]] = []
    best_r2 = float("-inf")
    best_name: str | None = None
    best_pipeline: Pipeline | None = None

    for name, (estimator, grid) in candidate_models().items():
        pipeline = Pipeline(
            [("preprocess", clone(preprocessor)), ("model", estimator)]
        )
        param_grid = {f"model__{k}": v for k, v in grid.items()}
        search = GridSearchCV(
            pipeline, param_grid, cv=config.cv_folds, scoring="r2", n_jobs=-1
        )

        with mlflow.start_run(run_name=name):
            search.fit(X_train, y_train)
            y_pred = search.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(
                {"r2": r2, "rmse": rmse, "mae": mae, "cv_r2": search.best_score_}
            )
            mlflow.sklearn.log_model(search.best_estimator_, name="model")

            logger.info("%s -> r2=%.4f rmse=%.4f mae=%.4f", name, r2, rmse, mae)
            results.append(
                {
                    "model": name,
                    "r2": r2,
                    "rmse": rmse,
                    "mae": mae,
                    "cv_r2": search.best_score_,
                    "best_params": search.best_params_,
                }
            )

            if r2 > best_r2:
                best_r2, best_name, best_pipeline = r2, name, search.best_estimator_

    if best_pipeline is None:
        raise RuntimeError("No model was successfully trained.")

    save_object(config.model_path, best_pipeline)
    logger.info("best model: %s (holdout r2=%.4f)", best_name, best_r2)
    return {"best_model": best_name, "best_r2": best_r2, "results": results}


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
