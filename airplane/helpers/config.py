"""Typed, immutable project configuration and artifact (de)serialization.

Replaces the previous ``config.yaml``: configuration is now a frozen
dataclass so values are typed, discoverable, and validated at construction.
``get_config(**overrides)`` returns the config (optionally overridden, e.g. in
tests), reading a couple of standard environment variables for deployment.
"""

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import joblib

from helpers.logger import logger


@dataclass(frozen=True, slots=True)
class Config:
    """All configuration for the airplane regression pipeline."""

    # Raw data sources.
    airplane_sales_link: str = (
        "https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/"
        "refs/heads/main/final_exam_2020/airplane_sales.csv"
    )
    airplane_specs_link: str = (
        "https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/"
        "refs/heads/main/final_exam_2020/airplane_specs.csv"
    )
    airplane_perf_link: str = (
        "https://raw.githubusercontent.com/LeeMorinUCF/QMB6358F23/"
        "refs/heads/main/final_exam_2020/airplane_perf.csv"
    )

    # Schema.
    features: tuple[str, ...] = (
        "age",
        "pass",
        "fixgear",
        "tdrag",
        "horse",
        "fuel",
        "ceiling",
        "cruise",
    )
    target: str = "price"
    log_features: tuple[str, ...] = ("horse", "fuel", "ceiling", "cruise")

    # Split / training.
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 4

    # Artifacts / experiment tracking.
    model_path: str = "artifacts/best_model.pkl"
    experiment_name: str = "airplane"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"

    # Serving.
    app_title: str = "Airplane Price Prediction"

    def __post_init__(self) -> None:
        if not 0.0 < self.test_size < 1.0:
            raise ValueError(f"test_size must be in (0, 1); got {self.test_size}")
        if self.cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2; got {self.cv_folds}")
        if self.target in self.features:
            raise ValueError("target must not appear in features")


def get_config(**overrides: Any) -> Config:
    """Return the project configuration, applying overrides and env vars."""
    env_overrides: dict[str, Any] = {}
    if "MLFLOW_TRACKING_URI" in os.environ:
        env_overrides["mlflow_tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]
    if "MODEL_PATH" in os.environ:
        env_overrides["model_path"] = os.environ["MODEL_PATH"]
    return replace(Config(), **{**env_overrides, **overrides})


def save_object(file_path: str | Path, obj: Any) -> None:
    """Persist a Python object (e.g. a fitted pipeline) with joblib."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info("Saved object to %s", path)


def load_object(file_path: str | Path) -> Any:
    """Load a joblib-serialized object, raising a clear error if missing."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}. Run the training pipeline first."
        )
    return joblib.load(path)
