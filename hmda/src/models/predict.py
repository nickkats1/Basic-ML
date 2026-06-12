"""Inference using the persisted end-to-end classification pipeline."""

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

import pandas as pd

from helpers.config import Config, get_config, load_object
from helpers.logger import logger


@lru_cache(maxsize=4)
def _load_pipeline(model_path: str) -> Any:
    """Load and cache the persisted pipeline by path."""
    return load_object(model_path)


def _to_frame(features: Any, columns: list[str]) -> pd.DataFrame:
    """Coerce dict / sequence / DataFrame input into an ordered frame."""
    if isinstance(features, pd.DataFrame):
        return features[columns]
    if isinstance(features, Mapping):
        return pd.DataFrame([features])[columns]
    if isinstance(features, Sequence) and not isinstance(features, str):
        return pd.DataFrame([dict(zip(columns, features, strict=True))])
    raise TypeError(f"Unsupported feature input type: {type(features)!r}")


def feature_schema(config: Config | None = None) -> tuple[list[str], list[str]]:
    """Return ``(numeric_features, categorical_features)`` from the pipeline.

    Inspects the persisted preprocessor's fitted transformers. If no model has
    been trained yet, every feature is treated as numeric.
    """
    config = config or get_config()
    try:
        pipeline = _load_pipeline(config.model_path)
    except FileNotFoundError:
        return list(config.features), []

    categorical: list[str] = []
    preprocess = pipeline.named_steps.get("preprocess")
    if preprocess is not None and hasattr(preprocess, "transformers_"):
        for name, _transformer, columns in preprocess.transformers_:
            if name == "cat":
                categorical.extend(columns)
    categorical_set = set(categorical)
    numeric = [c for c in config.features if c not in categorical_set]
    return numeric, categorical


def predict(features: Any, config: Config | None = None) -> float:
    """Predict the probability of the positive class for raw features.

    Args:
        features: A mapping, sequence, or single-row DataFrame of raw features.
        config: Project configuration. Defaults to :func:`get_config`.

    Returns:
        The predicted positive-class probability, rounded to 4 decimals.
    """
    config = config or get_config()
    pipeline = _load_pipeline(config.model_path)
    X = _to_frame(features, list(config.features))
    proba = float(pipeline.predict_proba(X)[0, 1])
    logger.info("predicted probability: %.4f", proba)
    return round(proba, 4)
