"""Train/test splitting and preprocessing pipeline construction.

The preprocessor built here is the *only* place feature transforms are
defined. It is embedded inside every candidate model pipeline, fit on the
training split only, and persisted with the chosen model so serving applies
identical transforms to raw inputs.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from helpers.config import Config, get_config
from helpers.logger import logger


def split(
    data: pd.DataFrame, config: Config | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target with a single aligned train/test split.

    Returns:
        ``(X_train, X_test, y_train, y_test)``.
    """
    config = config or get_config()
    X = data[list(config.features)].copy()
    y = data[config.target].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )
    logger.info("split shapes -> X_train=%s X_test=%s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


def build_preprocessor(
    X: pd.DataFrame, config: Config | None = None
) -> ColumnTransformer:
    """Build a ColumnTransformer driven by dtypes and ``log_features``.

    - ``log_features`` (numeric): ``log1p`` then standardize.
    - other numeric columns: standardize.
    - categorical columns: one-hot encode.
    """
    config = config or get_config()
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    log_features = [c for c in config.log_features if c in numeric]
    plain_numeric = [c for c in numeric if c not in log_features]

    transformers: list[tuple] = []
    if log_features:
        log_pipe = Pipeline(
            [
                ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("log", log_pipe, log_features))
    if plain_numeric:
        transformers.append(("num", StandardScaler(), plain_numeric))
    if categorical:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
        )

    logger.info(
        "preprocessor: log=%s numeric=%s categorical=%s",
        log_features,
        plain_numeric,
        categorical,
    )
    return ColumnTransformer(transformers, remainder="drop")
