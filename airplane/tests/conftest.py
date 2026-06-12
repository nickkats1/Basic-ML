"""Shared pytest fixtures.

Tests are network-free: a synthetic dataset stands in for the remote CSVs, and
a single session-scoped training run produces an artifact reused across tests.
"""

import numpy as np
import pandas as pd
import pytest

from helpers.config import get_config


def _synthetic_frame(n: int = 80) -> pd.DataFrame:
    """Build a synthetic dataset with the configured schema."""
    cfg = get_config()
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({col: rng.uniform(1, 100, n) for col in cfg.features})
    # A learnable linear relationship plus noise.
    frame[cfg.target] = 10 + 2.0 * frame[cfg.features[0]] + rng.normal(0, 3, n)
    return frame


@pytest.fixture
def raw_data() -> pd.DataFrame:
    """A fresh synthetic raw dataset."""
    return _synthetic_frame()


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Train once on synthetic data and expose the config + holdout split."""
    out = tmp_path_factory.mktemp("artifacts")
    cfg = get_config(
        model_path=str(out / "model.pkl"),
        mlflow_tracking_uri=f"sqlite:///{out / 'mlflow.db'}",
        experiment_name="test",
        cv_folds=2,
    )

    from src.data.data_transformation import build_preprocessor, split
    from src.data.feature_engineering import clean
    from src.models.model_trainer import train

    data = clean(_synthetic_frame(), cfg)
    X_train, X_test, y_train, y_test = split(data, cfg)
    preprocessor = build_preprocessor(X_train, cfg)
    summary = train(X_train, X_test, y_train, y_test, preprocessor, cfg)
    return {
        "config": cfg,
        "summary": summary,
        "X_test": X_test,
        "y_test": y_test,
    }
