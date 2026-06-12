"""Shared pytest fixtures.

Tests are network-free: a synthetic dataset stands in for the CSV, and a
single session-scoped training run produces an artifact reused across tests.
"""

import numpy as np
import pandas as pd
import pytest

from helpers.config import get_config


def _synthetic_frame(n: int = 100) -> pd.DataFrame:
    """Build a synthetic dataset with the configured schema."""
    cfg = get_config()
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "year": rng.integers(1950, 2020, n),
            "age": rng.integers(0, 70, n),
            "beds": rng.integers(1, 6, n),
            "baths": rng.integers(1, 4, n),
            "home_size": rng.uniform(800, 5000, n),
            "parcel_size": rng.uniform(3000, 20000, n),
            "pool": rng.integers(0, 2, n),
            "dist_cbd": rng.uniform(1000, 30000, n),
            "dist_lakes": rng.uniform(1000, 20000, n),
            "x_coord": rng.uniform(500000, 600000, n),
            "y_coord": rng.uniform(1400000, 1600000, n),
        }
    )
    frame[cfg.target] = (
        50_000 + 60.0 * frame["home_size"] + 8000 * frame["beds"]
        + rng.normal(0, 10_000, n)
    )
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
    return {"config": cfg, "summary": summary, "X_test": X_test, "y_test": y_test}
