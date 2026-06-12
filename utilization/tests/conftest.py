"""Shared pytest fixtures.

Tests are network-free: a synthetic dataset stands in for the remote CSVs, and
a single session-scoped training run produces an artifact reused across tests.
The synthetic frame carries the *raw* columns that ``clean`` derives the target
and encodings from.
"""

import numpy as np
import pandas as pd
import pytest

from helpers.config import get_config


def _synthetic_raw(n: int = 120) -> pd.DataFrame:
    """Build a synthetic raw dataset prior to feature engineering."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "income": rng.uniform(20_000, 120_000, n),
            "homeownership": rng.choice(["Rent", "Own"], n),
            "purchases": rng.uniform(500, 5_000, n),
            "credit_limit": rng.uniform(6_000, 20_000, n),
            "fico": rng.uniform(550, 820, n),
            "num_late": rng.integers(0, 10, n),
            "past_def": rng.integers(0, 3, n),
            "num_bankruptcy": rng.integers(0, 2, n),
            "avg_income": rng.uniform(20_000, 120_000, n),
            "density": rng.uniform(0, 1, n),
        }
    )


@pytest.fixture
def raw_data() -> pd.DataFrame:
    """A fresh synthetic raw dataset prior to feature engineering."""
    return _synthetic_raw()


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

    data = clean(_synthetic_raw(), cfg)
    X_train, X_test, y_train, y_test = split(data, cfg)
    preprocessor = build_preprocessor(X_train, cfg)
    summary = train(X_train, X_test, y_train, y_test, preprocessor, cfg)
    return {"config": cfg, "summary": summary, "X_test": X_test, "y_test": y_test}
