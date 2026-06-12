"""Shared pytest fixtures.

Tests are network-free: a synthetic dataset carrying the *raw* survey-code
columns (``sNN``) stands in for the remote TSV, and a single session-scoped
training run produces an artifact reused across tests.
"""

import numpy as np
import pandas as pd
import pytest

from helpers.config import get_config


def _synthetic_raw(n: int = 200) -> pd.DataFrame:
    """Build a synthetic raw dataset prior to feature engineering."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "s5": rng.integers(1, 4, n),  # occupancy
            "s7": rng.integers(1, 4, n),  # approval code (3 == approved)
            "s13": rng.integers(1, 4, n),  # race
            "s15": rng.integers(1, 3, n),  # sex
            "s17": rng.uniform(10, 200, n),  # income
            "s23a": rng.choice(["M", "U"], n),  # married
            "s40": rng.integers(1, 3, n),  # credit history
            "s43": rng.uniform(0, 5, n),  # chcp
            "s44": rng.uniform(0, 5, n),  # chpr
            "s45": rng.uniform(0, 100, n),  # debt_to_expense
            "s46": rng.uniform(0, 1, n),  # di_ratio
            "s52": rng.integers(0, 2, n),  # pmi_sought
            "s53": rng.integers(0, 2, n),  # pmi_denied
            "s56": rng.integers(0, 2, n),  # unverifiable
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
