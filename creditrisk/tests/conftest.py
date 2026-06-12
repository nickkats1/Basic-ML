"""Shared pytest fixtures.

Tests are network-free: a synthetic German-credit-style dataset (mixed numeric
and categorical columns) stands in for the remote CSV, and a single
session-scoped training run produces an artifact reused across tests.
"""

import numpy as np
import pandas as pd
import pytest

from helpers.config import get_config

_NUMERIC = ("duration", "amount", "installment", "residence", "age", "cards", "liable")
_CATEGORICAL = (
    "checkingstatus1",
    "history",
    "purpose",
    "savings",
    "employ",
    "status",
    "others",
    "property",
    "otherplans",
    "housing",
    "job",
    "tele",
    "foreign",
)


def _synthetic_frame(n: int = 160) -> pd.DataFrame:
    """Build a synthetic dataset with the configured schema and a signal."""
    rng = np.random.default_rng(0)
    data = {col: rng.integers(1, 50, n) for col in _NUMERIC}
    data["amount"] = rng.uniform(250, 18_000, n)
    for col in _CATEGORICAL:
        data[col] = rng.choice([f"A{col[:2]}{i}" for i in range(3)], n)
    frame = pd.DataFrame(data)
    # Higher amount / longer duration -> higher default probability.
    score = 0.00008 * frame["amount"] + 0.02 * frame["duration"]
    prob = 1 / (1 + np.exp(-(score - score.mean())))
    frame["Default"] = (rng.uniform(0, 1, n) < prob).astype(int)
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
