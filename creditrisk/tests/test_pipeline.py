"""Unit tests for the credit-risk classification pipeline."""

import pytest

from helpers.config import Config, get_config
from src.data.data_transformation import build_preprocessor, split
from src.data.feature_engineering import clean
from src.models.model_evaluation import evaluate
from src.models.predict import predict


def test_get_config_defaults():
    cfg = get_config()
    assert isinstance(cfg, Config)
    assert cfg.target not in cfg.features


def test_get_config_validates():
    with pytest.raises(ValueError):
        get_config(cv_folds=1)


def test_clean_selects_schema(raw_data):
    cfg = get_config()
    cleaned = clean(raw_data, cfg)
    assert list(cleaned.columns) == [*cfg.features, cfg.target]
    assert not cleaned.isna().any().any()


def test_split_is_aligned(raw_data):
    cfg = get_config()
    data = clean(raw_data, cfg)
    X_train, X_test, y_train, y_test = split(data, cfg)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_preprocessor_handles_categoricals(raw_data):
    cfg = get_config()
    data = clean(raw_data, cfg)
    X_train, _, _, _ = split(data, cfg)
    pre = build_preprocessor(X_train, cfg)
    transformed = pre.fit_transform(X_train)
    # One-hot expansion yields more columns than raw features.
    assert transformed.shape[0] == len(X_train)
    assert transformed.shape[1] >= len(cfg.features)


def test_training_persists_pipeline(trained):
    from pathlib import Path

    cfg: Config = trained["config"]
    assert Path(cfg.model_path).exists()
    assert trained["summary"]["best_model"] is not None


def test_evaluate_returns_metrics(trained):
    metrics = evaluate(trained["X_test"], trained["y_test"], trained["config"])
    assert {"roc_auc", "accuracy", "f1", "n_test"} <= metrics.keys()
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_predict_returns_probability(trained):
    cfg = trained["config"]
    row = trained["X_test"].iloc[0].to_dict()
    proba = predict(row, cfg)
    assert isinstance(proba, float)
    assert 0.0 <= proba <= 1.0
