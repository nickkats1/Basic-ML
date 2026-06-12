"""Unit tests for the airplane training/inference pipeline."""

import numpy as np
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
    assert 0 < cfg.test_size < 1


def test_get_config_validates():
    with pytest.raises(ValueError):
        get_config(test_size=1.5)


def test_clean_drops_nan_and_duplicates(raw_data):
    cfg = get_config()
    dirty = raw_data.copy()
    dirty.loc[0, cfg.features[0]] = np.nan
    cleaned = clean(dirty, cfg)
    assert not cleaned.isna().any().any()
    assert list(cleaned.columns) == [*cfg.features, cfg.target]


def test_clean_raises_when_empty(raw_data):
    cfg = get_config()
    empty = raw_data.iloc[0:0]
    with pytest.raises(ValueError):
        clean(empty, cfg)


def test_split_is_aligned(raw_data):
    cfg = get_config()
    data = clean(raw_data, cfg)
    X_train, X_test, y_train, y_test = split(data, cfg)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert list(X_train.columns) == list(cfg.features)


def test_preprocessor_fits_and_transforms(raw_data):
    cfg = get_config()
    data = clean(raw_data, cfg)
    X_train, _, _, _ = split(data, cfg)
    pre = build_preprocessor(X_train, cfg)
    transformed = pre.fit_transform(X_train)
    assert transformed.shape[0] == len(X_train)


def test_training_persists_pipeline(trained):
    from pathlib import Path

    cfg: Config = trained["config"]
    assert Path(cfg.model_path).exists()
    assert trained["summary"]["best_model"] is not None
    assert trained["summary"]["best_r2"] > 0.5  # learnable synthetic signal


def test_evaluate_returns_metrics(trained):
    metrics = evaluate(trained["X_test"], trained["y_test"], trained["config"])
    assert {"r2", "rmse", "mae", "n_test"} <= metrics.keys()
    assert metrics["n_test"] == len(trained["y_test"])


def test_predict_returns_float(trained):
    cfg = trained["config"]
    row = trained["X_test"].iloc[0].to_dict()
    prediction = predict(row, cfg)
    assert isinstance(prediction, float)
