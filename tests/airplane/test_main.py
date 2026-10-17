from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.airplane.config import Cfg
from src.airplane.main import main


def test_training_uses_cleaned_features(airplane_raw_data):
    airplane_raw_data = pd.concat([airplane_raw_data] * 2, ignore_index=True)
    search = MagicMock()
    search.best_score_ = -100.0
    search.best_params_ = {}
    search.best_estimator_.predict.return_value = [120_000, 180_000]
    registry = {"example": {"model": MagicMock(), "params": {}}}

    with (
        patch("src.airplane.main.DataIngestion.load_all", return_value=airplane_raw_data),
        patch("src.airplane.main.MODEL_REGISTRY", registry),
        patch("src.airplane.main.GridSearchCV", return_value=search),
    ):
        main()

    X_train, y_train = search.fit.call_args.args
    X_test = search.best_estimator_.predict.call_args.args[0]
    features = pd.concat([X_train, X_test]).sort_index()

    assert list(features.columns) == list(Cfg.features)
    assert Cfg.target not in features.columns
    assert "0Sale_ID" not in features.columns
    for column in Cfg.log_features:
        np.testing.assert_allclose(features[column], np.log1p(airplane_raw_data[column]))
    pd.testing.assert_series_equal(y_train, airplane_raw_data.loc[y_train.index, Cfg.target])
    assert set(X_train.index).isdisjoint(X_test.index)


def test_ranks_by_cv_mse_even_when_test_rmse_disagrees(airplane_raw_data):
    airplane_raw_data = pd.concat([airplane_raw_data] * 2, ignore_index=True)
    registry = {
        "better_test": {"model": MagicMock(), "params": {}},
        "better_cv": {"model": MagicMock(), "params": {}},
    }
    searches = []
    for score, prediction in [(-100.0, 120_000), (-10.0, 0)]:
        search = MagicMock()
        search.best_score_ = score
        search.best_params_ = {}
        search.best_estimator_.predict.return_value = [prediction, prediction]
        searches.append(search)

    with (
        patch("src.airplane.main.DataIngestion.load_all", return_value=airplane_raw_data),
        patch("src.airplane.main.MODEL_REGISTRY", registry),
        patch("src.airplane.main.GridSearchCV", side_effect=searches),
    ):
        results = main()

    assert results["model"].tolist() == ["better_cv", "better_test"]
    assert results["CV_MSE"].tolist() == [10.0, 100.0]
    assert results.loc[0, "RMSE"] > results.loc[1, "RMSE"]
