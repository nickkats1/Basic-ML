import numpy as np
import pandas as pd

from src.credit.config import Cfg
from src.credit.preprocessing import build_preprocessor


class TestPreprocessing:
    def test_scales_numeric_and_encodes_categories(self, creditrisk_raw_data):
        X = creditrisk_raw_data[list(Cfg.features)]
        preprocessor = build_preprocessor(X)

        transformed = preprocessor.fit_transform(X)

        numeric = X.select_dtypes(include="number")
        categorical = X.select_dtypes(exclude="number")

        scaled = preprocessor.named_transformers_["numeric"].transform(numeric)

        np.testing.assert_allclose(scaled.mean(axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(scaled.std(axis=0), 1, atol=1e-10)

        expected_columns = numeric.shape[1] + categorical.nunique().sum()
        assert transformed.shape == (len(X), expected_columns)

    def test_handles_unknown_categories(self, creditrisk_raw_data):
        X = creditrisk_raw_data[list(Cfg.features)]
        preprocessor = build_preprocessor(X)
        preprocessor.fit(X)

        new_data = X.iloc[[0]].copy()
        new_data["purpose"] = "unknown"

        transformed = preprocessor.transform(new_data)

        assert transformed.shape[0] == 1

    def test_does_not_mutate_input(self, creditrisk_raw_data):
        X = creditrisk_raw_data[list(Cfg.features)]
        before = X.copy(deep=True)

        build_preprocessor(X).fit_transform(X)

        pd.testing.assert_frame_equal(X, before)