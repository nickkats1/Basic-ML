import numpy as np
import pandas as pd
import pytest

from src.airplane.config import Cfg
from src.airplane.preprocessing import clean


class TestPreprocessing:
    def test_keeps_features_and_target(self, airplane_raw_data):
        cleaned = clean(airplane_raw_data)

        assert list(cleaned.columns) == [*Cfg.features, Cfg.target]

    def test_transforms_only_log_features(self, airplane_raw_data):
        cleaned = clean(airplane_raw_data)

        for column in Cfg.log_features:
            np.testing.assert_allclose(
                cleaned[column], np.log1p(airplane_raw_data[column])
            )

        unchanged = [
            column for column in [*Cfg.features, Cfg.target]
            if column not in Cfg.log_features
        ]
        pd.testing.assert_frame_equal(
            cleaned[unchanged], airplane_raw_data[unchanged]
        )

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_drops_invalid_rows_and_resets_index(self, airplane_raw_data, value):
        raw = airplane_raw_data.copy()
        raw["age"] = raw["age"].astype(float)
        raw.loc[1, "age"] = value

        cleaned = clean(raw)

        assert cleaned["price"].tolist() == [120000, 60000]
        assert cleaned.index.tolist() == [0, 1]

    def test_does_not_mutate_input(self, airplane_raw_data):
        before = airplane_raw_data.copy(deep=True)

        clean(airplane_raw_data)

        pd.testing.assert_frame_equal(airplane_raw_data, before)
