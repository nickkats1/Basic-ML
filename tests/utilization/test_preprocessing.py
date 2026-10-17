import numpy as np
import pandas as pd
import pytest

from src.utilization.config import Cfg
from src.utilization.preprocessing import clean


class TestPreProcessing:
    def test_clean_contains_target(self, utilization_raw_data):
        """clean adds the target column"""
        cleaned = clean(utilization_raw_data)

        assert Cfg.target in cleaned.columns

    def test_clean_keeps_only_features_and_target(self, utilization_raw_data):
        """clean keeps the Cfg features plus the target, in that order"""
        cleaned = clean(utilization_raw_data)

        assert list(cleaned.columns) == [*Cfg.features, Cfg.target]

    def test_target_is_log_odds_of_utilization(self, utilization_raw_data):
        """target is log(u / (1 - u)) where u = purchases / credit_limit"""
        cleaned = clean(utilization_raw_data)

        u = utilization_raw_data["purchases"] / utilization_raw_data["credit_limit"]
        expected = np.log(u / (1 - u))

        np.testing.assert_allclose(cleaned[Cfg.target], expected)

    def test_homeownership_encoded_as_rent_flag(self, utilization_raw_data):
        """homeownership becomes 1 for Rent, 0 otherwise"""
        cleaned = clean(utilization_raw_data)

        assert cleaned["homeownership"].tolist() == [1, 0, 1]

    def test_drops_rows_with_infinite_target(self, utilization_raw_data):
        """utilization of 1 gives infinite log-odds, so the row is dropped"""
        raw = utilization_raw_data.copy()
        raw.loc[0, "purchases"] = raw.loc[0, "credit_limit"]

        cleaned = clean(raw)

        assert len(cleaned) == 2
        assert np.isfinite(cleaned[Cfg.target]).all()

    def test_drops_duplicates_and_missing(self, utilization_raw_data):
        """duplicate and NaN rows are removed and the index is reset"""
        raw = pd.concat(
            [utilization_raw_data, utilization_raw_data.iloc[[0]]], ignore_index=True
        )
        raw.loc[1, "fico"] = np.nan

        cleaned = clean(raw)

        assert len(cleaned) == 2
        assert isinstance(cleaned.index, pd.RangeIndex)
        assert cleaned.index.tolist() == [0, 1]

    def test_raises_when_nothing_left(self, utilization_raw_data):
        """clean raises if every row is dropped"""
        raw = utilization_raw_data.copy()
        raw["purchases"] = raw["credit_limit"]

        with pytest.raises(ValueError, match="No rows remain"):
            clean(raw)

    def test_does_not_mutate_input(self, utilization_raw_data):
        """clean works on a copy and leaves the input untouched"""
        before = utilization_raw_data.copy()

        clean(utilization_raw_data)

        pd.testing.assert_frame_equal(utilization_raw_data, before)
