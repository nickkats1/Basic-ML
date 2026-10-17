from unittest.mock import patch

import pandas as pd

from src.utilization.config import Cfg
from src.utilization.data_ingestion import DataIngestion


class TestDataIngestion:
    def test_loads_app(self, utilization_raw_data):
        """load_app returns the app columns defined in Cfg"""
        app_raw = utilization_raw_data[Cfg.app_features]

        with patch(
            "src.utilization.data_ingestion.pd.read_csv", return_value=app_raw
        ) as mock_read_csv:
            app_df = DataIngestion().load_app()

        mock_read_csv.assert_called_once_with(Cfg.app_url)
        assert list(app_df.columns) == Cfg.app_features
        
    def test_loads_credit(self, utilization_raw_data):
        """load_credit returns the credit columns defined in Cfg"""
        
        credit_raw = utilization_raw_data[Cfg.credit_features]
        
        with patch(
            "src.utilization.data_ingestion.pd.read_csv", return_value=credit_raw
        ) as mock_read_csv:
            credit_df = DataIngestion().load_credit()
            mock_read_csv.assert_called_once_with(Cfg.credit_url)
            assert list(credit_df.columns) == Cfg.credit_features
    
    def test_loads_demo(self, utilization_raw_data):
        """load_demo returns the demo columns defined in Cfg"""
        
        demo_raw = utilization_raw_data[Cfg.demographic_features]
        
        with patch(
            "src.utilization.data_ingestion.pd.read_csv", return_value=demo_raw
        ) as mock_read_csv:
            demo_df = DataIngestion().load_demographic()
            mock_read_csv.assert_called_once_with(Cfg.demo_url)
            assert list(demo_df.columns) == Cfg.demographic_features

    def test_load_all_merges_all_sources(self, utilization_raw_data):
        """load_all joins app, credit and demo back into the full table"""
        app_raw = utilization_raw_data[Cfg.app_features]
        credit_raw = utilization_raw_data[Cfg.credit_features]
        demo_raw = utilization_raw_data[Cfg.demographic_features]

        with (
            patch.object(DataIngestion, "load_app", return_value=app_raw) as mock_app,
            patch.object(DataIngestion, "load_credit", return_value=credit_raw) as mock_credit,
            patch.object(DataIngestion, "load_demographic", return_value=demo_raw) as mock_demo,
        ):
            merged = DataIngestion().load_all()

        mock_app.assert_called_once()
        mock_credit.assert_called_once()
        mock_demo.assert_called_once()
        assert len(merged) == len(app_raw)
        assert not any(col.endswith(("_x", "_y")) for col in merged.columns)
        pd.testing.assert_frame_equal(merged, utilization_raw_data)

    def test_load_all_keeps_unmatched_apps(self, utilization_raw_data):
        """load_all left-joins, so apps without credit/demo rows are kept with NaN"""
        app_raw = utilization_raw_data[Cfg.app_features]
        credit_raw = utilization_raw_data[Cfg.credit_features].iloc[:-1]  # drop app 3
        demo_raw = utilization_raw_data[Cfg.demographic_features].iloc[1:]  # drop zip 10001

        with (
            patch.object(DataIngestion, "load_app", return_value=app_raw),
            patch.object(DataIngestion, "load_credit", return_value=credit_raw),
            patch.object(DataIngestion, "load_demographic", return_value=demo_raw),
        ):
            merged = DataIngestion().load_all()

        assert len(merged) == len(app_raw)
        assert merged.loc[merged["app_id"] == 3, "fico"].isna().all()
        assert merged.loc[merged["zip_code"] == 10001, "avg_income"].isna().all()
        assert merged["fico"].notna().sum() == 2

