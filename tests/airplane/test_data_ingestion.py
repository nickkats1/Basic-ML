from unittest.mock import call, patch

import pandas as pd
import pytest

from src.airplane.config import Cfg
from src.airplane.data_ingestion import DataIngestion


class TestDataIngestion:
    def test_loads_sales(self):
        sales_raw = pd.DataFrame({"price": [100_000, 150_000]})

        with patch(
            "src.airplane.data_ingestion.pd.read_csv",
            return_value=sales_raw,
        ) as mock_read_csv:
            sales_df = DataIngestion().load_sales()

        mock_read_csv.assert_called_once_with(Cfg.sales_url)
        pd.testing.assert_frame_equal(sales_df, sales_raw)

    def test_loads_specs(self):
        specs_raw = pd.DataFrame({"horse": [150, 200]})

        with patch(
            "src.airplane.data_ingestion.pd.read_csv",
            return_value=specs_raw,
        ) as mock_read_csv:
            specs_df = DataIngestion().load_specs()

        mock_read_csv.assert_called_once_with(Cfg.specs_url)
        pd.testing.assert_frame_equal(specs_df, specs_raw)

    def test_loads_performance(self):
        performance_raw = pd.DataFrame({"cruise": [120, 160]})

        with patch(
            "src.airplane.data_ingestion.pd.read_csv",
            return_value=performance_raw,
        ) as mock_read_csv:
            performance_df = DataIngestion().load_performance()

        mock_read_csv.assert_called_once_with(Cfg.performance_url)
        pd.testing.assert_frame_equal(performance_df, performance_raw)

    def test_load_all_aligns_sources_by_sale_id(self):
        sales = pd.DataFrame({"0Sale_ID": [1, 2], "price": [100_000, 150_000]})
        specs = pd.DataFrame({"0Sale_ID": [2, 1], "pass": [6, 4]})
        performance = pd.DataFrame({"0Sale_ID": [2, 1], "horse": [200, 150]})
        expected = pd.DataFrame({
            "0Sale_ID": [1, 2],
            "price": [100_000, 150_000],
            "pass": [4, 6],
            "horse": [150, 200],
        })

        with patch(
            "src.airplane.data_ingestion.pd.read_csv",
            side_effect=[sales, specs, performance],
        ) as mock_read_csv:
            combined = DataIngestion().load_all()

        assert mock_read_csv.call_args_list == [
            call(Cfg.sales_url), call(Cfg.specs_url), call(Cfg.performance_url)
        ]
        pd.testing.assert_frame_equal(combined, expected)

    def test_load_all_rejects_duplicate_sale_ids(self):
        sales = pd.DataFrame({"0Sale_ID": [1, 2], "price": [100_000, 150_000]})
        specs = pd.DataFrame({"0Sale_ID": [1, 1], "pass": [4, 6]})
        performance = pd.DataFrame({"0Sale_ID": [1, 2], "horse": [150, 200]})

        with patch(
            "src.airplane.data_ingestion.pd.read_csv",
            side_effect=[sales, specs, performance],
        ), pytest.raises(pd.errors.MergeError):
            DataIngestion().load_all()
