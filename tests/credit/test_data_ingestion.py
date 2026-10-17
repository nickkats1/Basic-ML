from unittest.mock import patch

from src.credit.config import Cfg
from src.credit.data_ingestion import DataIngestion


class TestDataIngestion:
    def test_load_all(self, creditrisk_raw_data):
        with patch(
            "src.credit.data_ingestion.pd.read_csv",
            return_value=creditrisk_raw_data,
        ) as mock_read_csv:
            DataIngestion().load_all()

        mock_read_csv.assert_called_once_with(Cfg.url)
