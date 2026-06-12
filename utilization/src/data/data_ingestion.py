"""Raw data ingestion for the credit-utilization project."""

import pandas as pd

from helpers.config import Config, get_config
from helpers.logger import logger


def fetch_raw_data(config: Config | None = None) -> pd.DataFrame:
    """Fetch and combine the credit bureau, applications, and demographic data.

    Returns:
        The combined raw dataset.

    Raises:
        Exception: If any source cannot be read or combined.
    """
    config = config or get_config()
    try:
        credit = pd.read_csv(config.credit_url)
        applications = pd.read_csv(config.app_url)
        demographic = pd.read_csv(config.demo_url)

        logger.info("credit bureau shape: %s", credit.shape)
        logger.info("applications shape: %s", applications.shape)
        logger.info("demographic shape: %s", demographic.shape)

        data = pd.concat([credit, applications, demographic], axis=1)
        data = data.loc[:, ~data.columns.duplicated()].copy()

        logger.info("combined raw dataset shape: %s", data.shape)
        return data
    except Exception:
        logger.exception("Failed to fetch raw utilization data")
        raise
