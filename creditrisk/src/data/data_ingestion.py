"""Raw data ingestion for the credit-risk project."""

import pandas as pd

from helpers.config import Config, get_config
from helpers.logger import logger


def fetch_raw_data(config: Config | None = None) -> pd.DataFrame:
    """Fetch the credit-risk dataset from the configured URL.

    Returns:
        The raw dataset.

    Raises:
        Exception: If the source cannot be read.
    """
    config = config or get_config()
    try:
        data = pd.read_csv(config.url_link)
        logger.info("raw dataset shape: %s", data.shape)
        return data
    except Exception:
        logger.exception("Failed to fetch raw credit-risk data")
        raise
