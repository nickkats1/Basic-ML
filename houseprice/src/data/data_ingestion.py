"""Raw data ingestion for the house-price project.

Reads the committed local CSV when present, otherwise falls back to the
configured URL.
"""

from pathlib import Path

import pandas as pd

from helpers.config import Config, get_config
from helpers.logger import logger


def fetch_raw_data(config: Config | None = None) -> pd.DataFrame:
    """Load the raw house-price dataset.

    Returns:
        The raw dataset.

    Raises:
        Exception: If neither the local file nor the URL can be read.
    """
    config = config or get_config()
    try:
        source = config.raw_path if Path(config.raw_path).exists() else config.url_link
        logger.info("reading house-price data from %s", source)
        data = pd.read_csv(source)
        logger.info("raw dataset shape: %s", data.shape)
        return data
    except Exception:
        logger.exception("Failed to load house-price data")
        raise
