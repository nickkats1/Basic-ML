"""Raw data ingestion for the airplane regression project."""

import pandas as pd

from helpers.config import Config, get_config
from helpers.logger import logger


def fetch_raw_data(config: Config | None = None) -> pd.DataFrame:
    """Fetch the sales, specs, and performance CSVs and combine them.

    Args:
        config: Project configuration. Defaults to :func:`get_config`.

    Returns:
        The combined raw dataset.

    Raises:
        Exception: If any source cannot be read or combined.
    """
    config = config or get_config()
    try:
        sales = pd.read_csv(config.airplane_sales_link)
        specs = pd.read_csv(config.airplane_specs_link)
        perf = pd.read_csv(config.airplane_perf_link)

        logger.info("airplane sales shape: %s", sales.shape)
        logger.info("airplane specs shape: %s", specs.shape)
        logger.info("airplane performance shape: %s", perf.shape)

        data = pd.concat([sales, specs, perf], axis=1)
        data = data.loc[:, ~data.columns.duplicated()].copy()

        logger.info("combined raw dataset shape: %s", data.shape)
        return data
    except Exception:
        logger.exception("Failed to fetch raw airplane data")
        raise
