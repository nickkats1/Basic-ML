"""Feature engineering / data cleaning for the credit-utilization project.

Derives the regression target (``log_odds_utils``) and encodes
``homeownership``, then performs row-level cleaning. Column scaling/encoding
lives inside the persisted sklearn pipeline (see ``data_transformation``).
"""

import numpy as np
import pandas as pd

from helpers.config import Config, get_config
from helpers.logger import logger


def clean(data: pd.DataFrame, config: Config | None = None) -> pd.DataFrame:
    """Derive the target, encode categoricals, and clean rows.

    Args:
        data: The raw combined dataset.
        config: Project configuration. Defaults to :func:`get_config`.

    Returns:
        A cleaned frame containing the configured features and target.

    Raises:
        ValueError: If no rows remain after cleaning.
    """
    config = config or get_config()
    data = data.copy()

    # Utilization ratio and its log-odds transform (the target).
    utility = data["purchases"] / data["credit_limit"]
    data["log_odds_utils"] = np.log(utility) / (utility - 1)

    # Encode homeownership as a binary "rents" indicator.
    data["homeownership"] = (data["homeownership"] == "Rent").astype(int)

    columns = [*config.features, config.target]
    cleaned = data.loc[:, columns].replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna().drop_duplicates().reset_index(drop=True)

    logger.info("cleaned dataset shape: %s", cleaned.shape)
    if cleaned.empty:
        raise ValueError("No rows remain after cleaning the dataset.")
    return cleaned
