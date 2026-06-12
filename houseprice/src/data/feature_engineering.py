"""Feature engineering / data cleaning for the airplane project.

Only row-level cleaning happens here (dropping NaNs/duplicates and non-finite
values). All *column transforms* (log scaling, standardization, encoding) live
inside the persisted sklearn pipeline built in :mod:`src.data.data_transformation`
so training and serving apply identical logic.
"""

import numpy as np
import pandas as pd

from helpers.config import Config, get_config
from helpers.logger import logger


def clean(data: pd.DataFrame, config: Config | None = None) -> pd.DataFrame:
    """Drop non-finite values, missing values, and duplicate rows.

    Args:
        data: The raw combined dataset.
        config: Project configuration. Defaults to :func:`get_config`.

    Returns:
        A cleaned copy retaining the configured features and target.

    Raises:
        ValueError: If no rows remain after cleaning.
    """
    config = config or get_config()
    columns = [*config.features, config.target]
    cleaned = data.loc[:, columns].copy()
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna().drop_duplicates().reset_index(drop=True)

    logger.info("cleaned dataset shape: %s", cleaned.shape)
    if cleaned.empty:
        raise ValueError("No rows remain after cleaning the dataset.")
    return cleaned
