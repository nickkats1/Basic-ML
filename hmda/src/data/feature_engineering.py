"""Feature engineering / data cleaning for the HMDA loan-approval project.

Renames the raw survey columns (``sNN``) to readable names, derives the binary
approval target, encodes a few categorical fields, then performs row-level
cleaning. Column scaling lives inside the persisted sklearn pipeline.
"""

import numpy as np
import pandas as pd

from helpers.config import Config, get_config
from helpers.logger import logger

# Raw survey code -> readable name (subset needed for features + target).
_RENAME = {
    "s5": "occupancy",
    "s7": "approved",
    "s13": "race",
    "s15": "sex",
    "s17": "income",
    "s23a": "married",
    "s40": "credit_history",
    "s43": "chcp",
    "s44": "chpr",
    "s45": "debt_to_expense",
    "s46": "di_ratio",
    "s52": "pmi_sought",
    "s53": "pmi_denied",
    "s56": "unverifiable",
}


def clean(data: pd.DataFrame, config: Config | None = None) -> pd.DataFrame:
    """Rename, derive the target, encode categoricals, and clean rows.

    Args:
        data: The raw tab-delimited HMDA dataset.
        config: Project configuration. Defaults to :func:`get_config`.

    Returns:
        A cleaned, numeric frame with the configured features and target.

    Raises:
        ValueError: If no rows remain after cleaning.
    """
    config = config or get_config()
    data = data.rename(columns=_RENAME).copy()

    # Approval == survey code 3; binary encodings per the original study.
    data["approved"] = (data["approved"] == 3).astype(int)
    data["race"] = (data["race"] != 3).astype(int)
    data["married"] = (data["married"] == "M").astype(int)
    data["sex"] = (data["sex"] == 1).astype(int)
    data["credit_history"] = (data["credit_history"] == 1).astype(int)

    columns = [*config.features, config.target]
    cleaned = data.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.dropna().drop_duplicates().reset_index(drop=True)

    logger.info("cleaned dataset shape: %s", cleaned.shape)
    if cleaned.empty:
        raise ValueError("No rows remain after cleaning the dataset.")
    return cleaned
