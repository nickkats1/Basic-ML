import pandas as pd

from src.hmda.config import Cfg

CODE_MAPS: dict[str, dict] = {
    "denied": {1: 0, 2: 0, 3: 1},
    "race": {3: 1, 5: 0},
    "sex": {1: 1, 2: 0},
    "marital_status": {"M": 0, "U": 1, "S": 2},
    "fixed_rate": {2: 1, 1: 0},
    "credit_meets_policy": {0: 0, 1: 1},
}


def load_data(file_path: str) -> pd.DataFrame:
    """Load the tab-separated HMDA dataset.

    Args:
        file_path: Path to the HMDA data file.

    Returns:
        The raw dataset, with the 999999.4 missing-value code read as NaN.
    """
    return pd.read_csv(file_path, sep="\t", na_values=[999999.4])


def clean(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Rename, select, and encode the HMDA columns.

    Args:
        dataframe: Raw HMDA data.

    Returns:
        The selected features and binary denial target, with missing
        rows removed and original indexes retained.
    """
    dataframe = dataframe.rename(columns=Cfg.COLUMN_NAMES)
    dataframe = dataframe[[*Cfg.features, Cfg.target]].copy()
    for column, codes in CODE_MAPS.items():
        dataframe[column] = dataframe[column].map(codes)
    return dataframe.dropna()
