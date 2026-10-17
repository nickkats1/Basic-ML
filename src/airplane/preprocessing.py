import numpy as np
import pandas as pd

from src.airplane.config import Cfg


def clean(data: pd.DataFrame) -> pd.DataFrame:
    data = data.loc[:, [*Cfg.features, Cfg.target]].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    for column in Cfg.log_features:
        data[column] = np.log1p(data[column])

    return data.reset_index(drop=True)
