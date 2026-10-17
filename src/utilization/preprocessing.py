import numpy as np
import pandas as pd

from src.utilization.config import Cfg


def clean(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    utilization = data["purchases"] / data["credit_limit"]
    data[Cfg.target] = np.log(utilization / (1 - utilization))
    data["homeownership"] = (data["homeownership"] == "Rent").astype(int)

    columns = [*Cfg.features, Cfg.target]
    data = data.loc[:, columns].replace([np.inf, -np.inf], np.nan)
    data = data.dropna().drop_duplicates().reset_index(drop=True)

    if data.empty:
        raise ValueError("No rows remain after cleaning the dataset.")

    return data