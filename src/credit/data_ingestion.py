import pandas as pd

from src.credit.config import Cfg


class DataIngestion:
    def load_all(self) -> pd.DataFrame:
        return pd.read_csv(Cfg.url)