import pandas as pd


class DataIngestion:
    def load(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)




