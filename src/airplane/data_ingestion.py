import pandas as pd

from src.airplane.config import Cfg


class DataIngestion:
    def load_sales(self) -> pd.DataFrame:
        return pd.read_csv(Cfg.sales_url)

    def load_specs(self) -> pd.DataFrame:
        return pd.read_csv(Cfg.specs_url)

    def load_performance(self) -> pd.DataFrame:
        return pd.read_csv(Cfg.performance_url)

    def load_all(self) -> pd.DataFrame:
        sales_df = self.load_sales()
        specs_df = self.load_specs()
        performance_df = self.load_performance()

        return (
            sales_df.merge(specs_df, on="0Sale_ID", validate="one_to_one")
            .merge(performance_df, on="0Sale_ID", validate="one_to_one")
        )
