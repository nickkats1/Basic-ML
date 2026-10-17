import pandas as pd

from src.utilization.config import Cfg


class DataIngestion:
    def load_app(self) -> pd.DataFrame:
        return pd.read_csv(Cfg.app_url)

    def load_credit(self) -> pd.DataFrame:
        return pd.read_csv(Cfg.credit_url)

    def load_demographic(self) -> pd.DataFrame:
        return pd.read_csv(Cfg.demo_url)

    def load_all(self) -> pd.DataFrame:
        app_df = self.load_app()
        credit_df = self.load_credit()
        demographic_df = self.load_demographic()

        return (
            app_df
            .merge(
                credit_df,
                on=["ssn", "zip_code"],
                how="left",
                validate="many_to_one",
            )
            .merge(
                demographic_df,
                on="zip_code",
                how="left",
                validate="many_to_one",
            )
        )