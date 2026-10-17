from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


class Cfg:
    data_path: str = str(_ROOT / "data" / "hmda.txt")

    COLUMN_NAMES: dict[str, str] = {
        "s6": "loan_amount",
        "s7": "denied",
        "s11": "county",
        "s15": "sex",
        "s17": "applicant_income",
        "s45": "expense_to_income",  # housing expense / income (bank calc)
        "s46": "debt_to_income",  # total obligations / income (bank calc)
        "s47": "fixed_rate",  # 1 = adjustable, 2 = fixed, 3 = not available
        "netw": "net_worth",
        "s35": "liquid_assets",
        "s43": "chcp",  # credit history: consumer payments
        "s42": "chmp",  # credit history: mortgage payments
        "s44": "public_record_history",
        "uria": "industry_unemployment_rate",
        "rtdum": "rtdum",  # neighbourhood minority share: 1 = more than 30%, 0 = 30% or less
        "bd": "bd",  # boarded-up buildings above area median (1/0)
        "mi": "median_tract_income",  # median tract income above area median (1/0)
        "vr": "tract_vacancy",  # vacant homes above area median (1/0)
        "chval": "tract_value_change",  # % change in neighbourhood home values, 1980-1990
        "s27a": "self_employed",
        "s50": "appraised_value",
        "s51": "property_type",  # 1 = condominium, 2 = single family, 3 = 2 to 4 families
        "s53": "pmi_denied",
        "s13": "race",
        "s52": "pmi_sought",
        "s40": "credit_meets_policy",
        "s56": "unverifiable_information",
        "s23a": "marital_status",
        "s24a": "dependents",
        "s25a": "years_employed",
        "s26a": "years_on_job",
        "s32": "housing_expense",
        "school": "education",
    }

    features: list[str] = [
        "loan_amount",
        "appraised_value",
        "property_type",
        "county",
        "sex",
        "applicant_income",
        "expense_to_income",
        "debt_to_income",
        "fixed_rate",
        "net_worth",
        "liquid_assets",
        "chcp",
        "chmp",
        "public_record_history",
        "industry_unemployment_rate",
        "rtdum",
        "bd",
        "median_tract_income",
        "tract_vacancy",
        "tract_value_change",
        "self_employed",
        "pmi_denied",
        "race",
        "pmi_sought",
        "credit_meets_policy",
        "unverifiable_information",
        "marital_status",
        "dependents",
        "years_employed",
        "years_on_job",
        "housing_expense",
        "education",
    ]
    target: str = "denied"

    test_size: float = 0.2
    seed: int = 42
    cv_folds: int = 4
