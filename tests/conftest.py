import pandas as pd
import pytest


@pytest.fixture
def airplane_raw_data():
    return pd.DataFrame({
        "0Sale_ID": [1, 2, 3],
        "age": [5, 10, 20],
        "pass": [4, 6, 2],
        "fixgear": [1, 0, 1],
        "tdrag": [0, 0, 1],
        "horse": [180, 250, 120],
        "fuel": [50, 80, 30],
        "ceiling": [15000, 20000, 12000],
        "cruise": [130, 180, 100],
        "price": [120000, 180000, 60000],
    })


@pytest.fixture
def utilization_raw_data():
    return pd.DataFrame({
        "app_id": [1, 2, 3],
        "ssn": [100000001, 100000002, 100000003],
        "zip_code": [10001, 10002, 10003],
        "income": [40000, 60000, 80000],
        "homeownership": ["Rent", "Own", "Rent"],
        "purchases": [1000, 2000, 3000],
        "credit_limit": [6000, 10000, 15000],
        "fico": [620, 700, 780],
        "num_late": [2, 0, 1],
        "past_def": [1, 0, 0],
        "num_bankruptcy": [0, 0, 0],
        "avg_income": [45000, 65000, 75000],
        "density": [20, 500, 800],
    })


@pytest.fixture
def creditrisk_raw_data():
    return pd.DataFrame({
        "duration": [12, 24, 36],
        "amount": [2000, 5000, 10000],
        "installment": [2, 3, 4],
        "residence": [1, 2, 4],
        "age": [25, 40, 55],
        "cards": [1, 2, 1],
        "liable": [1, 1, 2],
        "checkingstatus1": ["A11", "A12", "A14"],
        "history": ["A32", "A33", "A34"],
        "purpose": ["A40", "A41", "A42"],
        "savings": ["A61", "A62", "A63"],
        "employ": ["A72", "A73", "A74"],
        "status": ["A91", "A92", "A93"],
        "others": ["A101", "A102", "A103"],
        "property": ["A121", "A122", "A123"],
        "otherplans": ["A141", "A142", "A143"],
        "housing": ["A151", "A152", "A153"],
        "job": ["A172", "A173", "A174"],
        "tele": ["A191", "A192", "A191"],
        "foreign": ["A201", "A202", "A201"],
        "Default": [0, 1, 0],
    })