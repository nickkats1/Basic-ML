from sklearn.ensemble import (
    BaggingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

MODEL_REGISTRY = {
    "LinearRegression": {
        "model": LinearRegression(),
        "params": {
            "fit_intercept": [True, False],
            "positive": [True, False],
        },
    },
    "Lasso": {
        "model": Lasso(max_iter=10_000),
        "params": {
            "alpha": [0.001, 0.01, 0.1, 1.0, 5.0, 10.0],
        },
    },
    "Ridge": {
        "model": Ridge(),
        "params": {
            "alpha": [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0],
        },
    },
    "DecisionTreeRegressor": {
        "model": DecisionTreeRegressor(random_state=42),
        "params": {
            "max_depth": [None, 5, 10, 15],
            "min_samples_leaf": [1, 2, 5],
        },
    },
    "RandomForestRegressor": {
        "model": RandomForestRegressor(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
        },
    },
    "GradientBoostingRegressor": {
        "model": GradientBoostingRegressor(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
        },
    },
    "BaggingRegressor": {
        "model": BaggingRegressor(random_state=42),
        "params": {
            "n_estimators": [50, 100],
            "max_samples": [0.8, 1.0],
        },
    },
    "XGBRegressor": {
        "model": XGBRegressor(random_state=42, n_jobs=-1),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1],
        },
    },
}