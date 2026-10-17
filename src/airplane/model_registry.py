from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.airplane.config import Cfg

MODEL_REGISTRY = {
    "LinearRegression": {
        "model": LinearRegression(),
        "params": {
            "fit_intercept": [True, False],
            "positive": [True, False],
        },
    },
    "Ridge": {
        "model": Ridge(random_state=Cfg.seed),
        "params": {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
            "solver": ["auto", "svd", "cholesky", "lsqr"],
        },
    },
    "Lasso": {
        "model": Lasso(random_state=Cfg.seed, max_iter=10_000),
        "params": {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            "selection": ["cyclic", "random"],
        },
    },
    "KNeighborsRegressor": {
        "model": KNeighborsRegressor(n_jobs=1),
        "params": {
            "n_neighbors": [3, 5, 7, 10, 15, 25, 40],
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },
    },
    "DecisionTreeRegressor": {
        "model": DecisionTreeRegressor(random_state=Cfg.seed),
        "params": {
            "criterion": ["squared_error", "friedman_mse"],
            "max_depth": [4, 6, 8, 12, 20, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 5, 10],
            "max_features": [None, "sqrt", "log2"],
        },
    },
    "RandomForestRegressor": {
        "model": RandomForestRegressor(
            random_state=Cfg.seed,
            n_jobs=1,
        ),
        "params": {
            "n_estimators": [300, 600, 1000],
            "max_depth": [None, 8, 12, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": [1.0, 0.7, "sqrt"],
            "bootstrap": [True],
        },
    },
    "XGBRegressor": {
        "model": XGBRegressor(
            random_state=Cfg.seed,
            verbosity=0,
            n_jobs=1,
            tree_method="hist",
            objective="reg:squarederror",
        ),
        "params": {
            "n_estimators": [400, 800, 1500],
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "min_child_weight": [1, 5, 10],
            "reg_alpha": [0.0, 0.1, 1.0],
            "reg_lambda": [1.0, 5.0, 10.0],
        },
    },
    "LGBMRegressor": {
        "model": LGBMRegressor(
            random_state=Cfg.seed,
            verbose=-1,
            n_jobs=1,
            objective="regression",
        ),
        "params": {
            "n_estimators": [400, 800, 1500],
            "max_depth": [-1, 6, 10, 15],
            "num_leaves": [15, 31, 63, 127],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "subsample_freq": [1, 5],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "min_child_samples": [5, 20, 50],
            "reg_alpha": [0.0, 0.1, 1.0],
            "reg_lambda": [0.0, 0.1, 1.0],
        },
    },
}