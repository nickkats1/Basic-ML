from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.credit.config import Cfg

MODEL_REGISTRY = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=5000, random_state=Cfg.seed),
        "params": {
            "C": [0.01, 0.1, 1.0, 10.0],
        },
    },
    "DecisionTreeClassifier": {
        "model": DecisionTreeClassifier(random_state=Cfg.seed),
        "params": {
            "max_depth": [None, 5, 10],
            "min_samples_leaf": [1, 2, 5],
        },
    },
    "RandomForestClassifier": {
        "model": RandomForestClassifier(random_state=Cfg.seed),
        "params": {
            "n_estimators": [400],
            "max_depth": [None, 5, 10],
            "min_samples_leaf": [2, 5, 10],
            "max_features": ["sqrt", 0.5],
        },
    },
    "XGBClassifier": {
        "model": XGBClassifier(
            random_state=Cfg.seed,
            eval_metric="logloss",
            n_jobs=1,
        ),
        "params": {
            "n_estimators": [200, 400],
            "max_depth": [1, 2, 3],
            "learning_rate": [0.03, 0.05],
            "min_child_weight": [3, 10],
            "reg_lambda": [1, 10],
        },
    },
}
