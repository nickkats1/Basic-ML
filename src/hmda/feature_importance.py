import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.hmda.config import Cfg
from src.hmda.data_ingestion import clean, load_data
from src.hmda.metrics import eval_metrics


def main() -> None:
    df = clean(load_data(Cfg.data_path))
    X = df[Cfg.features]
    y = df[Cfg.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=Cfg.test_size,
        random_state=Cfg.seed,
        stratify=y,
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        min_child_weight=3,
        reg_lambda=10,
        eval_metric="logloss",
        random_state=Cfg.seed,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    print(eval_metrics(y_test, predictions, probabilities))

    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values()
    importance.plot.barh(figsize=(8, 9), color="#318A9A")
    plt.title("XGBoost feature importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
