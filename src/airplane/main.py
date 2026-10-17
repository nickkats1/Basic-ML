import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split

from src.airplane.config import Cfg
from src.airplane.data_ingestion import DataIngestion
from src.airplane.metrics import evaluate
from src.airplane.model_registry import MODEL_REGISTRY
from src.airplane.preprocessing import clean


def main() -> pd.DataFrame:
    df = clean(DataIngestion().load_all())
    X = df[list(Cfg.features)]
    y = df[Cfg.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Cfg.test_size, random_state=Cfg.seed
    )

    results = []

    for name, settings in MODEL_REGISTRY.items():
        search = GridSearchCV(
            settings["model"],
            settings["params"],
            cv=Cfg.cv_folds,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        predictions = search.best_estimator_.predict(X_test)

        results.append({
            "model": name,
            "best_params": search.best_params_,
            "CV_MSE": -search.best_score_,
            **evaluate(y_test, predictions),
        })

    results = (
        pd.DataFrame(results)
        .sort_values("CV_MSE")
        .reset_index(drop=True)
    )

    print(results.to_string(index=False))
    return results


if __name__ == "__main__":
    main()