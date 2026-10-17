from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

from src.prices.config import Cfg
from src.prices.data_ingestion import DataIngestion
from src.prices.metrics import eval_metrics
from src.prices.model_registry import MODEL_REGISTRY


def main() -> None:
    df = DataIngestion().load(Cfg.data_path)
    X = df[list(Cfg.features)]
    y = df[Cfg.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=Cfg.test_size,
        random_state=Cfg.random_state,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    searches = {}

    for name, settings in MODEL_REGISTRY.items():
        search = RandomizedSearchCV(
            settings["model"],
            settings["params"],
            n_iter=min(Cfg.n_iter, len(ParameterGrid(settings["params"]))),
            cv=Cfg.cv_folds,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            random_state=Cfg.random_state,
        )
        search.fit(X_train, y_train)
        searches[name] = search

        print(f"{name}: CV MSE = {-search.best_score_:.4f}")

    best_name = max(searches, key=lambda name: searches[name].best_score_)
    best_model = searches[best_name]
    predictions = best_model.predict(X_test)

    print(f"\nBest model: {best_name}")
    print(f"Best parameters: {best_model.best_params_}")
    print(eval_metrics(y_test, predictions))


if __name__ == "__main__":
    main()