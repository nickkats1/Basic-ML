from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.hmda.config import Cfg
from src.hmda.data_ingestion import clean, load_data
from src.hmda.metrics import eval_metrics
from src.hmda.model_registry import MODEL_REGISTRY


def main() -> None:
    config = Cfg()
    df = clean(load_data(config.data_path))
    X = df[list(config.features)]
    y = df[config.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=y,
    )

    searches = {}

    for name, settings in MODEL_REGISTRY.items():
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", settings["model"]),
            ]
        )
        params = {f"model__{key}": values for key, values in settings["params"].items()}
        search = GridSearchCV(
            pipeline, params, cv=config.cv_folds, scoring="roc_auc", n_jobs=-1
        )
        search.fit(X_train, y_train)
        searches[name] = search

        print(f"{name}: CV AUC = {search.best_score_:.4f}")

    best_name = max(searches, key=lambda name: searches[name].best_score_)
    best_model = searches[best_name]

    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]

    print(f"\nBest model: {best_name}")
    print(f"Best parameters: {best_model.best_params_}")
    print(eval_metrics(y_test, predictions, probabilities))


def run() -> None:
    main()


if __name__ == "__main__":
    main()
