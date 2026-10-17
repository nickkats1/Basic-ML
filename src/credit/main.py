from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.credit.config import Cfg
from src.credit.data_ingestion import DataIngestion
from src.credit.metrics import eval_metrics
from src.credit.model_registry import MODEL_REGISTRY
from src.credit.preprocessing import build_preprocessor


def main():
    df = DataIngestion().load_all()
    X = df[list(Cfg.features)]
    y = df[Cfg.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Cfg.test_size, random_state=Cfg.seed, stratify=y
    )

    searches = {}

    for name, settings in MODEL_REGISTRY.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("model", settings["model"]),
        ])
        params = {
            f"model__{key}": values
            for key, values in settings["params"].items()
        }

        search = GridSearchCV(
            pipeline, params, cv=Cfg.cv_folds, scoring="roc_auc", n_jobs=-1
        )
        search.fit(X_train, y_train)
        searches[name] = search

        print(f"{name}: CV AUC = {search.best_score_:.4f}")

    best_name = max(searches, key=lambda name: searches[name].best_score_)
    best_model = searches[best_name]

    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]

    print(f"\nBest model: {best_name}")
    print(eval_metrics(y_test, predictions, probabilities))


if __name__ == "__main__":
    main()