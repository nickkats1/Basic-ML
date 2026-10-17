from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utilization.config import Cfg
from src.utilization.data_ingestion import DataIngestion
from src.utilization.metrics import eval_metrics
from src.utilization.model_registry import MODEL_REGISTRY
from src.utilization.preprocessing import clean


def run():
    config = Cfg()

    # Data ingestion and cleaning
    data_ingestion = DataIngestion()
    df = clean(data_ingestion.load_all())

    x = df.drop(columns=config.target)
    y = df[config.target]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=config.seed,
    )

    # Cross-validation and parameter tuning
    searches = {}

    for name, settings in MODEL_REGISTRY.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", settings["model"]),
        ])

        params = {
            f"model__{key}": values
            for key, values in settings["params"].items()
        }

        search = GridSearchCV(
            pipeline,
            params,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        search.fit(x_train, y_train)
        searches[name] = search

        print(f"{name}: CV MSE={-search.best_score_:.4f}")

    # Select using CV scores, then evaluate on the test set
    best_name = max(searches, key=lambda name: searches[name].best_score_)
    best_search = searches[best_name]

    y_pred = best_search.predict(x_test)
    metrics = eval_metrics(y_test, y_pred)

    print(f"\nBest model: {best_name}")
    print(f"Best parameters: {best_search.best_params_}")
    print(f"Test metrics: {metrics}")


if __name__ == "__main__":
    run()
    
    
    
    
    
    